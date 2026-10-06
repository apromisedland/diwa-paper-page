"""
action_model.py

"""
from .models import DiT
from ..action_model import create_diffusion
from . import gaussian_diffusion as gd
from .respace import FMDiffusion
import torch
from torch import nn

# Create model sizes of ActionModels
def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)
def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)
def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)
def DiT_Debug(**kwargs):
    """Small CPU verification model; not the paper's reference action head."""
    return DiT(depth=2, hidden_size=64, num_heads=4, **kwargs)

# Model size
DiT_models = {'DiT-S': DiT_S, 'DiT-B': DiT_B, 'DiT-L': DiT_L, 'DiT-Debug': DiT_Debug}

# Create ActionModel
class ActionModel(nn.Module):
    def __init__(self, 
                 token_size, 
                 model_type, 
                 in_channels, 
                 future_action_window_size, 
                 past_action_window_size,
                 diffusion_steps = 100,
                 noise_schedule = 'squaredcos_cap_v2'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = noise_schedule, diffusion_steps=self.diffusion_steps, sigma_small=True, learn_sigma = False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.net = DiT_models[model_type](
                                        token_size = token_size, 
                                        in_channels=in_channels, 
                                        class_dropout_prob = 0.1, 
                                        learn_sigma = learn_sigma, 
                                        future_action_window_size = future_action_window_size, 
                                        past_action_window_size = past_action_window_size
                                        )

    def decision_response(self, x, z, noise=None, timestep=None):
        """Return the denoiser response under reusable diffusion randomness.

        DIWA uses the same ``noise`` and ``timestep`` before and after a latent
        intervention. This prevents diffusion sampling variance from being
        mistaken for decision influence.
        """
        if noise is None:
            noise = torch.randn_like(x)
        if timestep is None:
            timestep = torch.randint(
                0,
                self.diffusion.num_timesteps,
                (x.size(0),),
                device=x.device,
            )
        x_t = self.diffusion.q_sample(x, timestep, noise)
        prediction = self.net(x_t, timestep, z)
        assert prediction.shape == noise.shape == x.shape
        return prediction, noise, timestep

    def loss_per_sample(self, x, z, noise=None, timestep=None):
        noise_pred, noise, timestep = self.decision_response(
            x, z, noise=noise, timestep=timestep
        )
        loss = ((noise_pred - noise) ** 2).mean(dim=tuple(range(1, x.ndim)))
        return loss, noise, timestep

    def decision_moments(
        self,
        x,
        z,
        num_probes=4,
        noise=None,
        timestep=None,
    ):
        """Estimate local policy moments with common-random diffusion probes.

        DIWA supplies the policy's detached action proposal as ``x``. The same
        proposal, noise and timesteps are reused for every intervention.
        """
        if num_probes < 2:
            raise ValueError("decision moments require at least two probes")
        batch_size = x.shape[0]
        if noise is None:
            noise = torch.randn(
                batch_size,
                num_probes,
                *x.shape[1:],
                device=x.device,
                dtype=x.dtype,
            )
        if timestep is None:
            timestep = torch.randint(
                0,
                self.diffusion.num_timesteps,
                (batch_size, num_probes),
                device=x.device,
            )
        if noise.shape[:2] != (batch_size, num_probes):
            raise ValueError("noise bank does not match batch/probe dimensions")
        if timestep.shape != (batch_size, num_probes):
            raise ValueError("timestep bank does not match batch/probe dimensions")

        repeated_x = x[:, None].expand(-1, num_probes, *x.shape[1:]).reshape(
            batch_size * num_probes, *x.shape[1:]
        )
        repeated_z = z[:, None].expand(-1, num_probes, *z.shape[1:]).reshape(
            batch_size * num_probes, *z.shape[1:]
        )
        flat_noise = noise.reshape_as(repeated_x)
        flat_timestep = timestep.reshape(-1)
        prediction, _, _ = self.decision_response(
            repeated_x,
            repeated_z,
            noise=flat_noise,
            timestep=flat_timestep,
        )
        x_t = self.diffusion.q_sample(
            repeated_x, flat_timestep, noise=flat_noise
        )
        alphas = torch.as_tensor(
            self.diffusion.alphas_cumprod,
            device=x.device,
            dtype=x.dtype,
        )[flat_timestep]
        alpha = alphas.view(-1, *([1] * (x.ndim - 1))).clamp_min(1e-6)
        clean_estimates = (
            x_t - (1.0 - alpha).sqrt() * prediction
        ) / alpha.sqrt()
        clean_estimates = clean_estimates.view(
            batch_size, num_probes, *x.shape[1:]
        )
        return (
            clean_estimates.mean(dim=1),
            clean_estimates.var(dim=1, unbiased=False),
            noise,
            timestep,
        )

    # Given condition z and ground truth token x, compute loss
    def loss(self, x, z):
        loss, _, _ = self.loss_per_sample(x, z)
        return loss.mean()

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        if (
            not isinstance(ddim_step, int)
            or isinstance(ddim_step, bool)
            or not 1 <= ddim_step <= self.diffusion_steps
        ):
            raise ValueError(
                "ddim_step must be an integer in [1, diffusion_steps]"
            )
        self.ddim_diffusion = create_diffusion(
            timestep_respacing="ddim" + str(ddim_step),
            noise_schedule=self.noise_schedule,
            diffusion_steps=self.diffusion_steps,
            sigma_small=True,
            learn_sigma=False,
        )
        return self.ddim_diffusion
    
    
class ActionModelFM(nn.Module):
    def __init__(self, 
                 token_size, 
                 model_type, 
                 in_channels, 
                 future_action_window_size, 
                 past_action_window_size,
                 diffusion_steps = 10,
                 noise_schedule = 'squaredcos_cap_v2'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = noise_schedule, diffusion_steps=self.diffusion_steps, sigma_small=True, learn_sigma = False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.net = DiT_models[model_type](
                                        token_size = token_size, 
                                        in_channels=in_channels, 
                                        class_dropout_prob = 0.1, 
                                        learn_sigma = learn_sigma, 
                                        future_action_window_size = future_action_window_size, 
                                        past_action_window_size = past_action_window_size
                                        )

    def decision_response(self, x, z, noise=None, timestep=None):
        """Return the flow response under reusable interpolation randomness."""
        if noise is None:
            noise = torch.randn_like(x)
        if timestep is None:
            timestep = torch.randint(
                0,
                self.diffusion.num_timesteps,
                (x.size(0),),
                device=x.device,
            ).float()
            timestep = timestep / self.diffusion.num_timesteps
        x_t = (
            timestep.view(-1, 1, 1) * x
            + (1 - timestep.view(-1, 1, 1)) * noise
        )
        prediction = self.net(x_t, timestep, z)
        assert prediction.shape == noise.shape == x.shape
        return prediction, noise, timestep

    def loss_per_sample(self, x, z, noise=None, timestep=None):
        prediction, noise, timestep = self.decision_response(
            x, z, noise=noise, timestep=timestep
        )
        target = x - noise
        loss = ((prediction - target) ** 2).mean(
            dim=tuple(range(1, x.ndim))
        )
        return loss, noise, timestep

    def decision_moments(
        self,
        x,
        z,
        num_probes=4,
        noise=None,
        timestep=None,
    ):
        """Estimate local flow-policy moments with common interpolation probes."""
        if num_probes < 2:
            raise ValueError("decision moments require at least two probes")
        batch_size = x.shape[0]
        if noise is None:
            noise = torch.randn(
                batch_size,
                num_probes,
                *x.shape[1:],
                device=x.device,
                dtype=x.dtype,
            )
        if timestep is None:
            timestep = torch.rand(
                batch_size,
                num_probes,
                device=x.device,
                dtype=x.dtype,
            )
        if noise.shape[:2] != (batch_size, num_probes):
            raise ValueError("noise bank does not match batch/probe dimensions")
        if timestep.shape != (batch_size, num_probes):
            raise ValueError("timestep bank does not match batch/probe dimensions")

        repeated_x = x[:, None].expand(-1, num_probes, *x.shape[1:]).reshape(
            batch_size * num_probes, *x.shape[1:]
        )
        repeated_z = z[:, None].expand(-1, num_probes, *z.shape[1:]).reshape(
            batch_size * num_probes, *z.shape[1:]
        )
        flat_noise = noise.reshape_as(repeated_x)
        flat_timestep = timestep.reshape(-1)
        prediction, _, _ = self.decision_response(
            repeated_x,
            repeated_z,
            noise=flat_noise,
            timestep=flat_timestep,
        )
        clean_estimates = flat_noise + prediction
        clean_estimates = clean_estimates.view(
            batch_size, num_probes, *x.shape[1:]
        )
        return (
            clean_estimates.mean(dim=1),
            clean_estimates.var(dim=1, unbiased=False),
            noise,
            timestep,
        )

    # Given condition z and ground truth token x, compute loss
    def loss(self, x, z):
        loss, _, _ = self.loss_per_sample(x, z)
        return loss.mean()

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        if (
            not isinstance(ddim_step, int)
            or isinstance(ddim_step, bool)
            or not 1 <= ddim_step <= self.diffusion_steps
        ):
            raise ValueError(
                "ddim_step must be an integer in [1, diffusion_steps]"
            )
        betas = gd.get_named_beta_schedule(
            self.noise_schedule, self.diffusion_steps
        )
        self.ddim_diffusion = FMDiffusion(
            betas=betas,
            model_mean_type=gd.ModelMeanType.EPSILON,
            model_var_type=gd.ModelVarType.FIXED_SMALL,
            loss_type=gd.LossType.MSE,
            sampling_steps=ddim_step,
        )
        return self.ddim_diffusion
