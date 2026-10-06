"""Dependency-light helpers shared by DIWA dataloaders and tests."""

import bisect
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
from torch.utils.data import DataLoader, Sampler


class DistributedTaskPairBatchSampler(Sampler):
    """Pair every anchor with a different episode of the same task per rank.

    ``episode_windows`` maps episode identities to disjoint, contiguous
    ranges of dataset indices. All windows occur as anchors each epoch;
    extra partners and end padding are explicit sampling with replacement.
    Every rank yields the same number of complete, even-sized batches.
    """

    def __init__(
        self,
        episode_windows: Mapping[object, Sequence[int]],
        episode_tasks: Mapping[object, object],
        batch_size: int,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("paired batch_size must be even and >= 2")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed rank/replica count")
        if not episode_windows or set(episode_windows) != set(episode_tasks):
            raise ValueError("episode windows and task metadata must match")
        self.windows = dict(episode_windows)
        self.tasks = dict(episode_tasks)
        self.by_task = defaultdict(list)
        intervals = []
        for episode, windows in self.windows.items():
            if not isinstance(windows, range) or windows.step != 1 or not windows:
                raise ValueError("episode windows must be nonempty contiguous ranges")
            intervals.append((windows.start, windows.stop, episode))
            self.by_task[self.tasks[episode]].append(episode)
        intervals.sort(key=lambda item: item[0])
        end = 0
        for start, stop, _ in intervals:
            if start != end:
                raise ValueError("episode window ranges must partition dataset indices")
            end = stop
        unpaired = [task for task, episodes in self.by_task.items() if len(episodes) < 2]
        if unpaired:
            raise ValueError(
                "each task needs at least two distinct episodes for DIWA pairs: "
                f"{unpaired[:8]}"
            )
        self.episodes = [item[2] for item in intervals]
        self.ends = [item[1] for item in intervals]
        self.size = end
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_batches = math.ceil(end / (num_replicas * (batch_size // 2)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        order_rng = np.random.default_rng(self.seed + self.epoch)
        anchors = order_rng.permutation(self.size)
        half = self.batch_size // 2
        total = len(self) * half * self.num_replicas
        anchors = np.resize(anchors, total)[self.rank::self.num_replicas]
        pair_rng = np.random.default_rng(
            [self.seed, self.epoch, self.rank, self.num_replicas]
        )
        for start in range(0, len(anchors), half):
            batch = []
            for anchor in anchors[start:start + half]:
                anchor = int(anchor)
                episode = self.episodes[bisect.bisect_right(self.ends, anchor)]
                partners = [
                    value for value in self.by_task[self.tasks[episode]]
                    if value != episode
                ]
                partner = partners[int(pair_rng.integers(len(partners)))]
                windows = self.windows[partner]
                partner_index = int(pair_rng.integers(windows.start, windows.stop))
                batch.extend((anchor, partner_index))
            yield batch


def padded_window_count(episode_length: int, supervised_length: int) -> int:
    """Count starts that retain every supervised state and pad only lookahead."""
    if episode_length < 1 or supervised_length < 1:
        raise ValueError("episode_length and supervised_length must be positive")
    return max(1, episode_length - supervised_length + 1)


def set_dataloader_epoch_metadata(
    dataloader: DataLoader,
    *,
    batch_size: int,
    world_size: int,
) -> DataLoader:
    """Record the batches and global samples the loader will actually yield.

    A map-style ``DataLoader`` length is determined by its sampler, batch
    size, and ``drop_last`` setting. Worker processes only fetch those batches
    and must not change the recorded epoch length. Training uses
    ``num_batches`` to flush the final gradient-accumulation group and to size
    the learning-rate schedule, so this metadata must equal ``len(loader)``.
    """
    if batch_size < 1 or world_size < 1:
        raise ValueError("batch_size and world_size must be positive")
    num_batches = len(dataloader)
    if num_batches < 1:
        raise ValueError(
            "the distributed dataloader yields no complete batches; reduce "
            "batch_size/world_size or provide more samples"
        )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_batches * batch_size * world_size
    return dataloader
