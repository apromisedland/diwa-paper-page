import os
import re
import random
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import h5py
import argparse
from pathlib import Path
from PIL import Image
import json
# from sentence_transformers import SentenceTransformer

try:
    from utils.diwa_schema import validate_candidate_rule_ids
except ModuleNotFoundError:  # Direct ``python utils/...`` execution.
    from diwa_schema import validate_candidate_rule_ids


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def extract_task_information(file_name, path):
    """
    Extracts task information from the given file name.
    """
    global dataset_name
    # Regular expression pattern to extract the task name
    if dataset_name == 'libero_10':
        pattern = r'{}/((.+)_SCENE[0-9]+_(.+))_demo\.hdf5'.format(path)
    else:
        pattern = r'{}/(.*)_demo\.hdf5'.format(re.escape(path))

    # Extracting the task name
    match = re.search(pattern, file_name)
    if dataset_name == 'libero_10':
        print(match.group(3).lower().replace("_", " "))
        return match.group(1).lower() if match else None, match.group(3).lower().replace("_", " ")
    else:
        task_full = match.group(1).lower()
        task_simple = task_full.replace("_", " ")
        print(task_simple)
        return task_full, task_simple

class DatasetConverter:
    def __init__(
        self,
        src_dir: str,
        tgt_dir: str,
        rank: int,
        num_worker: int,
        start_episode_idx,
        end_episode_idx,
    ):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.rank = rank
        self.num_worker = num_worker
        self.start_episode_idx = start_episode_idx
        self.end_episode_idx = end_episode_idx
        # self.lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def process_episode(self, episode_dir, language_instructions, demo_data, episode_index, episode_index_in_task):
        i = episode_index_in_task
        demo = demo_data['demo_{}'.format(i)]
        # get episode dir
        episode_dir.mkdir(exist_ok=True)

        ### Get agent's view camera
        obs = np.array(demo['obs']['agentview_rgb'])
        # obs = obs.transpose(0,3,1,2)
        
        ### Get wrist's view camera
        obs_wrist = np.array(demo['obs']['eye_in_hand_rgb'])
        # obs_wrist = obs_wrist.transpose(0,3,1,2)

        ### Get actions
        action = np.array(demo['actions'])  # -1 open, 1 close
        
        joint_state = np.array(demo['obs']['joint_states'])
        ee_state = np.array(demo['obs']['ee_states'])

        gripper_state = np.zeros_like(action[:, -1])
        gripper_state[1:] = action[:-1, -1]
        gripper_state[0] = action[0, -1]

        gripper_position = np.array(demo['obs']['gripper_states'])
        measured_supervision = None
        supervision_keys = (
            "rewards",
            "dones",
            "progress",
            "candidate_actions",
            "candidate_q_values",
            "candidate_rule_ids",
        )
        present_supervision = [key in demo for key in supervision_keys]
        # Native LIBERO rewards/dones are ordinary trajectory fields. They
        # do not imply that a complete DIWA candidate annotation is embedded;
        # measured DIWA labels may instead be supplied in external sidecars.
        has_diwa_annotation = any(key in demo for key in supervision_keys[2:])
        if has_diwa_annotation and not all(present_supervision):
            missing = [
                key
                for key, present in zip(
                    supervision_keys, present_supervision
                )
                if not present
            ]
            raise ValueError(
                f"partial DIWA supervision in demo_{i}; missing {missing}"
            )
        if all(present_supervision):
            measured_supervision = {
                "reward": np.asarray(demo["rewards"]),
                "done": np.asarray(demo["dones"]),
                "progress": np.asarray(demo["progress"]),
                "candidate_actions": np.asarray(
                    demo["candidate_actions"], dtype=np.float32
                ),
                "candidate_q_values": np.asarray(
                    demo["candidate_q_values"], dtype=np.float32
                ),
                "candidate_rule_ids": np.asarray(
                    demo["candidate_rule_ids"]
                ),
            }

        # task emb
        # task_emb = self.lang_model.encode(language_instructions)

        # get episode length
        num_steps = obs.shape[0]
        native_outcomes = {}
        for source, destination in (("rewards", "reward"), ("dones", "done")):
            if source in demo:
                values = np.asarray(demo[source]).reshape(-1)
                if values.shape != (num_steps,) or not np.isfinite(values).all():
                    raise ValueError(f"invalid native {source} in demo_{i}")
                if source == "dones" and not np.isin(values, (0, 1)).all():
                    raise ValueError(f"native dones must be binary in demo_{i}")
                native_outcomes[destination] = values
        if measured_supervision is not None:
            lengths = {
                measured_supervision[key].shape[0]
                for key in (
                    "reward",
                    "done",
                    "progress",
                    "candidate_actions",
                    "candidate_q_values",
                )
            }
            if lengths != {num_steps}:
                raise ValueError(
                    f"DIWA supervision length does not match demo_{i}"
                )
            candidate_actions = measured_supervision[
                "candidate_actions"
            ]
            candidate_q_values = measured_supervision[
                "candidate_q_values"
            ]
            if (
                candidate_actions.ndim != 4
                or candidate_actions.shape[-1] != 7
                or candidate_q_values.ndim != 2
                or candidate_actions.shape[:2]
                != candidate_q_values.shape
            ):
                raise ValueError(
                    f"invalid candidate action/Q shapes in demo_{i}"
                )
            validate_candidate_rule_ids(
                measured_supervision["candidate_rule_ids"],
                candidate_count=candidate_actions.shape[1],
            )
            if not all(
                np.isfinite(measured_supervision[key]).all()
                for key in (
                    "reward",
                    "done",
                    "progress",
                    "candidate_actions",
                    "candidate_q_values",
                )
            ):
                raise ValueError(
                    f"non-finite DIWA supervision in demo_{i}"
                )
            progress = measured_supervision["progress"]
            if ((progress < 0) | (progress > 1)).any():
                raise ValueError(
                    f"DIWA progress outside [0, 1] in demo_{i}"
                )
        
        episode_dir = episode_dir/str(episode_index).zfill(6)
        episode_dir.mkdir(exist_ok=True)

        # save episode length and language instruction
        with h5py.File(f'{episode_dir}/meta_info.h5', 'w') as h5_file:
            h5_file.create_dataset(name='length', data=num_steps)
        
        steps_dir = episode_dir/'steps'
        steps_dir.mkdir(exist_ok=True)
        for step_index in range(num_steps):
            step_dir = episode_dir/'steps'/str(step_index).zfill(4)
            step_dir.mkdir(exist_ok=True)
            
            with h5py.File(f'{step_dir}/other.h5', 'w') as h5_file:
                # language instruction
                h5_file.create_dataset('language_instruction', data=np.array(language_instructions, dtype=h5py.string_dtype(encoding='utf-8')))
                # task emb
                # h5_file.create_dataset(name='task_emb', data=task_emb)

                # episode length
                h5_file.create_dataset(name='episode_length', data=num_steps)

                # action
                h5_file.create_dataset(name='action', data=action[step_index])
                if measured_supervision is not None:
                    h5_file.create_dataset(
                        name="reward",
                        data=measured_supervision["reward"][step_index],
                    )
                    h5_file.create_dataset(
                        name="done",
                        data=measured_supervision["done"][step_index],
                    )
                    h5_file.create_dataset(
                        name="progress",
                        data=measured_supervision["progress"][step_index],
                    )
                    h5_file.create_dataset(
                        name="candidate_actions",
                        data=measured_supervision["candidate_actions"][
                            step_index
                        ],
                    )
                    h5_file.create_dataset(
                        name="candidate_q_values",
                        data=measured_supervision["candidate_q_values"][
                            step_index
                        ],
                    )
                    h5_file.create_dataset(
                        name="candidate_rule_ids",
                        data=np.asarray(
                            measured_supervision["candidate_rule_ids"],
                            dtype=h5py.string_dtype(encoding="utf-8"),
                        ),
                    )
                else:
                    for name, values in native_outcomes.items():
                        h5_file.create_dataset(name, data=values[step_index])

                # observation (timestep, proprio, image_XXX)
                observation_group = h5_file.create_group(name='observation')

                ## image
                # ### image_primary
                Image.fromarray(obs[step_index]).save(f'{step_dir}/image_primary.jpg')
                ### image_wrist
                Image.fromarray(obs_wrist[step_index]).save(f'{step_dir}/image_wrist.jpg')

                ## proprio
                observation_group.create_dataset(name='proprio', data=joint_state[step_index])

                ## tcp_pose
                observation_group.create_dataset(name='tcp_pose', data=ee_state[step_index])

                ## gripper state (-1 or 1)
                observation_group.create_dataset(name='gripper_state', data=gripper_state[step_index])

                ## gripper position (n, 2)
                observation_group.create_dataset(name='gripper_position', data=gripper_position[step_index])

    def convert_origin_dataset_to_target(self, dataset_by_task):
        # /dataset_0
        # |_meta_info.h5
        # |_/episodes
        # | |_/0
        # | | |_/steps
        # | |   |_/0
        # |     | |_other.h5
        # |     | |_XXX.jpg
        # |     |...
        # | |_/1
        # | |_...
        # /dataset_1
        # |
        episodes_dir = self.tgt_dir/'episodes'
        episodes_dir.mkdir(exist_ok=True)

        num_episodes = 0
        for dataset in dataset_by_task:
            num_episodes += dataset['num_episode']

        if self.rank == 0:
            with h5py.File(f'{str(self.tgt_dir)}/meta_info.h5', 'w') as h5_file:
                h5_file.create_dataset(name='num_episodes', data=num_episodes)

        processed_task_num_episode = 0
        task_index = 0

        for episode_index in range(num_episodes):
            episode_index_in_task = episode_index - processed_task_num_episode

            if episode_index < self.start_episode_idx:
                continue
            if self.end_episode_idx is not None:
                if episode_index >= self.end_episode_idx:
                    break
            if episode_index % self.num_worker != self.rank:
                if episode_index_in_task+1 == dataset_by_task[task_index]['num_episode']:
                    processed_task_num_episode += dataset_by_task[task_index]['num_episode']
                    task_index += 1
                continue
            print(self.rank, episode_index, '/' , num_episodes)
            self.process_episode(episode_dir=episodes_dir, language_instructions=dataset_by_task[task_index]['language'], demo_data=dataset_by_task[task_index]['data'], episode_index=episode_index, episode_index_in_task=episode_index_in_task)

            if episode_index_in_task+1 == dataset_by_task[task_index]['num_episode']:
                processed_task_num_episode += dataset_by_task[task_index]['num_episode']
                task_index += 1

    def run(self):
        print(f'target dir: {self.tgt_dir}')

        dataset_by_task = []
        source_files = sorted(Path(self.src_dir).glob("*_demo.hdf5"))
        if not source_files:
            raise FileNotFoundError(
                f"no *_demo.hdf5 files found in {self.src_dir}"
            )
        for path in source_files:
            path_name = str(path)
            task_name, task_language = extract_task_information(path_name, self.src_dir)
            demo_data = h5py.File(path_name, 'r')['data']
            num_episode = len(demo_data)
            dataset = {
                'language': task_language,
                'num_episode': num_episode,
                'data': demo_data
            }
            dataset_by_task.append(dataset)

        self.convert_origin_dataset_to_target(dataset_by_task)

        print(f'data saved at {self.tgt_dir}')

        # get data_info.json
        data_info = []
        episode_idx = 0
        total_step = 0
        for path in source_files:
            path_name = str(path)
            demo_data = h5py.File(path_name, 'r')['data']
            num_episode = len(demo_data)
            for i in range(num_episode):
                num_steps = np.array(demo_data['demo_{}'.format(i)]['obs']['agentview_rgb']).shape[0]
                data_info.append([str(episode_idx).zfill(6), num_steps])
                episode_idx += 1
                total_step += num_steps
        # print(total_step)
        with open(f'./data_info/{dataset_name}_converted.json', 'w') as f:
            json.dump(data_info, f)

def convert_worker(
    rank,
    port,
    num_worker,
    dataset_name_arg,
    src_dir,
    tgt_dir,
    start_episode_idx=0,
    end_episode_idx=None,
):
    if num_worker > 1:
        setup(rank, world_size=num_worker, port=port)

    global dataset_name
    dataset_name = dataset_name_arg
    tgt_dir = Path(tgt_dir)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    dataset_converter = DatasetConverter(
        src_dir=src_dir,
        tgt_dir=tgt_dir,
        rank=rank,
        num_worker=num_worker,
        start_episode_idx=start_episode_idx,  # the dataset[start_episode_idx] will be processed
        end_episode_idx=end_episode_idx,  # None means the last episode. if not none, the dataset[end_episode_idx - 1] will be processed and the dataset[end_episode_idx] will not be processed
    )
    dataset_converter.run()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert official LIBERO HDF5 demonstrations to DreamVLA steps."
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        choices=(
            "libero_10",
            "libero_90",
            "libero_spatial",
            "libero_object",
            "libero_goal",
        ),
    )
    parser.add_argument(
        "--src-dir",
        required=True,
        help="directory containing official LIBERO *_demo.hdf5 files",
    )
    parser.add_argument(
        "--tgt-dir",
        required=True,
        help="output converted dataset directory",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--start-episode-idx", type=int, default=0)
    parser.add_argument("--end-episode-idx", type=int)
    parser.add_argument(
        "--master-port",
        type=int,
        default=(random.randint(0, 3000) % 3000) + 27000,
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be positive")
    if args.start_episode_idx < 0:
        parser.error("--start-episode-idx must be non-negative")
    if (
        args.end_episode_idx is not None
        and args.end_episode_idx <= args.start_episode_idx
    ):
        parser.error("--end-episode-idx must exceed --start-episode-idx")
    return args


if __name__ == "__main__":
    args = parse_args()
    worker_args = (
        args.master_port,
        args.num_workers,
        args.dataset_name,
        args.src_dir,
        args.tgt_dir,
        args.start_episode_idx,
        args.end_episode_idx,
    )
    if args.num_workers == 1:
        convert_worker(0, *worker_args)
    else:
        mp.spawn(
            convert_worker,
            args=worker_args,
            nprocs=args.num_workers,
            join=True,
        )
