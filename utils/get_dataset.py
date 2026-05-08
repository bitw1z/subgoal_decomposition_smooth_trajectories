# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import sys
import shutil
import torch
import clip

from rvt.libs.peract.helpers.utils import extract_obs
from rvt.utils.rvt_utils import ForkedPdb
import torch.distributed as dist

from rvt.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    EPISODE_FOLDER,
    VARIATION_DESCRIPTIONS_PKL,
    DEMO_AUGMENTATION_EVERY_N,
    ROTATION_RESOLUTION,
    VOXEL_SIZES,
)
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer


def get_dataset(
    tasks,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_TEST,
    TRAIN_REPLAY_STORAGE_DIR,
    TEST_REPLAY_STORAGE_DIR,
    DATA_FOLDER,
    NUM_TRAIN,
    NUM_VAL,
    refresh_replay,
    device,
    num_workers,
    only_train,
    sample_distribution_mode="transition_uniform",
    agent: str = "our"
):
    if agent == "dp3":
        from rvt.utils.dataset import create_replay, fill_replay 

        train_replay_buffer = create_replay(
            batch_size=BATCH_SIZE_TRAIN,
            timesteps=1,
            disk_saving=True,
            cameras=CAMERAS,
            replay_size=3e5,
        )
        
        if not only_train:
            test_replay_buffer = create_replay(
                batch_size=BATCH_SIZE_TEST,
                timesteps=1,
                disk_saving=True,
                cameras=CAMERAS,
                replay_size=3e5,
            )
    else:
        # RVT dataset builders (original keyframe dataset)
        from rvt.utils.dataset import create_replay, fill_replay

        train_replay_buffer = create_replay(
            batch_size=BATCH_SIZE_TRAIN,
            timesteps=1,
            disk_saving=True,
            cameras=CAMERAS,
            voxel_sizes=VOXEL_SIZES,
        )
        if not only_train:
            test_replay_buffer = create_replay(
                batch_size=BATCH_SIZE_TEST,
                timesteps=1,
                disk_saving=True,
                cameras=CAMERAS,
                voxel_sizes=VOXEL_SIZES,
            )

    # load pre-trained language model
    try:
        clip_model, _ = clip.load("RN50", device="cpu")  # CLIP-ResNet50
        clip_model = clip_model.to(device)
        clip_model.eval()
    except RuntimeError:
        print("WARNING: Setting Clip to None. Will not work if replay not on disk.")
        clip_model = None

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    for task in tasks:  # for each task
        # print("---- Preparing the data for {} task ----".format(task), flush=True)
        EPISODES_FOLDER_TRAIN = f"train/{task}/all_variations/episodes"
        EPISODES_FOLDER_VAL = f"val/{task}/all_variations/episodes"
        data_path_train = os.path.join(DATA_FOLDER, EPISODES_FOLDER_TRAIN)
        data_path_val = os.path.join(DATA_FOLDER, EPISODES_FOLDER_VAL)
        train_replay_storage_folder = f"{TRAIN_REPLAY_STORAGE_DIR}/{task}"
        test_replay_storage_folder = f"{TEST_REPLAY_STORAGE_DIR}/{task}"
        stats_save_path = os.path.join(train_replay_storage_folder, "dp3_norm_stats.pkl")

        # if refresh_replay, then remove the existing replay data folder
        if refresh_replay:
            print("[Info] Remove exisitng replay dataset as requested.", flush=True)
            if os.path.exists(train_replay_storage_folder) and os.path.isdir(
                train_replay_storage_folder
            ):
                shutil.rmtree(train_replay_storage_folder)
                print(f"remove {train_replay_storage_folder}")
            if os.path.exists(test_replay_storage_folder) and os.path.isdir(
                test_replay_storage_folder
            ):
                shutil.rmtree(test_replay_storage_folder)
                print(f"remove {test_replay_storage_folder}")


        # print("----- Train Buffer -----")
        if agent == "dp3":
            if rank == 0: 
                # -------- DP3 PATH --------
                fill_replay(
                    replay=train_replay_buffer,
                    task=task,
                    task_replay_storage_folder=train_replay_storage_folder,
                    start_idx=0,
                    num_demos=NUM_TRAIN,
                    cameras=CAMERAS,
                    data_path=data_path_train,
                    episode_folder=EPISODE_FOLDER,
                    variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
                    clip_model=clip_model,
                    device=device,
                    action_horizon=16, 
                    agent=agent,
                    collect_stats=True,
                    stats_save_path=stats_save_path,
                )

            # wait until rank 0 finishes writing
            if world > 1:
                dist.barrier()

            # all ranks load from disk (safe now)
            if rank != 0:
                train_replay_buffer.recover_from_disk(task, train_replay_storage_folder)
        else:
            # -------- RVT PATH --------
            fill_replay(
                replay=train_replay_buffer,
                task=task,
                task_replay_storage_folder=train_replay_storage_folder,
                start_idx=0,
                num_demos=NUM_TRAIN,
                demo_augmentation=True,
                demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
                cameras=CAMERAS,
                rlbench_scene_bounds=SCENE_BOUNDS,
                voxel_sizes=VOXEL_SIZES,
                rotation_resolution=ROTATION_RESOLUTION,
                crop_augmentation=False,
                data_path=data_path_train,
                episode_folder=EPISODE_FOLDER,
                variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
                clip_model=clip_model,
                device=device,
                agent=agent,
            )

        if not only_train:
            if agent == "dp3":
                if rank == 0: 
                    fill_replay(
                        replay=test_replay_buffer,
                        task=task,
                        task_replay_storage_folder=test_replay_storage_folder,
                        start_idx=0,
                        num_demos=NUM_VAL,
                        cameras=CAMERAS,
                        data_path=data_path_val,
                        episode_folder=EPISODE_FOLDER,
                        variation_descriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
                        clip_model=clip_model,
                        device=device,
                        action_horizon=16,
                        agent=agent,
                        collect_stats=False,
                        stats_save_path=None,
                    )

                # wait until rank 0 finishes writing
                if world > 1:
                    dist.barrier()

                # all ranks load from disk (safe now)
                if rank != 0:
                    train_replay_buffer.recover_from_disk(task, train_replay_storage_folder)
            else:
                fill_replay(
                    replay=test_replay_buffer,
                    task=task,
                    task_replay_storage_folder=test_replay_storage_folder,
                    start_idx=0,
                    num_demos=NUM_VAL,
                    demo_augmentation=True,
                    demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
                    cameras=CAMERAS,
                    rlbench_scene_bounds=SCENE_BOUNDS,
                    voxel_sizes=VOXEL_SIZES,
                    rotation_resolution=ROTATION_RESOLUTION,
                    crop_augmentation=False,
                    data_path=data_path_val,
                    episode_folder=EPISODE_FOLDER,
                    variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
                    clip_model=clip_model,
                    device=device,
                    agent=agent,
                )

    # delete the CLIP model since we have already extracted language features
    del clip_model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # wrap buffer with PyTorch dataset and make iterator
    train_wrapped_replay = PyTorchReplayBuffer(
        train_replay_buffer,
        sample_mode="random",
        num_workers=num_workers,
        sample_distribution_mode=sample_distribution_mode,
    )
    train_dataset = train_wrapped_replay.dataset()

    if only_train:
        test_dataset = None
    else:
        test_wrapped_replay = PyTorchReplayBuffer(
            test_replay_buffer,
            sample_mode="enumerate",
            num_workers=num_workers,
        )
        test_dataset = test_wrapped_replay.dataset()
    return train_dataset, test_dataset
