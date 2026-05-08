# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import time
import tqdm
import random
import yaml
import argparse

from collections import defaultdict
from contextlib import redirect_stdout

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import config as exp_cfg_mod

import rvt.utils.ddp_utils as ddp_utils
import rvt.mvt.config as mvt_cfg_mod

from rvt.mvt.mvt import MVT

from rvt.utils.get_dataset import get_dataset
from rvt.utils.rvt_utils import (
    TensorboardManager,
    short_name,
    get_num_feat,
    load_agent,
    RLBENCH_TASKS,
)
from rvt.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    DATA_FOLDER,
)

from rvt.models.dp3_agent import dp3_agent
import matplotlib.pyplot as plt

def train(agent, dataset, training_iterations, agent_type, epoch, rank=0):
    agent.train()
    log = defaultdict(float)          
    count = 0                         

    data_iter = iter(dataset)
    iter_command = range(training_iterations)

    for iteration in tqdm.tqdm(
        iter_command, disable=(rank != 0), position=0, leave=True
    ):
        raw_batch = next(data_iter)
        batch = {
            k: v.to(agent._device)
            for k, v in raw_batch.items()
            if isinstance(v, torch.Tensor)
        }
        batch["tasks"] = raw_batch["tasks"]
        if agent_type == "our":
            batch["lang_goal"] = raw_batch["lang_goal"]

        update_args = {
            "step": iteration,
            "replay_sample": batch,
            "backprop": True,
            "reset_log": (iteration == 0),
            "eval_log": False,
        }

        step_log = agent.update(**update_args)

        for k, v in step_log.items():
            log[k] += v
        count += 1

    log = {k: v / count for k, v in log.items()}

    if rank == 0:
        print(f"[Epoch {epoch}] Training losses:")
        for k, v in log.items():
            print(f"  {k}: {v:.6f}")

    return log


def save_agent(agent, path, epoch):
    model = agent._network
    optimizer = agent._optimizer
    lr_sched = agent._lr_sched

    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    '''
    torch.save({
        "epoch": epoch,
        "model_state": agent.policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "lr_sched_state": lr_sched.state_dict(),
        "normalizer_state": agent.policy.normalizer.state_dict(),
    }, path)
    '''
    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "lr_sched_state": lr_sched.state_dict(),
        },
        path,
    )

def get_tasks(exp_cfg):
    parsed_tasks = exp_cfg.tasks.split(",")
    if parsed_tasks[0] == "all":
        tasks = RLBENCH_TASKS
    else:
        tasks = parsed_tasks
    return tasks


def get_logdir(cmd_args, exp_cfg):
    log_dir = os.path.join(cmd_args.log_dir, exp_cfg.exp_id)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir):
    with open(f"{log_dir}/exp_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(exp_cfg.dump())

    with open(f"{log_dir}/mvt_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(mvt_cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)

def plot_losses(loss_history, save_path=None):
    plt.figure(figsize=(8, 5))
    for k, values in loss_history.items():
        plt.plot(values, label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DP3 Training Losses")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def experiment(rank, cmd_args, devices, port):
    """experiment.

    :param rank:
    :param cmd_args:
    :param devices: list or int. if list, we use ddp else not
    """
    device = devices[rank]
    device = f"cuda:{device}"
    ddp = len(devices) > 1
    ddp_utils.setup(rank, world_size=len(devices), port=port)

    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))

    if ddp:
        print(f"Running DDP on rank {rank}.")

    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id

    exp_cfg.peract.lr *= len(devices) * exp_cfg.bs
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.exp_id += f"_{short_name(cmd_args.exp_cfg_opts)}"
    if cmd_args.mvt_cfg_opts != "":
        exp_cfg.exp_id += f"_{short_name(cmd_args.mvt_cfg_opts)}"

    if rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
    exp_cfg.freeze()

    # Things to change
    BATCH_SIZE_TRAIN = exp_cfg.bs
    NUM_TRAIN = 100
    # to match peract, iterations per epoch
    TRAINING_ITERATIONS = int(exp_cfg.train_iter // (exp_cfg.bs * len(devices)))
    EPOCHS = exp_cfg.epochs
    TRAIN_REPLAY_STORAGE_DIR = "replay/replay_train"
    TEST_REPLAY_STORAGE_DIR = "replay/replay_val"
    log_dir = get_logdir(cmd_args, exp_cfg)
    tasks = get_tasks(exp_cfg)
    print("Training on {} tasks: {}".format(len(tasks), tasks))

    t_start = time.time()
    get_dataset_func = lambda: get_dataset(
        tasks,
        BATCH_SIZE_TRAIN,
        None,
        TRAIN_REPLAY_STORAGE_DIR,
        None,
        DATA_FOLDER,
        NUM_TRAIN,
        None,
        cmd_args.refresh_replay,
        device,
        num_workers=exp_cfg.num_workers,
        only_train=True,
        sample_distribution_mode=exp_cfg.sample_distribution_mode,
        agent=exp_cfg.agent, 
    )
    train_dataset, _ = get_dataset_func()

    import os

    print("\n" + "=" * 90, flush=True)
    print(f"[DBG][rank={rank} pid={os.getpid()}] After get_dataset_func()", flush=True)
    print(f"[DBG][rank={rank}] train_dataset type: {type(train_dataset)}", flush=True)
    print(f"[DBG][rank={rank}] TRAIN_REPLAY_STORAGE_DIR: {TRAIN_REPLAY_STORAGE_DIR}", flush=True)
    print(f"[DBG][rank={rank}] DATA_FOLDER: {DATA_FOLDER}", flush=True)
    print(f"[DBG][rank={rank}] refresh_replay: {cmd_args.refresh_replay}", flush=True)
    print(f"[DBG][rank={rank}] num_workers(exp_cfg): {exp_cfg.num_workers}", flush=True)

    # Try to locate the underlying replay buffer
    rb = getattr(train_dataset, "_replay_buffer", None)
    if rb is None and hasattr(train_dataset, "dataset"):
        rb = getattr(train_dataset.dataset, "_replay_buffer", None)

    print(f"[DBG][rank={rank}] has _replay_buffer: {rb is not None}", flush=True)

    if rb is not None:
        # Print key counters (different YARR versions name these differently)
        for name in ["_add_count", "add_count", "_num_transitions", "num_transitions", "_size", "size"]:
            if hasattr(rb, name):
                try:
                    v = getattr(rb, name)
                    v = v() if callable(v) else v
                    print(f"[DBG][rank={rank}] replay_buffer.{name} = {v}", flush=True)
                except Exception as e:
                    print(f"[DBG][rank={rank}] replay_buffer.{name} read failed: {e}", flush=True)

        # Print the exact parameters referenced by the error
        for name in ["_stack_size", "stack_size", "_update_horizon", "update_horizon"]:
            if hasattr(rb, name):
                try:
                    v = getattr(rb, name)
                    v = v() if callable(v) else v
                    print(f"[DBG][rank={rank}] replay_buffer.{name} = {v}", flush=True)
                except Exception as e:
                    print(f"[DBG][rank={rank}] replay_buffer.{name} read failed: {e}", flush=True)

    print("=" * 90 + "\n", flush=True)

    t_end = time.time()
    print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

    if exp_cfg.agent == "our":
        import rvt.models.rvt_agent as rvt_agent
        from rvt.models.rvt_agent import print_eval_log, print_loss_log

        mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
        if cmd_args.mvt_cfg_path != "":
            mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
        if cmd_args.mvt_cfg_opts != "":
            mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

        mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
        mvt_cfg.freeze()

        # for maintaining backward compatibility
        assert mvt_cfg.num_rot == exp_cfg.peract.num_rotation_classes, print(
            mvt_cfg.num_rot, exp_cfg.peract.num_rotation_classes
        )

        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        
        rvt = MVT(
            renderer_device=device,
            **mvt_cfg,
        ).to(device)
        
        if ddp:
            rvt = DDP(rvt, device_ids=[device])

        agent = rvt_agent.RVTAgent(
            network=rvt,
            image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
            add_lang=mvt_cfg.add_lang,
            stage_two=mvt_cfg.stage_two,
            rot_ver=mvt_cfg.rot_ver,
            scene_bounds=SCENE_BOUNDS,
            cameras=CAMERAS,
            log_dir=f"{log_dir}/test_run/",
            cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
            **exp_cfg.peract,
            **exp_cfg.rvt,
        )
        agent.build(training=True, device=device)
    elif exp_cfg.agent == "dp3":
        with open(exp_cfg.dp3_policy_config, "r") as f:
            dp3_policy_cfg = yaml.safe_load(f)

        agent = dp3_agent(
            policy_cfg=dp3_policy_cfg,
            cameras=CAMERAS,
            **exp_cfg.dp3,
        )

        # 3. Build optimizer + scheduler
        agent.build(training=True, device=device)
        
        stats_path = os.path.join(
            TRAIN_REPLAY_STORAGE_DIR,
            tasks[0],
            "dp3_norm_stats.pkl",
        )
        agent.load_normalizer_from_stats(
            stats_path=stats_path,
            mode="limits",     
            range_eps=1e-4,
            lang_std_floor=1e-3,
            verbose=True,
        )
    else:
        assert False, "Incorrect agent"

    start_epoch = 0
    end_epoch = EPOCHS
    '''
    if exp_cfg.resume != "":
        agent_path = exp_cfg.resume
        print(f"Recovering model and checkpoint from {exp_cfg.resume}")
        epoch = load_agent(agent_path, agent, only_epoch=False)
        start_epoch = epoch + 1
    '''
    if exp_cfg.resume != "":
        agent_path = exp_cfg.resume
        print(f"Recovering model and checkpoint from {exp_cfg.resume}")
        epoch = load_agent(agent_path, agent, only_epoch=False)

        # IMPORTANT: force all DP3 modules back to GPU after loading
        agent.policy.to(device)
        agent._network.to(device)

        start_epoch = epoch + 1
    dist.barrier()

    if rank == 0:
        ## logging unchanged values to reproduce the same setting
        temp1 = exp_cfg.peract.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        exp_cfg.peract.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()
        tb = TensorboardManager(log_dir)

    if rank == 0 and exp_cfg.agent == "our":
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)

    print("Start training ...", flush=True)

    loss_history = defaultdict(list)
    
    i = start_epoch
    while True:
        if i == end_epoch:
            break

        print(f"Rank [{rank}], Epoch [{i}]: Training on train dataset")
        out = train(agent, train_dataset, TRAINING_ITERATIONS, exp_cfg.agent, epoch=i, rank=rank)

        if rank == 0:
            print(f"\n[Epoch {i}] Training losses:")
            for k, v in out.items():
                print(f"  {k}: {v:.6f}")
                loss_history[k].append(v)
            tb.update("train", i, out)

        if rank == 0:
            # TODO: add logic to only save some models
            # Always save the latest checkpoint
            save_agent(agent, f"{log_dir}/model_last.pth", i)
        
            # Save every 10 epochs
            if (i + 1) % 10 == 0 and (i + 1) > 60:
                save_agent(agent, f"{log_dir}/model_{i + 1}.pth", i)
            
        i += 1

    if rank == 0:
        tb.close()
        print("[Finish]")
        plot_losses(loss_history, save_path=f"{log_dir}/training_losses.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())

    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--mvt_cfg_path", type=str, default="")
    parser.add_argument("--exp_cfg_path", type=str, default="")

    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")

    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--with-eval", action="store_true", default=False)

    cmd_args = parser.parse_args()
    del (
        cmd_args.entry
    )  # hack for multi processing -- removes an argument called entry which is not picklable

    devices = cmd_args.device.split(",")
    devices = [int(x) for x in devices]

    port = (random.randint(0, 3000) % 3000) + 27000
    mp.spawn(experiment, args=(cmd_args, devices, port), nprocs=len(devices), join=True)
