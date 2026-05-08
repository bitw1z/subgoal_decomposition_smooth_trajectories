from yacs.config import CfgNode as CN


def get_cfg_defaults():
    _C = CN()

    # ==================================================
    # Agent selection
    # ==================================================
    _C.agent = "our"   # options: ["our", "dp3"]

    # ==================================================
    # Shared experiment settings
    # ==================================================
    _C.exp_id = ""
    _C.resume = ""

    _C.tasks = ""
    _C.bs = 24
    _C.num_workers = 3
    _C.epochs = 100
    _C.train_iter = 160000
    _C.sample_distribution_mode = "task_uniform"

    # ==================================================
    # RVT / PerAct (only used if agent == "our")
    # ==================================================
    _C.peract = CN()
    _C.peract.lr = 1.25e-5
    _C.peract.warmup_steps = 2000
    _C.peract.optimizer_type = "lamb"
    _C.peract.num_rotation_classes = 72
    _C.peract.add_rgc_loss = True
    _C.peract.lr_cos_dec = True
    _C.peract.amp = True
    _C.peract.bnb = True
    _C.peract.lambda_weight_l2 = 1e-4

    _C.peract.transform_augmentation = True
    _C.peract.transform_augmentation_xyz = [0.1, 0.1, 0.1]
    _C.peract.transform_augmentation_rpy = [0.0, 0.0, 20.0]

    _C.rvt = CN()
    _C.rvt.gt_hm_sigma = 1.5
    _C.rvt.img_aug = 0.1
    _C.rvt.place_with_mean = True
    _C.rvt.move_pc_in_bound = True

    # ==================================================
    # DP3 (only used if agent == "dp3")
    # ==================================================
    _C.dp3 = CN()

    _C.dp3.lr = 1e-4
    _C.dp3.optimizer_type = "adamw"
    _C.dp3.weight_decay = 1e-4
    _C.dp3.amp = True
    _C.dp3.bnb = False

    _C.dp3.warmup_steps = 2000
    _C.dp3.cos_dec_max_step = 160000

    _C.dp3.horizon = 16
    _C.dp3.n_action_steps = 16
    _C.dp3.n_obs_steps = 1

    _C.dp3.noise_scheduler_cfg = CN()
    _C.dp3.noise_scheduler_cfg._target_ = (
        "diffusers.schedulers.scheduling_ddpm.DDPMScheduler"
    )
    _C.dp3.noise_scheduler_cfg.num_train_timesteps = 100
    _C.dp3.noise_scheduler_cfg.beta_schedule = "linear"

    # ==================================================
    # DP3 network config path
    # ==================================================
    _C.dp3_policy_config = ""

    # arguments present in peract official
    _C.peract_official = CN()
    _C.peract_official.cfg_path = "configs/peract_official_config.yaml"

    return _C
