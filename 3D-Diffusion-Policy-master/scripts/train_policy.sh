# Examples:
# bash scripts/train_policy.sh dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 dexart_laptop 0322 0 0
# bash scripts/train_policy.sh simple_dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 metaworld_basketball 0602 0 0


# export MUJOCO_PY_MUJOCO_PATH=/project_data/held/xinyul5/.mujoco/mujoco210
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project_data/held/xinyul5/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
# export MUJOCO_GL=egl


DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
# eef_points=${5}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=0

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1 
# export CUDA_VISIBLE_DEVICES=${gpu_id}
# python train.py --config-name=${config_name}.yaml \
#                             task=${task_name} \
#                             hydra.run.dir=${run_dir} \
#                             training.debug=$DEBUG \
#                             training.seed=${seed} \
#                             training.device="cuda:0" \
#                             exp_name=${exp_name} \
#                             logging.mode=${wandb_mode} \
#                             checkpoint.save_ckpt=${save_ckpt}


python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cpu" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            # policy.eef_points=${eef_points}
                                