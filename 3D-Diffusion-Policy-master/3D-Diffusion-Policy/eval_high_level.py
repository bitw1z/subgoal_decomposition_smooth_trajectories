import matplotlib.pyplot as plt
import torch
from torch import optim
import pickle
import sys
from tqdm import tqdm
from diffusion_policy_3d.model.vision.articubot import PointNet2_super
from train_high_level import compute_weighted_displacement
import hydra
import zarr

def extract_model_input(obs_batch, idx, device):
    pcd = torch.from_numpy(obs_batch["point_cloud"][idx, :, :3])
    imagined_pcd = torch.from_numpy(obs_batch["imagin_robot"][idx, :, :3])
    obs = torch.cat([pcd, imagined_pcd], axis=0)[None].permute(0,2,1).to(device).float()
    return obs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_file = "outputs/2025-08-07/12-32-23_12pts/dp3_epoch_10.pt"  #outputs/2025-08-06/18-02-45_4pts/dp3_epoch_11.pt
model_type = "pn_plus_plus"
goal_type = "12points"

# eval_file = "/data/xinyu/demo_dexart_Jun18/laptop/demo_3707.pkl"
# eval_data = pickle.load(open(eval_file, "rb"))
# output_file = "demo_3707_with_pred.pkl"
# output_data = []

eval_file = "/data/xinyu/dexart_laptop_expert_eval_oraclegoal_1000_seen.zarr"
demo_data = zarr.open(eval_file, mode='r')

demo_idx = 30
output_file = "outputs/2025-08-07/12-32-23_12pts/demo_30.pkl"
output_data = []

if model_type == "pn_plus_plus":
    if goal_type == "4points":
        model = PointNet2_super(num_classes=13, input_channel=3).to(device)
    elif goal_type == "12points":
        model = PointNet2_super(num_classes=37, input_channel=3).to(device)
    else:
        raise NotImplementedError
    model.load_state_dict(torch.load(ckpt_file))
else:
    raise NotImplementedError


episode_ends = demo_data["meta"]["episode_ends"]
start_idx = 0 if demo_idx == 0 else episode_ends[demo_idx - 1]
last_idx = episode_ends[demo_idx] - 1
demo_len = episode_ends[demo_idx] - episode_ends[demo_idx - 1]

for step in range(demo_len):
    idx = start_idx + step
    obs = extract_model_input(demo_data["data"], idx, device)
    with torch.no_grad():
        pred = model(obs)
    pred_points = compute_weighted_displacement(obs, pred, goal_type)
    # print(pred_points.shape)
    # print(pred_points)

    output = {
        "point_cloud": demo_data["data"]["point_cloud"][idx, :, :3],
        "imagin_robot": demo_data["data"]["imagin_robot"][idx, :, :3],
        "goal_gripper_pcd": pred_points[0].detach().cpu().numpy()
    }

    output_data.append(output)


with open(output_file, 'wb') as f:
    pickle.dump(output_data, f)
