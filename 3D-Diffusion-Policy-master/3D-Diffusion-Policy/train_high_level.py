import matplotlib.pyplot as plt
import torch
from torch import optim
# from dp3_dexart_dataset import DP3DexArtDataset, get_dataloaders  # change this
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.dataset.dexart_dataset import DexArtDataset
import sys
# sys.path.append('tax3d-conditioned-mimicgen')
from diffusion_policy_3d.model.vision.articubot import PointNet2_super
import hydra
from torch.utils.data import DataLoader

def compute_weighted_displacement(scene_pcd, pred, goal_type):
    scene_pcd = scene_pcd.permute(0,2,1)
    batch_size, num_points, _ = scene_pcd.shape
    scene_pcd = scene_pcd[:, :, None, :3]

    weights = pred[:, :, -1]  # B, N
    # softmax the weights
    weights = torch.nn.functional.softmax(weights, dim=1)

    outputs = pred[:, :, :-1]  # B, N, 12
    # sum the displacement of the predicted gripper point cloud according to the weights
    if goal_type == "4points":
        pred_points = weights[:, :, None, None] * (
            scene_pcd + outputs.reshape(batch_size, num_points, 4, 3)
        )
    elif goal_type == "12points":
        pred_points = weights[:, :, None, None] * (
            scene_pcd + outputs.reshape(batch_size, num_points, 12, 3)
        )
    else:
        raise NotImplementedError
    
    pred_points = pred_points.sum(dim=1)
    return pred_points

def compute_high_level_loss(scene_pcd, pred, target, loss_type, goal_type):
    if loss_type == "weighted_displacement":
        pred_points = compute_weighted_displacement(scene_pcd, pred, goal_type)
        loss = torch.nn.functional.mse_loss(pred_points, target)
    else:
        raise NotImplementedError
    return loss

def prepare_model_input(obs_batch, goal_type, device):
    obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
    goal_pcd = obs_batch['goal_gripper_pcd'][:, -1, :, :3]

    # We only want current observation, not a full history
    pcd, imagined_pcd = obs_batch["point_cloud"][:, -1, :, :3], obs_batch["imagin_robot"][:, -1, :, :3]
    obs = torch.cat([pcd, imagined_pcd], axis=1).permute(0,2,1)

    if goal_type == "4points":
        chosen_four_point_idx = torch.tensor([16, 40, 64, 88]) # One on each finger
        target = goal_pcd[:,chosen_four_point_idx]
    elif goal_type == "12points":
        chosen_four_point_idx = torch.tensor([4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92]) # One on each link
        target = goal_pcd[:,chosen_four_point_idx]
    else:
        raise NotImplementedError

    return obs, target


@hydra.main(version_base="1.1", config_path="diffusion_policy_3d/config", config_name="high_level")
def main(cfg):

    num_epochs = cfg.num_epochs
    lr = cfg.lr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Oracle goals for training high-level
    # data_dir = cfg.data_dir
    # dataset = DP3DexArtDataset(data_dir, goal_mode="pointcloud_oracle")
    # train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size)

    dataset: BaseDataset
    dataset = hydra.utils.instantiate(cfg.dataset)
    train_loader = DataLoader(dataset, **cfg.dataloader)
    val_dataset = dataset.get_validation_dataset()
    val_loader = DataLoader(val_dataset, **cfg.val_dataloader)

    if cfg.model == "pn_plus_plus":
        if cfg.goal_type == "4points":
            model = PointNet2_super(num_classes=13, input_channel=3).to(device)
        elif cfg.goal_type == "12points":
            model = PointNet2_super(num_classes=37, input_channel=3).to(device)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    optimizer = optim.Adam(model.parameters(), lr=lr)

    with open("loss_log.txt", "w") as f:
        f.write("Epoch,TrainLoss,ValLoss\n")

    avg_train_losses = []
    avg_val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_val_loss = 0.0
        train_count = 0
        val_count = 0

        for batch in train_loader:
            obs, target = prepare_model_input(batch["obs"], cfg.goal_type, device)
            pred = model(obs)
            loss = compute_high_level_loss(obs, pred, target, cfg.loss_type, cfg.goal_type)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_count += 1

        avg_train_loss = total_train_loss / train_count
        avg_train_losses.append(avg_train_loss)

        # ===== Validation =====
        model.eval()
        total_val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                obs, target = prepare_model_input(batch["obs"], cfg.goal_type, device)
                pred = model(obs)
                loss = compute_high_level_loss(obs, pred, target, cfg.loss_type, cfg.goal_type)

                total_val_loss += loss.item()
                val_count += 1

        avg_val_loss = total_val_loss / val_count
        avg_val_losses.append(avg_val_loss)

        if epoch % 3 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            torch.save(model.state_dict(), f"dp3_epoch_{epoch+1}.pt")

        with open("loss_log.txt", "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f}\n")


    # ===== Final Test Evaluation =====
    model.eval()
    total_test_loss = 0.0
    test_count = 0

    with torch.no_grad():
        for batch in test_loader:
            obs, target = prepare_model_input(batch["obs"], cfg.goal_type, device)
            pred = model(obs)
            loss = compute_high_level_loss(obs, pred, target, cfg.loss_type, cfg.goal_type)

            total_test_loss += loss.item()
            test_count += 1

    avg_test_loss = total_test_loss / test_count
    print(f"Final Test Loss: {avg_test_loss:.4f}")


    if avg_train_losses:
        try:
            plt.figure()
            plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Training Loss')
            plt.title('Training Loss vs. Epochs')
            plt.grid(True)
            plt.savefig("training_loss_plot.png")
            print("Plot saved successfully.")
        except Exception as e:
            print("Failed to plot:", e)
    else:
        print("avg_train_losses is empty — skipping plot.")



if __name__ == "__main__":
    main()
