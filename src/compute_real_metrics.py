import os 

from data_modules.data_h5_womd import DataH5womd
import pl_modules.waymo_motion as waymo_motion_module
from utils.transform_utils import torch_pos2global, torch_rad2rot, torch_rad2global
from models.metrics.custom_metrics import RealMetrics
from utils.real_features import compute_real_metric_features

import torch
import pickle
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import trajectory_utils

CKPG_NAME = "2cti1q5z"
CKPG_VER = "v41"
# CKPG_NAME = "285yb3yb"
# CKPG_VER = "v59"
HOME_DIR = "/home/yangyh408/codes/DataDriven_BehaviourModels/deep_learning_model/v1.0"
PKL_DIR = f"{HOME_DIR}/outputs/real_metrics_{CKPG_NAME}_{CKPG_VER}"

def save(rollout, filename):  
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as file:  
        pickle.dump(rollout, file)  

if __name__ == '__main__':
    # 加载数据集
    single_data = DataH5womd(
        data_dir = f"{HOME_DIR}/data/h5_womd_sim_agent",
        filename_train = "training",
        filename_val = "validation",
        filename_test = "testing",
        n_agent = 64,
        batch_size = 4,
        num_workers = 4,
    )
    single_data.setup(stage="validate")
    val_dataloader = single_data.val_dataloader()

    # 加载训练后的模型
    ckpt_path = f"{HOME_DIR}/checkpoints/{CKPG_NAME}_{CKPG_VER}.ckpt"
    
    checkpoint_version = f"yangyh408/traffic_bots/{CKPG_NAME}:{CKPG_VER}"
    model = waymo_motion_module.WaymoMotion.load_from_checkpoint(
        ckpt_path, wb_artifact=checkpoint_version
    )

    for batch_idx, batch in enumerate(val_dataloader):
        if not os.path.exists(os.path.join(PKL_DIR, f"scene_{batch['episode_idx'][-1]}.pkl")):
            print(f"Is dealing batch {batch_idx} ...")

            batch["scenario_bytes"] = []
            for scene_idx in batch['episode_idx']:
                with open(rf"{HOME_DIR}/data/h5_wosac/val_scenarios/{scene_idx}.pickle", "rb") as handle:
                    batch["scenario_bytes"].append(pickle.load(handle).hex())
            
            model.eval()
            with torch.no_grad():
                rollout = model.manual_joint_future_pred(batch=batch, batch_idx=batch_idx)
            
            # 保存每个batch计算的rollout数据
            # save(rollout, rollout_path)

            # 转换为全局坐标系
            trajs = rollout.preds.permute(0, 2, 1, 3, 4)

            scenario_center = batch["scenario_center"].unsqueeze(1)  # [n_sc, 1, 2]
            scenario_rot = torch_rad2rot(batch["scenario_yaw"])  # [n_sc, 2, 2]

            pos_sim, yaw_sim = trajs[..., :2], trajs[..., 2:3]  # [n_sc, n_joint_future, n_ag, n_step_future, 2/1]
            pos_sim = torch_pos2global(pos_sim.flatten(1, 3), scenario_center, scenario_rot).view(pos_sim.shape)
            yaw_sim = torch_rad2global(yaw_sim.flatten(1, 4), batch["scenario_yaw"]).view(yaw_sim.shape)
            agent_size = batch['agent/size'][..., :2]

            for si in range(trajs.shape[0]):
                try:
                    real_metrics = RealMetrics()

                    # 筛选车辆
                    mask_evaluated_agent = tf.logical_and(
                        tf.reduce_sum(tf.cast(rollout.valid[si, :, 0, :], tf.int32), axis=-1) > 10,
                        
                        # 根据总坐标偏移量进行掩码：处理从中间帧加入的车可能有问题
                        # tf.reduce_sum(
                        #     tf.where(
                        #         rollout.valid[si, :, 0, 1:], 
                        #         tf.reduce_sum((rollout.preds[si, :, 0, 1:, :2] - rollout.preds[si, :, 0, :-1, :2]) ** 2, axis=-1), 
                        #         0
                        #     ),
                        #     axis=-1
                        # ) > 1,

                        # 根据总速度进行掩码
                        tf.reduce_sum(tf.abs(rollout.preds[si, :, 0, :, -1]), axis=-1) > 90
                    )
                    agent_filter = tf.range(rollout.valid[si, :, 0, :].shape[0])[mask_evaluated_agent].numpy()
                    object_id = batch['agent/object_id'][si][agent_filter]

                    # 计算回放轨迹的真实指标特征
                    origin_scenario = scenario_pb2.Scenario.FromString(bytes.fromhex(batch['scenario_bytes'][si]))
                    logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(origin_scenario)
                    logged_trajectories = logged_trajectories.gather_objects_by_id(object_id)
                    logged_trajectories = logged_trajectories.slice_time(start_index=1)

                    real_metrics.add_log_features(compute_real_metric_features(
                        logged_trajectories.x, 
                        logged_trajectories.y, 
                        logged_trajectories.length, 
                        logged_trajectories.width, 
                        logged_trajectories.heading, 
                        logged_trajectories.valid
                    ))

                    # 计算每一个仿真rollout的真实指标特征
                    length = tf.repeat(agent_size[si, agent_filter, 0, tf.newaxis], trajs.shape[-2], axis=-1)
                    width = tf.repeat(agent_size[si, agent_filter, 1, tf.newaxis], trajs.shape[-2], axis=-1)
                    for ri in range(trajs.shape[1]):
                        center_x = pos_sim[si, ri, agent_filter, :, 0]
                        center_y = pos_sim[si, ri, agent_filter, :, 1]
                        heading = yaw_sim[si, ri, agent_filter, :, 0]
                        valid = rollout.valid[si, agent_filter, ri, :]
                        
                        real_metrics.add_sim_features(
                            compute_real_metric_features(center_x, center_y, length, width, heading, valid))
                    
                    real_metrics.compute_js_divergence(method='histogram', plot=False)
                    # 保存单场景真实性指标
                    real_metrics.save(os.path.join(PKL_DIR, f"scene_{batch['episode_idx'][si]}.pkl"))
                except Exception as e:
                    print(f"Error with loading episode {batch['episode_idx'][si]}!")