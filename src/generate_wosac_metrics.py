import os
import time
import torch
import pickle
import multiprocessing
import tensorflow as tf

from models.metrics.wosac import WOSACMetrics
from data_modules.data_h5_womd import DataH5womd
from pl_modules.waymo_motion import WaymoMotion

#-------------------------------------------------------
CKPG_NAME = "2cti1q5z"
CKPG_VER = "v41"
# CKPG_NAME = "285yb3yb"
# CKPG_VER = "v59"

OUTPUT_DIR = f"outputs/wosac_metric_{CKPG_NAME}_{CKPG_VER}"
#-------------------------------------------------------

device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)

def pickle_scene_metric(scene_idx, config, scenario_bytes, scenario_rollout):
    print(f"pickle scene {scene_idx}") 
    scenario_metrics = WOSACMetrics._compute_scenario_metrics(
        config, 
        scenario_bytes,
        scenario_rollout
    )
    with open(f"{OUTPUT_DIR}/scene_{scene_idx}.pkl", 'wb') as f:  
        pickle.dump(scenario_metrics, f)

if __name__ == '__main__':
    # 加载数据集
    single_data = DataH5womd(
        data_dir = "data/h5_womd_sim_agent",
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
    ckpt_path = f"checkpoints/{CKPG_NAME}_{CKPG_VER}.ckpt"
    checkpoint_version = f"yangyh408/traffic_bots/{CKPG_NAME}:{CKPG_VER}"
    model = WaymoMotion.load_from_checkpoint(ckpt_path, wb_artifact=checkpoint_version)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    config = model.wosac_metrics.load_metrics_config()

    for batch_idx, batch in enumerate(val_dataloader):
        tic = time.time()    

        skip_sign = True
        batch["scenario_bytes"] = []
        for scene_idx in batch['episode_idx']:
            output_path = f"{OUTPUT_DIR}/scene_{scene_idx}.pkl"
            with open(rf"data/h5_wosac/val_scenarios/{scene_idx}.pickle", "rb") as handle:
                batch["scenario_bytes"].append(pickle.load(handle).hex())
            if not os.path.exists(output_path):
                skip_sign = False
        
        if not skip_sign:
            # 使用 LightningModule 的 validation_step 进行单 batch 验证
            try:
                model.eval()
                with torch.no_grad():
                    rollout = model.manual_joint_future_pred(batch=batch, batch_idx=batch_idx)

                wosac_data = model.wosac_post_processing(batch, rollout)
                scenario_rollouts = model.wosac_post_processing.get_scenario_rollouts(wosac_data)
                # model.wosac_metrics.update(scenario_rollouts, batch["scenario_bytes"])
                
                # 准备要传递给每个进程的数据  
                tasks = [(scene_idx, config, batch["scenario_bytes"][i], scenario_rollouts[i])  
                        for i, scene_idx in enumerate(batch['episode_idx'])] 
                
                with multiprocessing.Pool(processes=2) as pool:
                    pool_scenario_metrics = pool.starmap(
                        pickle_scene_metric,
                        tasks,
                    )
            except Exception as e:
                print(f"Generate Failed in batch {batch_idx} with error {repr(e)}!")
                
            print(f"processing epoch {batch_idx} cost {time.time()-tic}s")

