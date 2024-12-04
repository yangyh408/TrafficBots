import torch
from data_modules.data_h5_womd import DataH5womd
import hydra

#-------------------------------------------------------
DATA_DIR = r"/home/gz0779601/TrafficSimulation/data/h5_womd_sim_agent"
# DATA_DIR = r"/media/yangyh408/4A259082626F01B9/h5_womd_sim_agent"
CKPG_NAME = "2cti1q5z"
CKPG_VER = "v41"
# CKPG_NAME = "285yb3yb"
# CKPG_VER = "v59"

#-------------------------------------------------------

# 加载配置信息到变量
hydra.initialize(config_path="../configs/")
config = hydra.compose(config_name="run.yaml")

# 加载数据集
single_data = DataH5womd(
    data_dir = DATA_DIR,
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
ckpt_path = f"src/ckpt/{CKPG_NAME}_{CKPG_VER}.ckpt"
checkpoint_version = f"yangyh408/traffic_bots/{CKPG_NAME}:{CKPG_VER}"
modelClass = hydra.utils.get_class(config.model._target_)
model = modelClass.load_from_checkpoint(
    ckpt_path, wb_artifact=checkpoint_version, **config.resume.model_overrides
)

target_batch_idx = 1
for batch_idx, batch in enumerate(val_dataloader):
    if target_batch_idx == None or batch_idx == target_batch_idx:
        # 使用 LightningModule 的 validation_step 进行单 batch 验证
        model.eval()
        with torch.no_grad():
            model.manual_test(batch=batch, batch_idx=batch_idx)
        break