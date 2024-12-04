conda config --append channels conda-forge
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install gym==0.21.0

pip install ipykernel
pip install hydra-core==1.1.1
pip install pytorch-lightning==1.5.10
pip install wandb
pip install transforms3d==0.3.1
pip install opencv-python==4.5.5.64
pip install gym==0.25.2
pip install imageio-ffmpeg==0.4.9
# pip install tensorflow==2.11.0
# pip install numpy==1.20.0
# pip install protobuf==3.20.0

pip install waymo-open-dataset-tf-2-12-0==1.6.4


# 接入limsim环境
pip install dearpygui==1.8.0
pip install sumolib==1.16.0
pip install traci==1.16.0
pip install rich==13.4.2
pip install pynput==1.7.6
pip install ray==2.30.0
