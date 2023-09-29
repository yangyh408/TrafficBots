#!/bin/bash
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.out
#SBATCH --time=120:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5000
#SBATCH --tmp=200000
#SBATCH --open-mode=truncate

trap "echo sigterm recieved, exiting!" SIGTERM

run () {
python -u src/pack_h5_womd.py --dataset=training \
--out-dir=/cluster/scratch/zhejzhan/h5_womd_sim_agent \
--data-dir=/cluster/scratch/zhejzhan/womd_scenario_v_1_2_0
}

# for validation and testing
# python -u scripts/pack_h5_womd.py --dataset=validation --rand-pos=-1 --rand-yaw=-1 \
# python -u scripts/pack_h5_womd.py --dataset=testing --rand-pos=-1 --rand-yaw=-1 \

source /cluster/project/cvl/zhejzhan/apps/miniconda3/etc/profile.d/conda.sh
conda activate traffic_bots

echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`


type run
echo START: `date`
run &
wait
echo DONE: `date`

mkdir -p ./logs/slurm
mv ./logs/$SLURM_JOB_ID.out ./logs/slurm/$SLURM_JOB_ID.out

echo finished at: `date`
exit 0;