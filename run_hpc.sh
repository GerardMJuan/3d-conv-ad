#!/bin/bash
#SBATCH -J 3dgpu
#SBATCH -p high
#SBATCH --workdir=/homedtic/gmarti/LOGS
#SBATCH --gres=gpu:1
#SBATCH --mem 10G
#SBATCH -o 3dcnn_%J.out # STDOUT
#SBATCH -e 3dcnn_%j.err # STDERR

source /etc/profile.d/lmod.sh
source /etc/profile.d/easybuild.sh
export PATH="$HOME/project/anaconda3/bin:$PATH"
source activate dlnn
module load CUDA/9.0.176
module load cuDNN/7.0.5-CUDA-9.0.176

python /homedtic/gmarti/CODE/3d-conv-ad/train.py --config_file /homedtic/gmarti/CODE/3d-conv-ad/configs/config.ini --output_directory_name full_train
