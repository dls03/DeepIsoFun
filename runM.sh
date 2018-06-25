#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=3-00:15:00     # 1 day and 15 minutes
#SBATCH --output=mytesty.stdout
#SBATCH --job-name="rM"
#SBATCH -p intel # This is the default partition, you can use any of the following; intel, batch, highmem,

# Print current date
date


module load caffe
module load cuda/8.0
module load cuDNN/6.0
module load opencv
module load glog



#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:Path_to_openblas
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/linux/centos/7.x/x86_64/pkgs/gflags/2.1.2/lib/
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/linux/centos/7.x/x86_64/pkgs/leveldb/1.20/out-shared
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/linux/centos/7.x/x86_64/pkgs/glog/0.3.5/lib:/opt/linux/centos/7.x/x86_64/pkgs/cuDNN/6.0/lib64:/opt/linux/centos/7.x/x86_64/pkgs/cuda/8.0/lib64:/opt/linux/centos/7.x/x86_64/pkgs/cuda/8.0/nvvm/lib64:/opt/linux/centos/7.x/x86_64/pkgs/cuda/7.0/lib64:/opt/linux/centos/7.x/x86_64/pkgs/cuda/7.0/nvvm/lib64:/opt/linux/centos/7.x/x86_64/pkgs/cuda/7.0/lib:/opt/linux/centos/7.x/x86_64/pkgs/python/2.7.5/lib:/opt/linux/centos/7.x/x86_64/pkgs/ggobi/2.1.11/lib:/opt/linux/centos/7.x/x86_64/pkgs/R/3.4.0/lib64/R/lib:/opt/linux/centos/7.x/x86_64/pkgs/openmpi/2.0.1-slurm-16.05.4/lib:/opt/linux/centos/7.x/x86_64/pkgs/openmpi/2.0.1-slurm-16.05.4/lib/openmpi:/opt/linux/centos/7.x/x86_64/pkgs/slurm/16.05.4/lib:/rhome/dshaw003/bigdata/caffe/openblas/lib

#module list
#echo $LD_LIBRARY_PATH

#cd ~/bigdata/caffe-master/
#make clean
#make all
#make pycaffe


python DeepIsoFun/deepisofun3.py


#interactive session
#srun --x11 --mem=1gb --cpus-per-task 1 --ntasks 1 --time 10:00:00 --pty bash -l


#highmen interactive job
#srun -p highmem --mem=100g --time=24:00:00 --pty bash -l

#gpu job non-interactive
#sbatch -p gpu --gres=gpu:1 --mem=100g --time=1:00:00 SBATCH_SCRIPT.sh

#gpu job interactive
#srun -p gpu --gres=gpu:1 --nodelist=gpu01 --mem=100g --time=50:00:00 --pty bash -l
#srun -p gpu --gres=gpu:1 --mem=10g --time=50:00:00 --pty bash -l


