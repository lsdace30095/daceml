#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --account=g34
#SBATCH --output=0_autodiff_2nodes_encoder.txt


module load daint-gpu
module load cudatoolkit/11.2.0_3.36-7.0.2.1_2.2__g3ff9ab1
module load PyTorch
module unload gcc
module load gcc/8.3.0

#module unload cray-libsci/20.09.1
#module load CMake/3.14.5
#module load NCCL
source ~/.bashrc
source ./venv/bin/activate
which nvcc
nvidia-smi

which python
export MPICH_RDMA_ENABLED_CUDA=1
export PMPI_CUDAIPC_ENABLE=1
export PMPI_GPU_AWARE=1
#
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export NCCL_DEBUG=INFO
#export NCCL_IB_HCA=ipogif0
#export NCCL_IB_CUDA_SUPPORT=1
#export NCCL_SOCKET_IFNAME=ipogif0

#export CUDA_LAUNCH_BLOCKING=1


export ORT_ROOT=/scratch/snx3000/shigang/eager-SGD-artifact/ort_gpu/onnxruntime
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/snx3000/shigang/eager-SGD-artifact/daceml/daceml_master/daceml/tests/autodiff/pytorch
#export DACE_compiler_use_cache=1

which CC
srun python ./tests/autodiff/pytorch/test_bert_encoder_backward.py
