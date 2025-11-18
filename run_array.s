#!/bin/bash
#
#SBATCH --account=pr_351_general
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=gpu_cuda
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --mem=24GB
#SBATCH --output=/scratch/%u/%A:%x/%4a.out
#SBATCH --error=/scratch/%u/%A:%x/%4a.err
#SBATCH --mail-type=END

singularity exec \
  --nv \
  --overlay /home/$USER/overlay.ext3:ro \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
  /bin/bash
  # /bin/bash -c ". /ext3/conda/bin/activate 3dgs; python /home/$USER/softbinary-vae/run_hpc.py"
  /bin/bash
