#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24 
#SBATCH --nodes=1 
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpudeadline
#SBATCH --job-name=train_ftr
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "########################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Script arguments received by sbatch script ($0): $@"
echo "Number of arguments: $#"
echo "########################################################################"

# --- Environment Setup ---
echo "Setting up environment..."

# Navigate to your workspace directory. 
cd /mnt/personal/korcadav/flipper_training || {
    echo "Failed to cd into directory"
    exit 1
}
echo "Changed directory to $(pwd)"
echo "------------------------------------------------------------------------"

# --- Singularity Tmp Fix ---
# Create a local tmp folder in your workspace
mkdir -p /mnt/personal/korcadav/flipper_training/tmp_singularity

# Tell Singularity to use this folder
export SINGULARITY_TMPDIR=/mnt/personal/korcadav/flipper_training/tmp_singularity
export APPTAINER_TMPDIR=/mnt/personal/korcadav/flipper_training/tmp_singularity

# --- Execution ---
COMMAND="singularity exec --nv isaaclab.sif env PYTHONPATH=src/FTR-benchmark python src/flipper_training/flipper_training/experiments/ppo/train_ftr.py --config src/flipper_training/flipper_training/experiments/ppo/ftr_config.yaml --num_envs 256 --terrain cur_steps_up --task Marv-Crossing-Direct-v0 --headless $@"

echo "Executing command via srun:"
echo "------------------------------------------------------------------------"

srun $COMMAND

EXIT_STATUS=$?
echo "########################################################################"
echo "Python script finished with exit status: $EXIT_STATUS"
echo "Job finished."
echo "########################################################################"
exit $EXIT_STATUS