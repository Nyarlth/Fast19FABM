#!/bin/bash

#SBATCH --job-name=sky_map             # Job name
#SBATCH --nodelist=node02   #gpu02     # Run all processes on a single node
#SBATCH --ntasks=1                   # Run n task
#SBATCH --cpus-per-task=20 #24           # Number of CPU cores per task
#SBATCH --mem=150gb                  # Job memory request
#SBATCH --time=10:00:00              # Time limit hrs:min:sec
#SBATCH --partition=batch # gpu if choose gpu node
#SBATCH --output=task_sky_map%j.log    # Standard output and error log

pwd; hostname; date
cd /home/xyzhao/code/Fast19FA/scripts
echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

python -u /home/xyzhao/code/Fast19FA/scripts/sky_map.py


date

