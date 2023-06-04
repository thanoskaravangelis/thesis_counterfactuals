#!/bin/bash
#
#SBATCH --job-name=beam120news # Όνομα για διαχωρισμό μεταξύ jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=56G
#SBATCH --output=runs/outputs/news_beam120.out.log
#SBATCH --error=runs/errors/news_beam120.error.log
#SBATCH --account=pa210503
#SBATCH -t 4-00:00:00 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


cd /users/pa21/ptzouv/tkaravangelis/mice_grad
module purge
module load gnu/8 cuda/10.1.168 intelmpi/2018 pytorch/1.7.0
source /users/pa21/ptzouv/tkaravangelis/venv/bin/activate

start=$(date +%s.%N)
srun python3 /users/pa21/ptzouv/tkaravangelis/scripts/run_news_beam120.py
deactivate

end=$(date +%s.%N)
runtime=$( echo "$end - $start" | bc -l )
echo "Total script time $runtime"
