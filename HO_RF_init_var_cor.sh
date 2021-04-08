#!/bin/bash
#SBATCH -o Run_HO_RF1.log-%j                                                           
#SBATCH -c 20

source /etc/profile
module load anaconda/2020b
module load cuda/11.0
#source "/home/gridsan/tiihonen/.conda/envs/coe-dmpnn-tl/bin"                                                                                    
#conda activate coe-dmpnn-tl                                                                                                                                          
#source activate coe-dmpnn-tl
#unset PYTHONPATH                                                                                                                                                     
#export PYTHONNOUSERSITE=True
run=1
task=0

echo "Run${run}-${task} $(date)" >> test_status_run${run}.txt
python -c "import sys; print(sys.executable); import torch; print(torch.cuda.is_available())"
echo "Run${run}-${task} ended $(date)" >> test_status_run${run}.txt
task=`expr $task + 1`
echo "Run${run}-${task} $(date)" >> test_status_run${run}.txt
python HO_RF_init_var_cor_datasets.py
echo "Run${run}-${task} ended $(date)" >> test_status_run${run}.txt
task=`expr $task + 1`
