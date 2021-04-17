#!/bin/bash
#SBATCH -o Run_rfe_rf.log-%j                                                           
#SBATCH -c 8

source /etc/profile
module load anaconda/2020b
#source "/home/gridsan/tiihonen/.conda/envs/coe-dmpnn-tl/bin"                                                                                    
#conda activate coe-dmpnn-tl                                                                                                                                          
#source activate coe-dmpnn-tl
#unset PYTHONPATH                                                                                                                                                     
#export PYTHONNOUSERSITE=True
run=2
task=0

echo "Run${run}-${task} $(date)" >> test_status_run${run}.txt
python RFE_RF.py
echo "Run${run}-${task} ended $(date)" >> test_status_run${run}.txt
task=`expr $task + 1`
