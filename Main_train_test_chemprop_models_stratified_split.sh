#!/bin/bash
#Created on Fri Jun  5 13:44:34 2020
#
#This code trains the following models and performs test set predictions:
#    1. DMPNN with Opt/cor/rdkit/morgancount/morgan/no features with and without HO.
#    2. Same for ffNN.
#chmod 700 Train_chemprop_models.sh
#@author: armi
#

#SBATCH -o chemprop_results/Results/Run_trains.log-%j                                                                                                                                             
#SBATCH -c 20                                                                                                                                                         
#--gres=gpu:volta:1                                                                                                                                                   

source /etc/profile
module load anaconda/2020b
module load cuda/11.0
#source "/home/gridsan/tiihonen/.conda/envs/coe-dmpnn-tl/bin"                                                                                                         
#conda init bash                                                                                                                                                      
#conda activate coe-dmpnn-tl                                                                                                                                          
source activate coe-dmpnn-tl
#unset PYTHONPATH                                                                                                                                                     
#export PYTHONNOUSERSITE=True
run=21
task=0

numGpus=20 # number of available gpus
gpu=0 # gpu num to run on

numTestFolds=20
dataName="opt" # Input y file, same for all the files.
seed="3"
dataPathTrain="Data/Downselection_data_files/y_${dataName}_train_seed${seed}.csv"
dataPathTest="Data/Downselection_data_files/y_${dataName}_test_seed${seed}.csv"
dataPathNewdata="Data/Downselection_data_files/y_${dataName}_newdata.csv"

smilesColumn="smiles"
targetColumn="log2mic"
valFoldIndex=1
testFoldIndex=2
dataType=regression

# These need to be defined every time the fingerprint changes - or, in case of Opt. and Cor., every time that fingerprint or task changes.
featuresPathRdkit="--features_generator rdkit_2d_normalized --no_features_scaling"
featuresPathMorgan="--features_generator morgan"
featuresPathMorganCount="--features_generator morgan_count"
featuresPathDefault=""
# featuresPathOpt="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
# featuresPathCor="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"

#########################################
modelName="DMPNN"
echo "DMPNN with Opt. features"
fingerprintName="opt"
echo "No HO"
hyperOpt="standard"
configPath=""

taskType="train"
featuresPathOpt="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
featuresPath=$featuresPathOpt
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}

echo "With HO"
hyperOpt="ho"
configPath="--config_path chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/ho.json"

taskType="train"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
#####################################################################################

#########################################
modelName="DMPNN"
echo "DMPNN with Cor. features"
fingerprintName="cor"
echo "No HO"
hyperOpt="standard"
configPath=""

taskType="train"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}

echo "With HO"
hyperOpt="ho"
configPath="--config_path chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/ho.json"

taskType="train"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
#####################################################################################

#########################################
modelName="DMPNN"
echo "DMPNN on Biol."
fingerprintName="rdkit"
echo "No HO"
hyperOpt="standard"
configPath=""

taskType="train"
featuresPath=$featuresPathRdkit
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}

echo "With HO"
hyperOpt="ho"
configPath="--config_path chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/ho.json"

taskType="train"
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
#####################################################################################


#########################################
modelName="DMPNN"
echo "DMPNN on Morgan"
fingerprintName="morgan"
echo "No HO"
hyperOpt="standard"
configPath=""

taskType="train"
featuresPath=$featuresPathMorgan
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}

echo "With HO"
hyperOpt="ho"
configPath="--config_path chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/ho.json"

taskType="train"
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
#####################################################################################




#########################################
modelName="DMPNN"
echo "DMPNN on Morgancount"
fingerprintName="morgancount"
echo "No HO"
hyperOpt="standard"
configPath=""

taskType="train"
featuresPath=$featuresPathMorganCount
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}

echo "With HO"
hyperOpt="ho"
configPath="--config_path chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/ho.json"

taskType="train"
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
#####################################################################################

#########################################
modelName="DMPNN"
echo "DMPNN with no extra features"
fingerprintName="no"
echo "No HO"
hyperOpt="standard"
configPath=""

taskType="train"
featuresPath=$featuresPathDefault
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}

echo "With HO"
hyperOpt="ho"
configPath="--config_path chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/ho.json"

taskType="train"
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
#####################################################################################

#####################################################################################

#####################################################################################

#####################################################################################

#####################################################################################

#########################################
modelName="ffNN"
echo "ffNN with Opt. features"
fingerprintName="opt"
echo "No HO"
hyperOpt="standard"
configPath=""

taskType="train"
featuresPathOpt="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
featuresPath=$featuresPathOpt
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --features_only --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}

echo "With HO"
hyperOpt="ho"
configPath="--config_path chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/ho.json"

taskType="train"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --features_only --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
#####################################################################################

#########################################
modelName="ffNN"
echo "ffNN with Cor. features"
fingerprintName="cor"
echo "No HO"
hyperOpt="standard"
configPath=""

taskType="train"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --features_only --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}

echo "With HO"
hyperOpt="ho"
configPath="--config_path chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/ho.json"

taskType="train"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --features_only --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}_seed${seed}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
featuresPath="--features_path Data/Downselection_data_files/x_${fingerprintName}_${taskType}.csv"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
#####################################################################################

#########################################
modelName="ffNN"
echo "ffNN on Biol."
fingerprintName="rdkit"
echo "No HO"
hyperOpt="standard"
configPath=""

taskType="train"
featuresPath=$featuresPathRdkit
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --features_only --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}

echo "With HO"
hyperOpt="ho"
configPath="--config_path chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/ho.json"

taskType="train"
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --features_only --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
#####################################################################################


#########################################
modelName="ffNN"
echo "ffNN on Morgan"
fingerprintName="morgan"
echo "No HO"
hyperOpt="standard"
configPath=""

taskType="train"
featuresPath=$featuresPathMorgan
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --features_only --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}

echo "With HO"
hyperOpt="ho"
configPath="--config_path chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/ho.json"

taskType="train"
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --features_only --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
#####################################################################################




#########################################
modelName="ffNN"
echo "ffNN on Morgancount"
fingerprintName="morgancount"
echo "No HO"
hyperOpt="standard"
configPath=""

taskType="train"
featuresPath=$featuresPathMorganCount
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --features_only --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}

echo "With HO"
hyperOpt="ho"
configPath="--config_path chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/ho.json"

taskType="train"
checkpointDir="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/checkpoints_${hyperOpt}"

for ((i=0;i<$numTestFolds;i++)); do
    saveDir="${checkpointDir}/fold_${i}"
    foldsFile="Data/Downselection_data_files/CV_splits/Seed${seed}/stratified_split_indices_cv${i}_train.pckl"
    #gpu=$((($gpu + 1) % $numGpus))
    CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path ${dataPathTrain} --dataset_type $dataType ${featuresPath} --save_dir ${saveDir} --split_type predetermined --folds_file ${foldsFile} --val_fold_index $valFoldIndex --test_fold_index $testFoldIndex --smiles_column ${smilesColumn} --target_column ${targetColumn} --show_individual_scores --features_only --extra_metrics r2 ${configPath}&
done
wait
echo "Predict test set"
taskType="test"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathTest} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
echo "Predict new data"
taskType="newdata"
predsPath="chemprop_models_stratsplit/Models/${modelName}/${fingerprintName}features/preds_${hyperOpt}_${taskType}.csv"
python predict.py --test_path ${dataPathNewdata} --checkpoint_dir ${checkpointDir} --preds_path ${predsPath} ${featuresPath} --smiles_column ${smilesColumn}
#####################################################################################

