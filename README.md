# MLforCOE
Machine learning modelling for novel antibiotics domains - here conjugated oligoelectrolytes

===========
## Description

Codes and data are described in the connecting article:

Armi Tiihonen, Sarah J. Cox-Vazquez, Qiaohao Liang, Mohamed Ragab, Zekun Ren, Noor Titan Putri Hartono, Zhe Liu, Shijing Sun, Senthilnath Jayavelu, Guillermo C. Bazan, and Tonio Buonassisi, "Predicting antimicrobial activity of conjugated oligoelectrolyte molecules via machine learning" (submitted for review 2021).

Molecule data in this repository is provided by Sarah J. Cox-Vazquez and Guillermo C. Bazan.

## Installation
To install, just clone this repository and chemprop repository:

`$ git clone https://github.com/PV-Lab/MLforCOE.git`

`$ cd MlforCOE`

`$ git clone https://github.com/chemprop/chemprop .`

To install the required packages, create a virtual environment using Anaconda (Optional but recommended setup):

`$ conda env create -f environment.yml`

`$ conda activate mlforcoe`

Try the desired parts of the project:
- Main_training_models.py: Trains RF, XGB, and GP models with downselected molecular fingerprints and reference fingerprints.
- Main_downselection.py: Repeats molecular descriptor downselection for the data and trains RF models at each stage of downselection.
- SHAP_for_RF_analysis.ipynb: Investigate the final RF model trained with Opt. fingerprint using SHAP analysis.
- Train_test_chemprop_models.sh: Train and test DMPNN and ffNN models. Running this code may take an hour or so, therefore it is better to run on a server.
- RFE_RF_run.sh: Run RFE for Cor. descriptors. Running this code may take an hour or so, therefore it is better to run on a server.
- HO_RF_init_var_cor.sh: Hyperparameter optimization for RF. Running this code may take an hour or so, therefore it is better to run on a server.
- Data: All the data files required for running the codes and created during running the codes.
- Results: All the resulting figures created when running the codes. 

## Authors
||                    |
| ------------- | ------------------------------ |
| **AUTHORS**      | Armi Tiihonen, Qiaohao Liang     | 
| **VERSION**      | 1.0 / April, 2021     | 
| **EMAILS**      | armi.tiihonen@gmail.com, hqliang@mit.edu | 
||                    |

## Attribution
This work is under a MIT License. Please, acknowledge use of this work with the appropiate citation to the repository and research article.

## Citation

    @Misc{mlforcoe2021,
      author =   {The MLforCOE authors},
      title =    {{MLforCOE}: Machine learning modelling for novel antibiotics domains},
      howpublished = {\url{https://github.com/PV-Lab/MLforCOE}},
      year = {2021}
    }
    
    {To be added: citation of the article}
