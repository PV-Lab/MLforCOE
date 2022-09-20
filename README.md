# MLforCOE
Machine learning modelling for novel antibiotics domains - here conjugated oligoelectrolytes

===========
## Description

Codes and data are described in the connecting article:

Armi Tiihonen, Sarah J. Cox-Vazquez, Qiaohao Liang, Mohamed Ragab, Zekun Ren, Noor Titan Putri Hartono, Zhe Liu, Shijing Sun, Cheng Zhou, Nathan C. Incandela, Jakkarin Limwonguyt, Alex S. Moreland, Senthilnath Jayavelu, Guillermo C. Bazan, and Tonio Buonassisi (2021). "Predicting antimicrobial activity of conjugated oligoelectrolyte molecules via machine learning". Journal of the American Chemical Society, 143(45), 18917-18931. https://doi.org/10.1021/jacs.1c05055

Arxiv preprint:

Armi Tiihonen, Sarah J. Cox-Vazquez, Qiaohao Liang, Mohamed Ragab, Zekun Ren, Noor Titan Putri Hartono, Zhe Liu, Shijing Sun, Cheng Zhou, Nathan C. Incandela, Jakkarin Limwonguyt, Alex S. Moreland, Senthilnath Jayavelu, Guillermo C. Bazan, and Tonio Buonassisi. "Predicting antimicrobial activity of conjugated oligoelectrolyte molecules via machine learning". arXiv:2105.10236 [physics.app-ph] (Nov. 2021). https://arxiv.org/abs/2105.10236

Molecule data in this repository is provided by Sarah J. Cox-Vazquez, Cheng Zhou, Nathan C. Incandela, Jakkarin Limwonguyt, Alex S. Moreland, and Guillermo C. Bazan.

## Installation
To install, just clone this repository and chemprop repository:

`$ git clone https://github.com/PV-Lab/MLforCOE.git`

`$ cd MlforCOE`

`$ git clone https://github.com/chemprop/chemprop .`

Extract Data_and_models.zip into MLforCOE folder. To install the required packages, create a virtual environment using Anaconda (Optional but recommended setup):

`$ conda env create -f environment.yml`

`$ conda activate mlforcoe`

Try the desired parts of the project:
- Main_downselection.py: Generates datafiles for other codes. Repeats molecular descriptor downselection for the data and trains RF models at each stage of downselection.
- Main_training_models.py: Trains RF, XGB, and GP models with downselected molecular fingerprints and reference fingerprints.
- SHAP_for_RF_analysis.ipynb: Investigate the final RF model trained with Opt. fingerprint using SHAP analysis.
- Main_train_test_chemprop_models_stratified_split.sh: Train and test DMPNN and ffNN models. Running this code may take an hour or so, therefore it is better to run on a server. Alternatively, the fully trained model is available by request from the authors (not included into this repository due to its large size).
- Main_plot_chemprop_models_and_violins.py: Plot neural network model results and all the violin plots. Works only after Main_train_test_chemprop_models_stratigfied_split.sh has been run.
- RFE_RF_run.sh: Run RFE for Cor. descriptors. Running this code may take an hour or so, therefore it is better to run on a server.
- HO_RF_init_var_cor.sh: Hyperparameter optimization for RF. Running this code may take an hour or so, therefore it is better to run on a server.
- Results: All the resulting figures created when running the codes.
- Pool BO RF Opt.ipynb, Performance visualization COE-RF.ipynb, Manifold visualization.ipynb: pool Bayesian optimization implementation and histograms for the COE data (adapted from https://github.com/PV-Lab/Benchmarking, please follow the up to date installation instructions in that repository to run the notebooks).
- TSNE_prediction_analysis_seed3_to_github.py: Supplementary Materials similarity plots

## Authors
||                    |
| ------------- | ------------------------------ |
| **AUTHORS**      | Armi Tiihonen, Qiaohao Liang     | 
| **VERSION**      | 2.0 / September, 2021     | 
| **EMAILS**      | armi.tiihonen@gmail.com, hqliang@mit.edu | 
||                    |

## Attribution
This work is under BSD-2-Clause License. Please, acknowledge use of this work with the appropiate citation to the repository and research article.

## Citation

    @Misc{mlforcoe2021,
      author =   {The MLforCOE authors},
      title =    {{MLforCOE}: Machine learning modelling for novel antibiotics domains},
      howpublished = {\url{https://github.com/PV-Lab/MLforCOE}},
      year = {2021}
    }
    
    @article{Tiihonen_2021,
	  doi = {10.1021/jacs.1c05055},
      url = {https://doi.org/10.1021%2Fjacs.1c05055},
      year = 2021,
	  month = {nov},
      publisher = {American Chemical Society ({ACS})},
      volume = {143},
      number = {45},
      pages = {18917--18931},
      author = {Armi Tiihonen and Sarah J. Cox-Vazquez and Qiaohao Liang and Mohamed Ragab and Zekun Ren and Noor Titan Putri Hartono and Zhe Liu and Shijing Sun and Cheng Zhou and Nathan C. Incandela and Jakkarin Limwongyut and Alex S. Moreland and Senthilnath Jayavelu and Guillermo C. Bazan and Tonio Buonassisi},
      title = {Predicting Antimicrobial Activity of Conjugated Oligoelectrolyte Molecules via Machine Learning},
      journal = {Journal of the American Chemical Society}
    }
