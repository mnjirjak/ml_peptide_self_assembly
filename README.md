# Machine learning for peptide self-assembly
> <div align="justify">Peptide self-assembly is of crucial importance for obtaining advanced supramolecular materials with rich chemical and structural properties. Here, we present a novel recurrent neural network-based method to evaluate the self-assembly potential of unclassified peptides with varying lengths using only peptide sequences as input. The key advantage of the approach lies in leveraging irregularly sampled features, specifically the aggregation propensity scores of amino acids, dipeptides, and tripeptides, as predictor variables for any given peptide under consideration. To assess the effectiveness of the proposed machine learning models, we employed a genetic algorithm-based generative approach to create new sequences with a high affinity towards self-assembly. Our models serve as a knowledgeable tool for complementing human intuition in an endeavour to find novel self-assembling compounds.</div>

## System requirements
A standard desktop computer with 4+ GB RAM and a single-core, 2+ GHz CPU should suffice to run the algorithms. No non-standard hardware is necessary. `Conda` package manager is required to set up a Python environment and install dependencies. Software dependencies along with version numbers are detailed in the ```ml_peptide_self_assembly.yml``` file. The code was tested on Ubuntu 22.04 and Windows 11 operating systems.

## Environment setup &amp; activation
Use `conda` and `ml_peptide_self_assembly.yml` to create an environment. This usually takes about 5 minutes.

    git clone https://github.com/mnjirjak/ml_peptide_self_assembly.git
    cd ml_peptide_self_assembly
    conda env create -f ml_peptide_self_assembly.yml
    conda activate ml_peptide_self_assembly

## Important notes
- The code should be run from the root (```ml_peptide_self_assembly```) directory with the conda environment active.
- If you want to train your model and perform hyperparameter optimization, modify the peptide list and labels in ```SA_ML_predictive/data/data_SA.csv```, and alter the hyperparameter values in the header of ```SA_ML_predictive/code/automate_training.py``` before running the scripts.
- Prediction time varies with the model used but usually takes about 15 seconds.
- The generative model takes about 5 minutes to suggest novel peptides.

## Predictive model(s)
Predictive models take a peptide sequence, represented by amino acid single-letter codes, and output the probability of the sequence exhibiting self-assembly.

                   ┌────────┐
    KFFAKK ───────►│ML model├─────► 98.3%
                   └────────┘

#### Use one of the existing, pre-trained models:

    python SA_ML_predictive/code/predict.py --sequence <peptide_sequence> --ml-model <AP, AP_SP, SP, TSNE_AP_SP, TSNE_SP>

The resulting probability of a single sequence exhibiting self-assembly will be visible in the command line.

#### Running hyperparameter selection for the model(s):

    python SA_ML_predictive/code/hyperparameter_selection.py --ml-model <AP, AP_SP, SP, TSNE_AP_SP, TSNE_SP>

#### Train the model(s) with the chosen hyperparameters:

    python SA_ML_predictive/code/training.py --ml-model <AP, AP_SP, SP, TSNE_AP_SP, TSNE_SP> --num-cells <an integer value> --kernel-size <an integer value>

The resulting model will be saved in the ```SA_ML_predictive/models``` directory and will overwrite an existing model.

## Generative model
The generative model is a machine-learning-guided genetic algorithm capable of efficiently exploring peptide chemical space and creating novel sequences.

    ┌────────┐
    │GA model├─────► FKFEFFKF, GFALLAGKK, AAVVLWEEEE, ...
    └────────┘

#### Use the generative model with one of the existing predictive models:

    python SA_ML_generative/find_novel_peptides.py --ml-model <AP, AP_SP, SP, TSNE_AP_SP, TSNE_SP>

#### Use the generative model with your predictive model:

    python SA_ML_generative/find_novel_peptides.py --ml-model-path <path to your .h5 file>

The results will be saved in a ```csv``` file. Additional parameters of the generative model, such as mutation probability or a list of allowed amino acids, can be modified in the header of the ```SA_ML_generative/find_novel_peptides.py``` script.
