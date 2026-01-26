# MASSCL
MASSCL integrates multi-source attention and semi-supervised contrastive learning for circRNA-disease association prediction. First, similarity and meta-path-induced networks are constructed and encoded by independent GCNs to learn network-specific representations. Then, attention mechanisms are applied to adaptively fuse multi-source features. Next, semi-supervised contrastive learning is employed to enhance node representations by leveraging both labeled and unlabeled data. Finally, a gated interaction decoder is used to predict circRNA-disease associations.
# Requirements
- python (tested on version 3.8.20)
- pytorch (tested on version 1.11.0)
- torch-geometric (tested on version 2.0.4)
- numpy (tested on version 1.24.4)
- scikit-learn(tested on version 1.3.2)
# Running the Code
To reproduce our results:
Run python main.py to run MASSCL
# Data description
- c_d.csv: all pairs of circRNAs and diseases
- circname.txt: list of circRNA names
- disname.txt: list of disease names
- circFSim.csv: circRNA functional similarity
- disSSim.csv: disease semantic similarity
- circGIPSim.csv: circRNA GIP similarity
- disGIPSim.csv: disease GIP similarity
# Folder
- code: Model code of MASSCL
- datasets: Data required by MASSCL
- results: Results of MASSCL run
