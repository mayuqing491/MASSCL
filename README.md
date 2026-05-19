# IHCDA
IHCDA introduces auxiliary relational structures to enhance sparse similarity information by incorporating structural relational knowledge, and seamlessly integrates it into the representation learning process. Furthermore, heterogeneous multi-view representations are adaptively modeled via view-specific attention mechanisms, and a semi-supervised contrastive learning strategy is designed to jointly optimize intra-view and cross-view relationships on the enhanced representations. Extensive experimental results demonstrate that IHCDA achieves competitive performance in circRNA-disease association prediction. Furthermore, case studies validate its effectiveness, reliability, and stability in identifying potential associations. 
# Requirements
- python (tested on version 3.8.20)
- pytorch (tested on version 1.11.0)
- torch-geometric (tested on version 2.0.4)
- numpy (tested on version 1.24.4)
- scikit-learn(tested on version 1.3.2)
# Running the Code
To reproduce our results:
Run python main.py to run IHCDA
# Data description
- c_d.csv: all pairs of circRNAs and diseases
- circname.txt: list of circRNA names
- disname.txt: list of disease names
- circFSim.csv: circRNA functional similarity
- disSSim.csv: disease semantic similarity
- circGIPSim.csv: circRNA GIP similarity
- disGIPSim.csv: disease GIP similarity
# Folder
- code: Model code of IHCDA
- datasets: Data required by IHCDA
- results: Results of MASSCL run
