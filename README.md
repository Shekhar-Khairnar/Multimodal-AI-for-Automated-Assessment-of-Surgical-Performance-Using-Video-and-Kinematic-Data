# Multimodal AI for Automated Assessment of Surgical Performance

This repository contains the **final training and evaluation code** used for our surgical skill assessment experiments.  
It is intended to support **reproducibility of results** using **pre-calculated multimodal features** and **fixed evaluation splits**.

Raw videos and clinical datasets are **not included** due to privacy, and size constraints.

---

### Precomputed feature CSVs are available here: https://drive.google.com/drive/folders/1HCiXYOFNgAS5uSFPeBJXXW33KhsO0CpU?usp=sharing (download and place in `data/`)


### JIGSAWS Dataset: Details regarding access and the request procedure are available at: https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws/

## Repository Structure
```text
data/
  pre_calculated_features/
    knottying_all_features_combined.csv
    needle_passing_all_features_combined.csv
    suturing_all_features_combined.csv

splits/
  kfold4.json

src/
  jigsaws/
    train.py

requirements.txt
README.md
.gitignore


---

## Data Overview

Each CSV file contains **all modalities already aligned and flattened** per trial:

- **Kinematics features**  
  Columns containing `Left_` or `Right_`

- **Video features (SlowFast)**  
  Columns named `feature_*`

- **Density / heatmap features**  
  Columns named `density_feat_*`

- **Temporal information**  
  `duration_seconds`

- **Target label**  
  `grs_score`

- **Identifiers**  
  `subject`, `subject_id`

No additional preprocessing is required before training.

---

## Evaluation Splits

Fixed cross-validation splits are provided in:

data/splits/kfold4.json

These splits are used directly by the training script.

---

## Running Experiments

Install dependencies:

```bash
pip install -r requirements.txt


python src/jigsaws/train.py --task Suturing --splits data/splits/kfold4.json

