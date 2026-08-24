# Week 3 — Unsupervised Learning and Clustering Analysis

## Wine Profile Segmentation with K-Means and Hierarchical Clustering

This repository contains a reproducible Week 3 clustering project using the public **UCI Wine dataset (ID 109)**.

### Dataset
- 178 observations
- 13 continuous chemical features
- No missing values according to UCI
- CC BY 4.0
- DOI: 10.24432/C5PC7J
- Official source: https://archive.ics.uci.edu/dataset/109/winedataset

### Workflow
1. Initial data audit
2. StandardScaler preprocessing
3. K-Means evaluation for k=2 through k=8
4. Elbow and silhouette analysis
5. Selected k=3
6. PCA visualization
7. Ward hierarchical clustering comparison
8. Cluster profiling using original-scale means and standardized deviations
9. Post-hoc Adjusted Rand Index against known cultivar labels (labels were NOT used for training)

### Actual results
- Selected k: **3**
- Best silhouette: **0.285**
- Hierarchical silhouette: **0.277**
- PCA PC1 + PC2 variance explained: **55.4%**
- Cluster sizes: **{'0': 65, '1': 51, '2': 62}**

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/clustering_analysis.py
```
