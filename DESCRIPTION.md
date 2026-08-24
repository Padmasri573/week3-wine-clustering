# Humanized Project Description — Week 3

## Wine Profile Segmentation Using Unsupervised Learning

For this task, I selected the UCI Wine dataset because it gives a clear opportunity to demonstrate real unsupervised learning without making the analysis unnecessarily complicated. The dataset contains 178 wines described by 13 chemical measurements. UCI explains that the wines come from the same Italian region but from three different cultivars.

The key idea in this project is that the clustering algorithm is not told the cultivar. It only sees the chemical measurements. I therefore treated the cultivar label as a reference variable, not as an input.

The first step was a basic quality audit. The dataset had no missing values and no duplicate rows, so I did not invent an imputation strategy just for the sake of having one. Instead, I focused preprocessing on the issue that matters most for K-Means: scale. Because the chemical variables use different numerical ranges, I standardized them before calculating distances.

I tested K-Means with between 2 and 8 clusters. Rather than choosing a cluster count by guesswork, I compared inertia and silhouette scores. The best silhouette result was obtained with **k=3**, so that became the main solution.

I then used PCA to create a two-dimensional visual explanation of the clustering. PCA is only a visualization aid here; the actual K-Means model used all 13 standardized features.

As a second check, I ran Ward hierarchical clustering. This was useful because it approaches grouping differently from K-Means. The hierarchical result at k=3 produced a silhouette score of **0.277**, which provides another perspective on the structure.

Finally, I profiled the clusters using their original chemical measurements and standardized differences from the overall average. This makes the results interpretable instead of leaving them as anonymous cluster numbers.

The known cultivar labels were compared only after clustering using Adjusted Rand Index. The resulting ARI was **0.897**. This is not treated as classification accuracy; it is simply a post-hoc check of how closely the unsupervised grouping resembles the known structure.

The final repository contains the Python script, notebook, figures, cluster assignments, metrics, and DOCX report so the entire workflow can be reproduced and reviewed.
