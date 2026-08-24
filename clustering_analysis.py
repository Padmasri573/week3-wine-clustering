from pathlib import Path
import json
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)

wine=load_wine(as_frame=True)
X=wine.data.copy(); y=wine.target.copy()
X_scaled=StandardScaler().fit_transform(X)

rows=[]; models={}
for k in range(2,9):
    model=KMeans(n_clusters=k,random_state=42,n_init=20)
    labels=model.fit_predict(X_scaled)
    rows.append((k,model.inertia_,silhouette_score(X_scaled,labels)))
    models[k]=(model,labels)

metrics=pd.DataFrame(rows,columns=['k','inertia','silhouette'])
best_k=int(metrics.loc[metrics.silhouette.idxmax(),'k'])
model,labels=models[best_k]
pca=PCA(n_components=2,random_state=42)
pca_data=pca.fit_transform(X_scaled)

Z=linkage(X_scaled,method='ward')
hier=fcluster(Z,t=best_k,criterion='maxclust')-1

result=X.copy()
result['cluster']=labels
result['pca_1']=pca_data[:,0]
result['pca_2']=pca_data[:,1]
result.to_csv(OUT/'wine_cluster_assignments.csv',index=False)
metrics.to_csv(OUT/'k_selection_metrics.csv',index=False)

summary={
'selected_k':best_k,
'best_silhouette':float(metrics.loc[metrics.k==best_k,'silhouette'].iloc[0]),
'hierarchical_silhouette':float(silhouette_score(X_scaled,hier)),
'posthoc_adjusted_rand_index':float(adjusted_rand_score(y,labels)),
'pca_variance_explained':[float(v) for v in pca.explained_variance_ratio_]
}
(OUT/'analysis_metrics.json').write_text(json.dumps(summary,indent=2))
print(summary)
