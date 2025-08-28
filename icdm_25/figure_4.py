import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, MaxNLocator
from matplotlib.colors import LogNorm

from utils import get_libs

# SET PLOT PARAMETERS
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 40
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['axes.titlesize'] = 40
plt.rcParams['xtick.labelsize'] = 30 # 40
plt.rcParams['ytick.labelsize'] = 30# 40
plt.rcParams['legend.fontsize'] = 40
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# LOAD DATA
data_file = './data/neighborhood_embeddings_4.npz'
nbhd_vecs = np.load(data_file, allow_pickle=True)
# what arrays are in the file?
print("Arrays in the file:", nbhd_vecs.files)

nbhd_expected = nbhd_vecs['NeighborhoodExpected']
for i, elem in enumerate(tqdm(nbhd_expected)):
    if len(elem) != 1_265_170:
        print(f"Error: Expected {i} vector length is not 1265170, but", len(elem))
        exit(0)
    sparse_vector = sp.csr_matrix(elem)
    nbhd_expected[i] = sparse_vector
nbhd_expected = sp.vstack(nbhd_expected)
print("Shape of the sparse matrix:", nbhd_expected.shape)
print("Number of non-zero elements in the sparse matrix:", nbhd_expected.nnz)


df_9th = pd.read_csv("./data/df_9th.csv")
df_12th = pd.read_csv("./data/df_12th.csv")
df_13th = pd.read_csv("./data/df_13th.csv")
df_16th = pd.read_csv("./data/df_16th.csv")

lib1_merged, lib2_merged, df_all, lib1_overlap, lib2_overlap = get_libs(df_9th, df_12th, df_13th, df_16th, clean=True)
# drop any sequences longer than 46
_, _, df_raw, _, _ = get_libs(df_9th, df_12th, df_13th, df_16th, clean=False)


# merge df_raw and 
# get the df_raw that aren't in df_all
df_missing = df_raw[~df_raw['Sequence'].isin(df_all['Sequence'])]
df_both = df_raw[df_raw['Sequence'].isin(df_all['Sequence'])]

#x = df_all["Redist_Count"].values
XMIN = df_all["Redist_Count"].min()
XMAX = df_all["Redist_Count"].max()
YMIN = df_all["Pressure"].min()
YMAX = df_all["Pressure"].max()


top_enrichment = np.percentile(df_all['Pressure'], 99)
bottom_enrichment = np.percentile(df_all['Pressure'], 99)
top_enrichment_indices = df_all[df_all['Pressure'] >= top_enrichment].index.values
bottom_enrichment_indices = df_all[df_all['Pressure'] < bottom_enrichment].index.values

top_cpm = np.percentile(df_all['Redist_Count'], 98)
bottom_cpm = np.percentile(df_all['Redist_Count'], 98)
top_cpm_indices = df_all[df_all['Redist_Count'] >= top_cpm].index.values
bottom_cpm_indices = df_all[df_all['Redist_Count'] < bottom_cpm].index.values

print(f"Top Enrichment: {top_enrichment} ({len(top_enrichment_indices)})")
print(f"Bottom Enrichment: {bottom_enrichment} ({len(bottom_enrichment_indices)})")
print(f"Top Redist_Count: {top_cpm} ({len(top_cpm_indices)})")
print(f"Bottom Redist_Count: {bottom_cpm} ({len(bottom_cpm_indices)})")



# intersect top cpm and bottom enrichment
intersect_good = list(np.intersect1d(bottom_cpm_indices, top_enrichment_indices))
intersect_bad = list(np.intersect1d(top_cpm_indices, bottom_enrichment_indices))

# we need to find the intersect_good indices referenced from df_raw
df_all_good_sequences = df_all.loc[intersect_good, 'Sequence'].values
df_all_bad_sequences = df_all.loc[intersect_bad, 'Sequence'].values

df_raw_good_sequences = df_raw[df_raw['Sequence'].isin(df_all_good_sequences)]
df_raw_bad_sequences = df_raw[df_raw['Sequence'].isin(df_all_bad_sequences)]

mean_enrichment = df_both['Pressure'].mean()
stdv_enrichment = df_both['Pressure'].std()

mean_cpm = df_both['Redist_Count'].mean()
stdv_cpm = df_both['Redist_Count'].std()

indices_good = df_raw[(df_raw['Pressure'] >= top_enrichment) & (df_raw['Sequence'].isin(df_all['Sequence']))
                     & (df_raw['Redist_Count'] <= bottom_cpm)
                      ].index.values

indices_bad = df_raw[(df_raw['Pressure'] <= bottom_enrichment) & (df_raw['Sequence'].isin(df_all['Sequence']))
                     & (df_raw['Redist_Count'] >= top_cpm)
                     ].index.values

null_check = list(np.intersect1d(bottom_enrichment_indices, top_enrichment_indices))


from sklearn.decomposition import NMF
n_components = 25
print(f"Starting NMF with {n_components} components...")
start = time.time()
nmf = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=1000)
print('Shape of input matrix:', nbhd_expected.shape)
W = nmf.fit_transform(nbhd_expected)
stop = time.time()
print(f"NMF completed in {stop - start:.2f} seconds.")
H = nmf.components_
print("Shape of W:", W.shape)
print("Shape of H:", H.shape)



# use HDBSCAN to cluster W
#import hdbscan
#print("Starting HDBSCAN...")
#start = time.time()
#clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=1, metric='euclidean')
#cluster_labels = clusterer.fit_predict(W)
#stop = time.time()
#print(f"HDBSCAN completed in {stop - start:.2f} seconds.")
#print("Number of clusters found:", len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0))

# use spectral clustering to cluster W
from sklearn.cluster import SpectralClustering
print("Starting Spectral Clustering...")
start = time.time()
n_clusters = 45
# let's find a score for each n_clusters from 5 to n_clusters
from sklearn.metrics import silhouette_score
best_n_clusters = 5
best_silhouette = -1
for n_cluster in range(5,n_clusters):
    spectral = SpectralClustering(n_clusters=n_cluster, affinity='nearest_neighbors', random_state=42)
    cluster_labels = spectral.fit_predict(W)
    silhouette_avg = silhouette_score(W, cluster_labels)
    print(f"For n_clusters = {n_cluster}, the average silhouette_score is : {silhouette_avg}")
    if silhouette_avg > best_silhouette:
        best_silhouette = silhouette_avg
        best_n_clusters = n_cluster
stop = time.time()
print(f"Spectral Clustering completed in {stop - start:.2f} seconds.")
print(f"Best n_clusters is {best_n_clusters} with silhouette score {best_silhouette}")

spectral = SpectralClustering(n_clusters=best_n_clusters, affinity='nearest_neighbors', random_state=42)
cluster_labels = spectral.fit_predict(W)


# count anomalies per cluster
from collections import Counter
anomaly_clusters = cluster_labels[np.concatenate((indices_good, indices_bad))]
anomaly_cluster_counts = Counter(anomaly_clusters)
print("Anomaly cluster counts:", anomaly_cluster_counts)
# count good and bad anomalies per cluster
good_anomaly_clusters = cluster_labels[indices_good]
bad_anomaly_clusters = cluster_labels[indices_bad]
good_anomaly_cluster_counts = Counter(good_anomaly_clusters)
bad_anomaly_cluster_counts = Counter(bad_anomaly_clusters)
print("Good anomaly cluster counts:", good_anomaly_cluster_counts)
print("Bad anomaly cluster counts:", bad_anomaly_cluster_counts)


# do t-SNE on W
from sklearn.manifold import TSNE
print("Starting t-SNE...")
start = time.time()
tsne = TSNE(n_components=2, random_state=42, perplexity=50)
W_tsne = tsne.fit_transform(W)
stop = time.time()
print(f"t-SNE completed in {stop - start:.2f} seconds.")
print("Shape of t-SNE result:", W_tsne.shape)

from sklearn.decomposition import PCA
#print("Starting PCA...")
#start = time.time()
#pca = PCA(n_components=2, random_state=42)
#W_tsne = pca.fit_transform(W)
#stop = time.time()
#print(f"PCA completed in {stop - start:.2f} seconds.")

# scatter plot of t-SNE result colored by Pressure
plt.figure(figsize=(16, 9))
# color by cluster labels
# create a colormap with a different color for each cluster
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
cmap = plt.get_cmap('tab20', num_clusters)
# create a scatter plot with colors based on cluster labels
sc = plt.scatter(W_tsne[:, 0], W_tsne[:, 1], s=80, alpha=1.0, c=cluster_labels, cmap=cmap, marker='o', lw=0)
# scatter green x on good anomalies
plt.scatter(W_tsne[indices_good, 0], W_tsne[indices_good, 1], s=100, c='green', marker='X', label='LC-HP Anomalies')
# scatter red x on bad anomalies
plt.scatter(W_tsne[indices_bad, 0], W_tsne[indices_bad, 1], s=100, c='red', marker='X', label='HC-LP Anomalies')
plt.legend(loc='upper center', fontsize=30, markerscale=2.0, ncols=2, bbox_to_anchor=(.5,1.145), handletextpad=0.2, handlelength=0.6)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.xlim(-65,65)
plt.ylim(-60,60)
#plt.axes('off')

print(f"Total data {(len(df_raw))}")
print(f"Adversaries {len(df_missing)}")
print(f"Good Anomalies {len(indices_good)}")
print(f"Bad Anomalies {len(indices_bad)}")

plt.savefig('./data/icdm_figure_4.png', format='png', bbox_inches='tight')
