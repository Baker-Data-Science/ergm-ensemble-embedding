import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tqdm import tqdm

import matplotlib.pyplot as plt


# SET PLOT PARAMETERS
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 40
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['axes.titlesize'] = 40
plt.rcParams['xtick.labelsize'] = 20 # 40
plt.rcParams['ytick.labelsize'] = 20# 40
plt.rcParams['legend.fontsize'] = 40
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# LOAD DATA
#df_9th = pd.read_csv("./data/df_9th.csv")
#df_12th = pd.read_csv("./data/df_12th.csv")
#df_13th = pd.read_csv("./data/df_13th.csv")
#df_16th = pd.read_csv("./data/df_16th.csv")

#lib1_merged, lib2_merged, dataframe, lib1_overlap, lib2_overlap = get_libs(df_9th, df_12th, df_13th, df_16th, clean=False)

#dataframe = parse_data_frame(dataframe)
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


"""
nbhd_presenceprob = nbhd_vecs['NeighborhoodPresenceProb']
for i, elem in enumerate(tqdm(nbhd_presenceprob)):
    if len(elem) != 1_265_170:
        print(f"Error: Presence probability {i} vector length is not 1265170, but", len(elem))
        exit(0)
    sparse_vector = sp.csr_matrix(elem)
    nbhd_presenceprob[i] = sparse_vector
nbhd_presenceprob = sp.vstack(nbhd_presenceprob)
print("Shape of the sparse matrix:", nbhd_presenceprob.shape)
print("Number of non-zero elements in the sparse matrix:", nbhd_presenceprob.nnz)
"""

del nbhd_vecs

#dataframe['NeighborhoodExpected'] = nbhd_expected
#dataframe['NeighborhoodPresenceProb'] = nbhd_presenceprob


# do nmf on nbhd_expected
from sklearn.decomposition import NMF
n_components = 25
print(f"Starting NMF with {n_components} components...")
start = time.time()
nmf = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=100)
W = nmf.fit_transform(nbhd_expected)
stop = time.time()
print(f"NMF completed in {stop - start:.2f} seconds.")
H = nmf.components_
print("Shape of W:", W.shape)
print("Shape of H:", H.shape)

# do t-SNE on W
from sklearn.manifold import TSNE
#print("Starting t-SNE...")
#start = time.time()
#tsne = TSNE(n_components=2, random_state=42, perplexity=50)
#W_tsne = tsne.fit_transform(W)
#stop = time.time()
#print(f"t-SNE completed in {stop - start:.2f} seconds.")
#print("Shape of t-SNE result:", W_tsne.shape)

from sklearn.decomposition import PCA
print("Starting PCA...")
start = time.time()
pca = PCA(n_components=2, random_state=42)
W_tsne = pca.fit_transform(W)
stop = time.time()
print(f"PCA completed in {stop - start:.2f} seconds.")


# scatter plot of t-SNE result colored by Pressure
plt.figure(figsize=(16, 9))
sc = plt.scatter(W_tsne[:, 0], W_tsne[:, 1], s=5, alpha=0.7)
plt.colorbar(sc, label='Pressure')
plt.title('t-SNE of Neighborhood Embeddings Colored by Pressure')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.savefig('./data/figure_2.png', bbox_inches='tight')
