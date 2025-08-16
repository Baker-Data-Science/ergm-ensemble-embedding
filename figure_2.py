import re
import ast
from collections import defaultdict

import numpy as np
import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, MaxNLocator
from matplotlib.colors import LogNorm

from utils import get_libs, process_nbhd_vecs2, create_exp_nbhd_mat, parse_data_frame


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
df_9th = pd.read_csv("./data/df_9th.csv")
df_12th = pd.read_csv("./data/df_12th.csv")
df_13th = pd.read_csv("./data/df_13th.csv")
df_16th = pd.read_csv("./data/df_16th.csv")

df_9th = parse_data_frame(df_9th)
df_12th = parse_data_frame(df_12th)
df_13th = parse_data_frame(df_13th)
df_16th = parse_data_frame(df_16th)

lib1_merged, lib2_merged, df_all, lib1_overlap, lib2_overlap = get_libs(df_9th, df_12th, df_13th, df_16th)

# I need a Neighborhood_vecs that combines everything for all equences
cleaned_df = pd.merge(df_13th, df_9th, on='Sequence', how='left')
cleaned_df['Neighborhood_vecs'] = cleaned_df['Neighborhood_vecs_x'].fillna(cleaned_df['Neighborhood_vecs_y'])
cleaned_df['Boltz_Distrib'] = cleaned_df['Boltz_Distrib_x'].fillna(cleaned_df['Boltz_Distrib_y'])
cleaned_df = cleaned_df.loc[:,['Sequence','Neighborhood_vecs','Boltz_Distrib', 'Count_x']]
cleaned_df = cleaned_df.rename(columns={'Count_x': 'Count'})

# print len cleaned_df
cleaned_df = pd.merge(cleaned_df, df_16th, on='Sequence', how='outer')
cleaned_df['Neighborhood_vecs'] = cleaned_df['Neighborhood_vecs_x'].fillna(cleaned_df['Neighborhood_vecs_y'])
cleaned_df['Boltz_Distrib'] = cleaned_df['Boltz_Distrib_x'].fillna(cleaned_df['Boltz_Distrib_y'])
cleaned_df['Count_x'] = cleaned_df['Count_x'].fillna(cleaned_df['Count_y'])
cleaned_df = cleaned_df.loc[:,['Sequence','Neighborhood_vecs','Boltz_Distrib', 'Count_x']]
cleaned_df = cleaned_df.rename(columns={'Count_x': 'Count'})

cleaned_df = pd.merge(cleaned_df, df_12th, on='Sequence', how='left')
cleaned_df['Neighborhood_vecs'] = cleaned_df['Neighborhood_vecs_x'].fillna(cleaned_df['Neighborhood_vecs_y'])
cleaned_df['Boltz_Distrib'] = cleaned_df['Boltz_Distrib_x'].fillna(cleaned_df['Boltz_Distrib_y'])
cleaned_df = cleaned_df.loc[:,['Sequence','Neighborhood_vecs','Boltz_Distrib', 'Count_x']]
cleaned_df = cleaned_df.rename(columns={'Count_x': 'Count'})

# we need to remove sequences that show up as substrings of other sequences
# reset the index
cleaned_df = cleaned_df.reset_index(drop=True)
# remove sequences that show up as substrings of other sequences
sequences = cleaned_df['Sequence'].tolist()
indices_to_drop = []
for i in range(len(sequences)):
  for j in range(i + 1, len(sequences)):
    if sequences[i] in sequences[j]:
      indices_to_drop.append(i)
      indices_to_drop.append(j)
      break # No need to check against other sequences if it's a substring of one
    elif sequences[j] in sequences[i]:
      indices_to_drop.append(j)
      indices_to_drop.append(i)
      break # No need to check against other sequences if it's a substring of one

cleaned_df = cleaned_df.drop(indices_to_drop).reset_index(drop=True)

cleaned_df['CPM'] = np.log2((cleaned_df['Count']+ 1) / sum(cleaned_df['Count']) * 1e6)

cleaned_df = cleaned_df[cleaned_df['Sequence'].str.len() == 46]
cleaned_df = cleaned_df.reset_index(drop=True)

print(cleaned_df.isna().sum())
process_nbhd_vecs2(cleaned_df)

exp_nbhd_cleaned = create_exp_nbhd_mat(cleaned_df)
print(cleaned_df.columns)

# I need a Neighborhood_vecs that combines everything for all equences
nbhd_df = pd.merge(df_9th, df_12th, on='Sequence', how='outer')
nbhd_df['Neighborhood_vecs'] = nbhd_df['Neighborhood_vecs_x'].fillna(nbhd_df['Neighborhood_vecs_y'])
nbhd_df['Boltz_Distrib'] = nbhd_df['Boltz_Distrib_x'].fillna(nbhd_df['Boltz_Distrib_y'])
nbhd_df = nbhd_df.loc[:,['Sequence','Neighborhood_vecs','Boltz_Distrib']]

nbhd_df = pd.merge(nbhd_df, df_13th, on='Sequence', how='outer')
nbhd_df['Neighborhood_vecs'] = nbhd_df['Neighborhood_vecs_x'].fillna(nbhd_df['Neighborhood_vecs_y'])
nbhd_df['Boltz_Distrib'] = nbhd_df['Boltz_Distrib_x'].fillna(nbhd_df['Boltz_Distrib_y'])
nbhd_df = nbhd_df.loc[:,['Sequence','Neighborhood_vecs','Boltz_Distrib']]

nbhd_df = pd.merge(nbhd_df, df_16th, on='Sequence', how='outer')
nbhd_df['Neighborhood_vecs'] = nbhd_df['Neighborhood_vecs_x'].fillna(nbhd_df['Neighborhood_vecs_y'])
nbhd_df['Boltz_Distrib'] = nbhd_df['Boltz_Distrib_x'].fillna(nbhd_df['Boltz_Distrib_y'])
nbhd_df = nbhd_df.loc[:,['Sequence','Neighborhood_vecs','Boltz_Distrib']]

print(nbhd_df.isna().sum())
# ok now process these into nbhd vecs 2
process_nbhd_vecs2(nbhd_df)


merged_9_13 = pd.merge(df_9th, df_13th, on='Sequence', how='right')
merged_9_13['Boltz_Distrib_x'] = merged_9_13['Boltz_Distrib_x'].fillna(merged_9_13['Boltz_Distrib_y'])
merged_9_13['Neighborhood_vecs_x'] = merged_9_13['Neighborhood_vecs_x'].fillna(merged_9_13['Neighborhood_vecs_y'])
merged_9_13['d_b_x'] = merged_9_13['d_b_x'].fillna(merged_9_13['d_b_y'])
merged_9_13['faces_x'] = merged_9_13['faces_x'].fillna(merged_9_13['faces_y'])
merged_9_13['energy_faces_x'] = merged_9_13['energy_faces_x'].fillna(merged_9_13['energy_faces_y'])
merged_9_13['Count_y'] = merged_9_13['Count_y'].fillna(0)
merged_9_13['Count_x'] = merged_9_13['Count_x'].fillna(0)
merged_9_13['Enrichment'] = np.log2( (merged_9_13['Count_y']+1) / (merged_9_13['Count_x']+1))
print(merged_9_13['Enrichment'].isna().sum())
merged_9_13['CPM'] = np.log2( (merged_9_13['Count_y']+ 1) / sum(merged_9_13['Count_y']) * 1e6 )
merged_9_13['Adversary'] = 0
merged_9_13.loc[(merged_9_13['CPM'] > 5) & (merged_9_13['Enrichment'] < 0), 'Adversary'] = 1
print(merged_9_13['Adversary'].value_counts())
merged_9_13 = merged_9_13.drop(columns=['Boltz_Distrib_y', 'Neighborhood_vecs_y'])
merged_9_13 = merged_9_13.rename(columns={'Boltz_Distrib_x': 'Boltz_Distrib', 'Neighborhood_vecs_x': 'Neighborhood_vecs', 'faces_x': 'faces', 'energy_faces_x': 'energy_faces'})

# process_nbhd_vecs2(merged_9_13)
# get the neighborhood vecs2 by matching sequences with sequences in nbhd_df
merged_9_13['Neighborhood_vecs2'] = merged_9_13['Sequence'].map(nbhd_df.set_index('Sequence')['Neighborhood_vecs2'])


merged_12_16 = pd.merge(df_12th, df_16th, on='Sequence', how='right')
merged_12_16['Boltz_Distrib_x'] = merged_12_16['Boltz_Distrib_x'].fillna(merged_12_16['Boltz_Distrib_y'])
merged_12_16['Neighborhood_vecs_x'] = merged_12_16['Neighborhood_vecs_x'].fillna(merged_12_16['Neighborhood_vecs_y'])
merged_12_16['d_b_x'] = merged_12_16['d_b_x'].fillna(merged_12_16['d_b_y'])
merged_12_16['faces_x'] = merged_12_16['faces_x'].fillna(merged_12_16['faces_y'])
merged_12_16['energy_faces_x'] = merged_12_16['energy_faces_x'].fillna(merged_12_16['energy_faces_y'])
merged_12_16['Count_y'] = merged_12_16['Count_y'].fillna(0)
merged_12_16['Count_x'] = merged_12_16['Count_x'].fillna(0)
# fill Count_y na with 0s
merged_12_16['Enrichment'] = np.log2( (merged_12_16['Count_y']+1) / (merged_12_16['Count_x']+1))
# tell me if nans in enrichment
print(merged_12_16['Enrichment'].isna().sum())
merged_12_16['CPM'] = np.log2( (merged_12_16['Count_y'] + 1) / sum(merged_12_16['Count_y']) * 1e6)
# make adversaries col all zeros
merged_12_16['Adversary'] = 0
# fill when CPM < 0 and Enrichment > 5
merged_12_16.loc[(merged_12_16['CPM'] > 5) & (merged_12_16['Enrichment'] < 0), 'Adversary'] = 1
# print the count of adversaries
print(merged_12_16['Adversary'].value_counts())
merged_12_16 = merged_12_16.drop(columns=['Boltz_Distrib_y', 'Neighborhood_vecs_y'])
merged_12_16 = merged_12_16.rename(columns={'Boltz_Distrib_x': 'Boltz_Distrib', 'Neighborhood_vecs_x': 'Neighborhood_vecs', 'faces_x': 'faces', 'energy_faces_x': 'energy_faces'})
merged_12_16['Neighborhood_vecs2'] = merged_12_16['Sequence'].map(nbhd_df.set_index('Sequence')['Neighborhood_vecs2'])

# print merged_9_13 shape
print(merged_9_13.shape)
# drop sequences from merged_9_13 that appear in merged_12_16

exp_nbhd_9_13 = create_exp_nbhd_mat(merged_9_13)
exp_nbhd_12_16 = create_exp_nbhd_mat(merged_12_16)

index = ~merged_9_13['Sequence'].isin(merged_12_16['Sequence'])
index12 = ~merged_12_16['Sequence'].isin(merged_9_13['Sequence'])

# count the adversaries in index
# print the count of adversaries
print(merged_9_13[~index]['Adversary'].value_counts())
print(merged_12_16[~index12]['Adversary'].value_counts())

# I want the index of the enrichment scores that differ
# need to sort by sequences so they align
# subindex_enrichment = np.where(merged_9_13[~index]['Enrichment'] != merged_12_16[~index12]['Enrichment'])
sorted_9_13_overlap = merged_9_13[~index].sort_values('Sequence').reset_index(drop=False)
sorted_12_16_overlap = merged_12_16[~index12].sort_values('Sequence').reset_index(drop=False)
average_enrichment = (sorted_9_13_overlap['Enrichment'] + sorted_12_16_overlap['Enrichment']) / 2
# now put these
sorted_9_13_overlap['Enrichment'] = average_enrichment
sorted_12_16_overlap['Enrichment'] = average_enrichment
# put these back based on the original index
# reset the index to 'index'
sorted_9_13_overlap = sorted_9_13_overlap.set_index('index')
sorted_12_16_overlap = sorted_12_16_overlap.set_index('index')
# print(merged_9_13[~index]['Enrichment'].describe())
merged_9_13.loc[~index,'Enrichment'] = sorted_9_13_overlap['Enrichment']
# print(merged_9_13[~index]['Enrichment'].describe())
merged_12_16.loc[~index12,'Enrichment'] = sorted_12_16_overlap['Enrichment']

# we actually want to drop ~index12 from merged_12_16
merged_12_16 = merged_12_16[index12]
print(merged_12_16['Adversary'].value_counts())
exp_nbhd_12_16 = exp_nbhd_12_16[index12]

# drop row with se
# print(merged_12_16[~index12]['Enrichment'].describe())

# faces = BoF_in_df(merged_9_13)
# print(faces.shape)


# # merged_9_13 = merged_9_13[index]
# # exp_nbhd_9_13 = exp_nbhd_9_13[index]

# # print merged_9_13 shape
# print(merged_9_13.shape)
# # print merged_12_16 shape
# print(merged_12_16.shape)

# print(exp_nbhd_9_13.shape)
# print(exp_nbhd_12_16.shape)


from nac import Kmer
kmer = Kmer(k=5, upto=True, normalize=True)
# kmer_cleaned = kmer.transform(cleaned_df['Sequence'])
kmer_cleaned = kmer.make_kmer_vec(list(cleaned_df['Sequence'].values))
kmer_cleaned = np.array(kmer_cleaned)
print(kmer_cleaned.shape)



# FEATURE CHOICE
# ==============
FEAT_REGISTRY = {
  'exp_nbhd': exp_nbhd_cleaned,
  'kmer': kmer_cleaned,
  'exp_nbhd_kmer': np.concatenate((exp_nbhd_cleaned, kmer_cleaned), axis=1)
}

# DECOMP CHOICE
# =============
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF
DECOMP_REGISTRY = {
  'lda': LDA(n_components=10, random_state=42),
  'nmf': NMF(n_components=10, random_state=42)
}

# MODEL CHOICE
# ============
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
MODEL_REGISTRY = {
'xgb': xgb.XGBClassifier(
    objective='reg:squarederror',
      reg_alpha=0.0,      # L1 on leaf weights
    reg_lambda=0.0,      # turn off L2 if you like
    colsample_bytree=0.3,
    learning_rate=0.1,
    max_depth=8,
    alpha=10,
    n_estimators=100),
'log_reg' : LogisticRegressionCV(
      penalty='l1', solver='saga', class_weight='balanced',
    cv=5, scoring='roc_auc', max_iter=5000, random_state=0),
 'rf' : RandomForestClassifier(n_estimators=100, random_state=42)
}


# VIS CHOICE
# ==========
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
VIS_REGISTRY = {
  'tsne': TSNE(n_components=2, random_state=42),
  'pca': PCA(n_components=2)
}

# CLUST CHOICE
# ============
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import hdbscan
CLUSTER_REGISTRY = {
  'kmeans': KMeans(n_clusters=5, random_state=42),
  'spectral': SpectralClustering(n_clusters=5, random_state=42),
  'hdbscan': hdbscan.HDBSCAN(min_cluster_size=10)
}


def run_pipeline(feat, decomp, model, vis, clusterer):
  features = FEAT_REGISTRY[feat.lower()]
  decomp = DECOMP_REGISTRY[decomp.lower()]
  model = MODEL_REGISTRY[model.lower()]
  vis = VIS_REGISTRY[vis.lower()]
  clusterer = CLUSTER_REGISTRY[clusterer.lower()]
  y = cleaned_df['Adversary']
  X = decomp.fit_transform(features)
  model.fit(X, y)

  print(model.score(X, y))
  try:
    feature_coeffs = model.feature_importances_
  except:
    feature_coeffs = model.coef_[0]

  top_10p_contribute = np.quantile(feature_coeffs, 0.8)
  positive_traits = feature_coeffs > top_10p_contribute

  X_embedded = vis.fit_transform(X[:, positive_traits])
  clusterer.fit(X[:, positive_traits])
  cluster_labels = clusterer.labels_


  plt.figure(figsize=(10,10))
  vmin = np.min(cluster_labels)
  vmax = np.max(cluster_labels)
  plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=cluster_labels, cmap='Dark2_r', alpha=0.9, vmin=-2, vmax=vmax, s=500)
  plt.colorbar()

  plt.scatter(X_embedded[cleaned_df['Adversary']==1, 0], X_embedded[cleaned_df['Adversary']==1, 1], c='red', marker='d', s=200)

  top_10p = np.quantile(cleaned_df['Enrichment'], 0.8)
  print(top_10p)
  top_10_enrch = cleaned_df['Enrichment'] > top_10p

  print(cleaned_df[top_10_enrch]['Adversary'].value_counts())
  plt.scatter(X_embedded[top_10_enrch, 0], X_embedded[top_10_enrch, 1], c='green', marker='d', s=200)

  top_10p = np.quantile(cleaned_df['CPM'], 0.9)
  top_10_cnt = cleaned_df['CPM'] > top_10p
  plt.scatter(X_embedded[top_10_cnt, 0], X_embedded[top_10_cnt, 1], c='cyan', marker='+', s=200)
  # top_5_cnt = np.argsort(cleaned_df['CPM'])[-5:]
  # plt.scatter(X_embedded[top_5_cnt, 0], X_embedded[top_5_cnt, 1], c='black', marker='o', s=200)

  plt.xlabel('t-SNE 1')
  plt.ylabel('t-SNE 2')

  # score each cluster based on count of adversaries vs total count
  cluster_adversary_counts = np.zeros(len(np.unique(cluster_labels)))
  for i in range(len(np.unique(cluster_labels))):
    cluster_adversary_counts[i] = cleaned_df[cluster_labels == i-1]['Adversary'].sum()
  print(cluster_adversary_counts)
  avg_adversary_count = np.zeros(len(np.unique(cluster_labels)))
  for i in range(len(np.unique(cluster_labels))):
    avg_adversary_count[i] = cluster_adversary_counts[i] / (len(cleaned_df[cluster_labels == i-1]) + 1)
  print(avg_adversary_count)
  avg_top_count = np.zeros(len(np.unique(cluster_labels)))
  for i in range(len(np.unique(cluster_labels))):
    avg_top_count[i] = cleaned_df[cluster_labels == i-1]['CPM'].mean()
  print(avg_top_count)
  avg_enrichment = np.zeros(len(np.unique(cluster_labels)))
  for i in range(len(np.unique(cluster_labels))):
    avg_enrichment[i] = cleaned_df[cluster_labels == i-1]['Enrichment'].mean()
  print(avg_enrichment)

  return cluster_labels


# score each cluster based on count of adversaries vs total count
cluster_adversary_counts = np.zeros(len(np.unique(cluster_labels)))
for i in range(len(np.unique(cluster_labels))):
  cluster_adversary_counts[i] = cleaned_df[cluster_labels == i-1]['Adversary'].sum()
print(cluster_adversary_counts)
avg_adversary_count = np.zeros(len(np.unique(cluster_labels)))
for i in range(len(np.unique(cluster_labels))):
  avg_adversary_count[i] = cluster_adversary_counts[i] / (len(cleaned_df[cluster_labels == i-1]) + 1)
print(avg_adversary_count)
avg_top_count = np.zeros(len(np.unique(cluster_labels)))
for i in range(len(np.unique(cluster_labels))):
  avg_top_count[i] = cleaned_df[cluster_labels == i-1]['CPM'].mean()
print(avg_top_count)
avg_enrichment = np.zeros(len(np.unique(cluster_labels)))
for i in range(len(np.unique(cluster_labels))):
  avg_enrichment[i] = cleaned_df[cluster_labels == i-1]['Enrichment'].mean()
print(avg_enrichment)


INDEX = 0
PCT = 0.8
top_10p = np.quantile(cleaned_df[cluster_labels == INDEX]['CPM'], PCT)
print(cleaned_df[cluster_labels == INDEX][cleaned_df['CPM'] > top_10p][['Sequence', 'CPM', 'Enrichment']])
top_10p_enrch = np.quantile(cleaned_df['Enrichment'], PCT)
print(cleaned_df[cluster_labels == INDEX][cleaned_df['Enrichment']>top_10p_enrch][['Sequence', 'CPM', 'Enrichment']])
