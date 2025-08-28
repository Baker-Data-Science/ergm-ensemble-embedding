import numpy as np
import pandas as pd

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
df_9th = pd.read_csv("./data/df_9th.csv")
df_12th = pd.read_csv("./data/df_12th.csv")
df_13th = pd.read_csv("./data/df_13th.csv")
df_16th = pd.read_csv("./data/df_16th.csv")

lib1_merged, lib2_merged, df_all, lib1_overlap, lib2_overlap = get_libs(df_9th, df_12th, df_13th, df_16th, clean=True)
# drop any sequences longer than 46
_, _, df_raw, _, _ = get_libs(df_9th, df_12th, df_13th, df_16th, clean=False)

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

print(f"Total data {(len(df_raw))}")
print(f"Adversaries {len(df_missing)}")
print(f"Good Anomalies {len(indices_good)}")
print(f"Bad Anomalies {len(indices_bad)}")

# PLOTTING
# ========
plt.figure()

# Configs
ring = 100
mid = .5


left, width = 0.12, 0.55
bottom, height = 0.12, 0.55
bottom_h = left_h = left + width + 0.02

rect_joint = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.285]
rect_histy = [left_h, bottom, 0.25, height]

axJoint = plt.axes(rect_joint)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# Remove inner tick labels for histograms
nullfmt = NullFormatter()
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)
# replace H with color values

print(df_all.shape)
# SCATTER PLOT
# =============
x = df_all["Redist_Count"].values
y = df_all["Pressure"].values

x_raw = df_raw["Redist_Count"].values
y_raw = df_raw["Pressure"].values

x_both = df_both["Redist_Count"].values
y_both = df_both["Pressure"].values


df_first = df_both[~(df_both['Sequence'].isin(df_raw['Sequence'][indices_good]))]
df_first = df_first[~(df_first['Sequence'].isin(df_raw['Sequence'][indices_bad]))]
x_first = df_first["Redist_Count"].values
y_first = df_first["Pressure"].values

axJoint.scatter(x_first, y_first, c='black', s=.8*ring, alpha=0.9, label='SELEX Data')


#axJoint.scatter(df_missing["Redist_Count"].values, df_missing["Pressure"].values,
                #c='orange', alpha=0.9, label='Adversaries',
                #s=ring, marker='o')

axJoint.scatter(x_raw[indices_good], y_raw[indices_good], c='green', s=ring, alpha=0.9, label='LC-HP Anomalies', marker='o')
axJoint.scatter(x_raw[indices_bad], y_raw[indices_bad], c='red', s=ring, alpha=0.9, label='HC-LP Anomalies', marker='o')



xlabel = "Normalied Count"
ylabel = "Selective Pressure"
axJoint.set_xlabel(xlabel)
axJoint.set_ylabel(ylabel)
axJoint.set_xlim(x_first.min()-.02,x_first.max()+.05)
axJoint.set_ylim(y_first.min()-.5,y_first.max()+.5)

axJoint.legend(loc='upper right', fontsize=30, markerscale=2.0, bbox_to_anchor=(1.53,1.55), handletextpad=0.2, handlelength=0.6)


# HISTOGRAMS
# ==========
nxbins = 50
#xbins = np.linspace(x_raw.min(), x_raw.max(), nxbins)
#axHistx.hist(x_raw, bins=xbins, color='orange', edgecolor='black', alpha=1.0)

xbins = np.linspace(x_both.min(), x_both.max(), nxbins)
axHistx.hist(x_both, bins=xbins, color='black', edgecolor='black', alpha=0.8)
axHistx.hist(x_raw[indices_good], bins=xbins, color='green', edgecolor='black', alpha=0.8)
axHistx.hist(x_raw[indices_bad], bins=xbins, color='red', edgecolor='black', alpha=0.8)

axHistx.set_xlim(axJoint.get_xlim())
axHistx.set_yscale('log')   # log scale for counts of x
axHistx.yaxis.set_major_locator(MaxNLocator(4))
axHistx.set_yticks([1e1, 1e2, 1e3], ['10', '100', '1000'])  # set x-ticks for log scale


nybins = 50
#ybins = np.linspace(y_raw.min(), y_raw.max(), nybins)
#axHisty.hist(y_raw, bins=ybins, color='orange', edgecolor='black', alpha=1.0, orientation='horizontal')

ybins = np.linspace(y_both.min(), y_both.max(), nxbins)
axHisty.hist(y_both, bins=ybins, color='black', edgecolor='black', alpha=0.8, orientation='horizontal')
#axHisty.hist(y_raw[indices_good], bins=ybins, color='#6c8ebf', edgecolor='black', alpha=0.8, orientation='horizontal')
#axHisty.hist(y_raw[indices_bad], bins=ybins, color='#b85450', edgecolor='black', alpha=0.8, orientation='horizontal')
# stack these on top  of the previous histogram
axHisty.hist([y_raw[indices_good], y_raw[indices_bad]], bins=ybins, color=['green','red'], edgecolor='black', alpha=0.8, orientation='horizontal', stacked=True)

axHisty.set_ylim(axJoint.get_ylim())
axHisty.set_xscale('log')   # log scale for counts of y
axHisty.xaxis.set_major_locator(MaxNLocator(4))
axHisty.set_xticks([1e1, 1e2, 1e3], ['10', '100', '1000'], rotation=45)  # set x-ticks for log scale


plt.savefig('./data/icdm_figure_3.png', format='png', bbox_inches='tight')
