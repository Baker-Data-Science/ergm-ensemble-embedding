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
df_raw = df_raw[~df_raw['Sequence'].isin(df_all['Sequence'])]
print(f"Number of sequences in df_raw not in df_all: {len(df_raw)}")

print(df_all.columns)

#x = df_all["CPM"].values
XMIN = df_all["CPM"].min()
XMAX = df_all["CPM"].max()
YMIN = df_all["Enrichment"].min()
YMAX = df_all["Enrichment"].max()

# MAKE HISTOGRAM
x = df_all["CPM"].values
y = df_all["Enrichment"].values

nxbins = 50
nybins = 50
xbins = np.linspace(x.min(), x.max(), nxbins)
ybins = np.linspace(y.min(), y.max(), nybins)

H, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])
H = H.T  # match imshow's orientation


top_enrichment = np.percentile(df_all['Enrichment'], 95)
bottom_enrichment = np.percentile(df_all['Enrichment'], 50)
print(f"Top enrichment threshold: {top_enrichment}")
print(f"Bottom enrichment threshold: {bottom_enrichment}")
top_enrichment_indices = df_all[df_all['Enrichment'] >= top_enrichment].index.values
bottom_enrichment_indices = df_all[df_all['Enrichment'] < bottom_enrichment].index.values

top_cpm = np.percentile(df_all['CPM'], 90)
bottom_cpm = np.percentile(df_all['CPM'], 95)
print(f"Top CPM threshold: {top_cpm}")
print(f"Bottom CPM threshold: {bottom_cpm}")
top_cpm_indices = df_all[df_all['CPM'] >= top_cpm].index.values

bottom_cpm_indices = df_all[df_all['CPM'] < bottom_cpm].index.values


# intersect top cpm and bottom enrichment
intersect_good = list(np.intersect1d(bottom_cpm_indices, top_enrichment_indices))
intersect_bad = list(np.intersect1d(top_cpm_indices, bottom_enrichment_indices))

null_check = list(np.intersect1d(bottom_enrichment_indices, top_enrichment_indices))
print(f"Number of aptamers failed check {len(null_check)}")


print(f"Number of top aptamers {len(intersect_good)}")
print(f"Number of bottom aptamers {len(intersect_bad)}")

# PLOTTING
# ========
colors = [0.5] * len(df_all)  # initialize colors with a neutral value
for bad in intersect_bad:
    colors[bad] = 0.0  # set top enrichment to 1.0
for good in intersect_good:
    colors[good] = 1.0  # set bottom enrichment to 0.0



print(df_all.shape)

# Axis labels
xlabel = "Count Per Million"
ylabel = "Enrichment"

# Figure geometry
left, width = 0.12, 0.55
bottom, height = 0.12, 0.55
bottom_h = left_h = left + width + 0.02

rect_joint = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.25]
rect_histy = [left_h, bottom, 0.25, height]

fig = plt.figure(figsize=(8, 8))

axJoint = plt.axes(rect_joint)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# Remove inner tick labels for histograms
nullfmt = NullFormatter()
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)
# replace H with color values


# plot the line
VIS_MIN = 0.75
slope = (YMAX - VIS_MIN) / (XMAX - XMIN)
intercept = VIS_MIN - slope * XMIN
#axJoint.plot([XMIN, XMAX], [slope * XMIN + intercept, slope * XMAX + intercept], color='black', linestyle='--', linewidth=3)
#axJoint.plot([XMIN, XMAX], [slope * XMIN - 7, slope * XMAX -7], color='black', linestyle='--', linewidth=3)

# plot the points
# put a red circle under the top % points
ring = 120
mid = .5
axJoint.scatter(x[intersect_good], y[intersect_good], c='blue', s=ring, alpha=0.5, edgecolor='none', label='Top 5% Enrichment')
# put a red circle under the bottom % points
axJoint.scatter(x[intersect_bad], y[intersect_bad], c='red', s=ring, alpha=0.5, edgecolor='none', label='Bottom 5% Enrichment')
axJoint.scatter(x, y, c=colors, s=mid*ring, alpha=1.0, edgecolor='none')
# scatter df_raw not in df_all
axJoint.scatter(df_raw["CPM"].values, df_raw["Enrichment"].values,
                c='black', alpha=1.0, edgecolor='none', label='Raw Data Not in Merged')



axJoint.set_xlabel(xlabel)
axJoint.set_ylabel(ylabel)

#axJoint.set_xlim(XMIN-.05, XMAX+.05)
#axJoint.set_ylim(YMIN-.05, YMAX+.05)

# Histogram axis limits
axHistx.hist(x, bins=xbins, color='#6c8ebf', edgecolor='black', alpha=0.7)
axHistx.set_xlim(axJoint.get_xlim())
axHistx.set_yscale('log')   # log scale for counts of x
axHistx.yaxis.set_major_locator(MaxNLocator(4))
axHistx.set_yticks([1e1, 1e2], ['10', '100'])  # set y-ticks for log scale



axHisty.hist(y, bins=ybins, color='#6c8ebf', edgecolor='black', alpha=0.7, orientation='horizontal')
axHisty.set_ylim(axJoint.get_ylim())
axHisty.set_xscale('log')   # log scale for counts of y
axHisty.xaxis.set_major_locator(MaxNLocator(4))
axHisty.set_xticks([1e1, 1e2], ['10', '100'], rotation=45)  # set x-ticks for log scale


# put green dots on the aptamers below the diagonal and red dots on the aptamers above the shifted linestyle
# Plotting the aptamers below and above the linestyle
#axJoint.scatter(df_all[df_all['Enrichment'] > slope*df_all['CPM'] + intercept]['CPM'], 
                 #df_all[df_all['Enrichment'] > slope*df_all['CPM'] + intercept]['Enrichment'], 
                 #color='green', s=80, alpha=0.5, edgecolor='none', label='Below Diagonal')

#axJoint.scatter(df_all[df_all['Enrichment'] < slope*df_all['CPM'] - 7]['CPM'], 
                 #df_all[df_all['Enrichment'] < slope*df_all['CPM'] - 7]['Enrichment'], 
                 #color='red', s=80, alpha=0.5, edgecolor='none', label='Above Shifted Line')


plt.savefig('./data/figure_1.png', bbox_inches='tight')
