from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
f1_name = 'embedding.tsv'
f2_name = 'name.tsv'
f1 = open(f1_name, 'r')
embs = []
for line in f1.readlines():
    emb = line.strip('\n').strip('\t').split('\t')
    emb = [float(xx) for xx in emb]
    embs.append(emb)
f2 = open(f2_name, 'r')
name2type = {}
type2id = {'PER': 0, 'TIME': 1, 'LOC': 2, 'NUM': 3, 'ORG': 4, 'MISC': 5}
id2type = {0: 'PER', 1: 'TIME', 2: 'LOC', 3: 'NUM', 4: 'ORG', 5: 'MISC'}
ids = []
names = []
types = []
flag = 0
for line in f2.readlines():
    if flag == 0:
        flag = 1
        continue
    line = line.strip('\n').split('\t')
    name2type[line[0]] = line[1]
    names.append(line[0])
    types.append(line[1])
    ids.append(type2id[line[1]])
X = np.array(embs)
y = np.array(ids)
feat_cols = [ names[i] for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = types
df['label'] = df['y']
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

N = 10000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
df_subset['pca-three'] = pca_result[:,2]
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(8,5))
plt.xlim(-15, 15)
plt.ylim(-15, 15)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 6),
    data=df_subset,
    legend="full",
    alpha=0.3
)
plt.show()