import os

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
from prince import PCA
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import umap.umap_ as umap
import matplotlib.pyplot as plt

import warnings


def dim_red(mat, p, method):
    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM list
        p : number of dimensions to keep
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''

    if method == 'ACP':

        df = pd.DataFrame(embeddings)
        # initialiser un model de ACP avec k composantes principales
        acp_model = PCA(n_components=p)

        # application de ACP sur notre dataset
        acp_model.fit(df)
        # renvoyer la dataset reduit
        df_acp = acp_model.transform(df).values

        red_mat = df_acp[:, :p]

    elif method == 'TSNE':
        # TNSE does'nt allow more than 3 components
        p = 3
        r_mat = TSNE(n_components=p,
                     learning_rate='auto',
                     init='random',
                     perplexity=3).fit_transform(mat)
        red_mat = r_mat[:, :p]

    elif method == 'UMAP':

        umap_model = umap.UMAP(n_components=p)

        red_mat = umap_model.fit_transform(mat)

        return red_mat

    else:
        raise Exception("Please select one of the three methods : APC, AFC, UMAP")

    return red_mat


def clust(mat, k):
    '''
    Perform clustering

    Input:
    -----
        mat : input list
        k : number of cluster
    Output:
    ------
        pred : list of predicted labels
    '''

    kmeans = KMeans(init='k-means++',
                    n_init=20,
                    n_clusters=k)
    kmeans.fit(mat)
    pred = kmeans.labels_

    return pred


"""
def save_as_csv(x, y):
    os.makedirs(name='data', exist_ok=True)
    df_x = pd.DataFrame(x)
    df_y = pd.DataFrame(y, columns=['y'])

    df_new = pd.concat([df_x, df_y], axis=1)
    df_new.to_csv(path_or_buf='./data/embedding.csv')

"""

# import data
# ng20 = fetch_20newsgroups(subset='test')
# corpus = ng20.data[:2000]
# labels = ng20.target[:2000]


data = pd.read_csv(filepath_or_buffer='./data/embedding.csv')
embeddings = data.iloc[:, :-1]
labels = data.iloc[:, -1]

k = len(set(labels))

# embedding
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# embeddings = model.encode(corpus)

# save_as_csv(embeddings, labels)

num_iterations = 2
nmi_scores = []
ari_scores = []

# Perform dimensionality reduction and clustering for each method
methods = ['ACP', 'TSNE', 'UMAP']
for method in methods:

    red_emb = dim_red(embeddings, 20, method)
    for _ in range(num_iterations):
        # perform clustering
        pred = clust(red_emb, k)

        # evaluate clustering results
        nmi_score = normalized_mutual_info_score(pred, labels)
        ari_score = adjusted_rand_score(pred, labels)

        nmi_scores.append(nmi_score)
        ari_scores.append(ari_score)

        if _ == num_iterations - 1:
            fig, axs = plt.subplots(1, 2, figsize=(16, 4))
            sns.scatterplot(x=red_emb[:, 0], y=red_emb[:, 1], hue=labels, palette='viridis', legend='full', ax=axs[0])
            axs[0].set_title(f'{method} visualisation with truee labels - Iteration {_ + 1}')

            sns.scatterplot(x=red_emb[:, 0], y=red_emb[:, 1], hue=pred, palette='viridis', legend='full', ax=axs[1])
            axs[1].set_title(f'{method} Clustering - Iteration {_ + 1}')
            plt.savefig(f'./fig/{method} + clustering.png')
    # Calculate average scores
    average_nmi = sum(nmi_scores) / num_iterations
    average_ari = sum(ari_scores) / num_iterations

    print(f'Average NMI on {num_iterations} iterations for {method}: {average_nmi:.2f}'
          f'\nAverage ARI on {num_iterations} iterations for {method}: {average_ari:.2f} \n')
