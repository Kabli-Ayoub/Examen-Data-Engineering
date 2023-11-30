from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap
import matplotlib.pyplot as plt


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
        red_mat = mat[:, :p]

    elif method == 'TSNE':
        # TNSE does'nt allow more than 3 component
        p = 3
        r_mat = TSNE(n_components=p,
                     learning_rate='auto',
                     init='random',
                     perplexity=3).fit_transform(mat)
        red_mat = r_mat[:, :p]

    elif method == 'UMAP':

        umap_model = umap.UMAP(random_state=42, n_components=p)

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
    kmeans = KMeans(n_clusters=k)  
    kmeans.fit(mat)
    pred = kmeans.labels_
    return pred


# import data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

num_iterations=2
nmi_scores = []
ari_scores = []

# Perform dimensionality reduction and clustering for each method
methods = ['ACP', 'AFC', 'UMAP']
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
            plt.show()
# Calculate average scores
average_nmi = sum(nmi_scores) / num_iterations
average_ari = sum(ari_scores) / num_iterations

print(f'Average NMI on {num_iterations}: {average_nmi:.2f}\nAverage ARI on {num_iterations}: {average_ari:.2f}')

