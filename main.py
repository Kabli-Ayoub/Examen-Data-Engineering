from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
from prince import PCA
import numpy as np
import pandas as pd


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
        # initialiser un model de ACP avec k composantes principales
        acp_model = PCA(n_components=k)

        # application de ACP sur notre dataset
        acp_model.fit(df)
        # renvoyer la dataset reduit
        df_acp = acp_model.transform(df)

        red_mat = df_acp[:, :p]

    elif method == 'AFC':
        red_mat = mat[:, :p]

    elif method == 'UMAP':
        red_mat = mat[:, :p]

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

    pred = np.random.randint(k, size=len(corpus))

    return pred


# import data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)
df = pd.DataFrame(embeddings)

# Perform dimensionality reduction and clustering for each method
methods = ['ACP', 'AFC', 'UMAP']
for method in methods:
    # Perform dimensionality reduction
    red_emb = dim_red(embeddings, 20, method)

    # Perform clustering
    pred = clust(red_emb, k)

    # Evaluate clustering results
    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)

    # Print results
    print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')


