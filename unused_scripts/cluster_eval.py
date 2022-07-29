import os
import re
import string
import warnings
from os.path import exists

import matplotlib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from clusteval import clusteval
from matplotlib import cm
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# for clustering
from scipy.cluster.hierarchy import ward, dendrogram, fcluster
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
warnings.filterwarnings("ignore")
stop_words = set(stopwords.words('english'))

# set DEBUG in your environment variables to enable debug mode
DEBUG = os.getenv("DEBUG", 'False').lower() in ('true', '1', 't')
PKL_CACHE_FN = "cache/people_abstracts_df.pkl"


def remove_stop_words_and_do_stemming(unfiltered_text):
    unfiltered_text = unfiltered_text.translate(str.maketrans("", "", string.punctuation))
    word_tokens = word_tokenize(unfiltered_text.lower())

    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    stem_words = []
    ps = PorterStemmer()
    for w in filtered_sentence:
        root_word = ps.stem(w)
        stem_words.append(root_word)

    return ' '.join(stem_words)



def get_top_keywords(clusters_df, cluster_index, labels, n_terms):
    r = clusters_df.iloc[cluster_index]
    return ','.join([labels[t] for t in np.argsort(r)[-n_terms:]])


def build_corpus(use_cache):
    if use_cache and exists(PKL_CACHE_FN):
        return pd.read_pickle(PKL_CACHE_FN)

    people = pd.read_json('../data/people.json')
    papers = pd.read_json('../data/papers.json')


    people_list = []
    text_list = []

    for personIdx in people.index:
        all_person_text = ""
        title = ""
        for paperIdx in people['publications'][personIdx]:
            row = papers.loc[papers['uri'] == paperIdx]

            if not row.empty and len(paperIdx) > 32:
                title = row['title'].values[0]
                abstract = row['paperAbstract']
                if len(abstract) == 1:
                    abstract = abstract.values.item()
                if isinstance(abstract, str):
                    all_person_text += " " + abstract
            # else:
            #   print("Missing paper: " + paperIdx)
        all_person_text = title + all_person_text
        person = people['name'][personIdx]
        if person in people_list:
            print("Duplicate name found, this shouldn't happen")
            exit()
        people_list.append(person)
        list_of_words = remove_stop_words_and_do_stemming(all_person_text)
        text_list.append(''.join(list_of_words))

    df = pd.DataFrame({
        'people': people_list,
        'text': text_list
    })
    df.to_pickle(PKL_CACHE_FN)
    return df




def main():
    abstracts_df = build_corpus(use_cache=True)

    if DEBUG:
        abstracts_df = abstracts_df.head(400)
    #configs = [("agglomerative", "derivative"), ("agglomerative", "silhouette"), ("agglomerative", "dbindex"), ("dbscan"), ("hdbscan")]
    configs = [("kmeans", "silhouette")] #dbindex doesn't work, derivative is not valid
    #configs = [("hdbscan")]

    max_clusters=10
    f = open('output/abstracts/cluster_info.txt', 'w')
    for config in configs:
        for ngram_size in range(2, 4):
            ngram_text_size = str(ngram_size)
            text = abstracts_df.text.tolist()
            vectorizer = CountVectorizer(analyzer='word', ngram_range=(ngram_size, ngram_size))
            vectors = vectorizer.fit(text)
            print(vectorizer.vocabulary_)
            vectors = vectorizer.transform(text)
            print(vectors.shape)
            print(vectors.toarray())
            print(vectors.nonzero())
            if len(config) == 2:
                ce = clusteval(cluster=config[0], evaluate=config[1], max_clust=max_clusters)
            else:
                ce = clusteval(cluster=config, max_clust=max_clusters)# , params_dbscan={'eps':None, 'epsres':50, 'min_samples':int(1), 'norm':False, 'n_jobs':-1})


            # Fit data X
            results = ce.fit(vectors.toarray())

            # The clustering label can be found in:
            print(results['labx'])

            # Plot
            ce.plot(savefig={'format':'png', 'fname':'output/plot' + '-'.join(config) + '-ngrm-sz' + "-" + ngram_text_size + '.png'})
            ce.scatter(vectors.todense(), savefig={'format':'png','fname':'output/scatter-' + '-'.join(config) + '-ngrm-sz' + "-" + ngram_text_size +
                                                 '.png'})
           # ce.dendrogram( max_d=1000.0, savefig={'format':'png', 'fname':'output/dendrogram' + '-'.join(config) + '-ngrm-sz' + "-" + ngram_text_size +
             #                                    '.png'})

    quit()


if __name__ == "__main__":
    main()
