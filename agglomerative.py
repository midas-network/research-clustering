import os
import re
import string
import warnings
from os.path import exists

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from matplotlib import cm
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# for clustering
from scipy.cluster.hierarchy import ward, dendrogram, fcluster
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
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


def plot_tsne_pca(cluster_num, ngram_size, data, labels):
    max_label = max(labels)
    max_items = np.arange(0, data.shape[0], 1, dtype=int)

    pca = PCA(n_components=2).fit_transform(data[max_items, :].todense())
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items, :].todense()))

    label_subset = labels[max_items]
    label_subset = [cm.hsv(i / max_label) for i in label_subset[max_items]]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(pca[max_items, 0], pca[max_items, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot: {} Clusters; {}-word n-grams'.format(cluster_num, ngram_size))

    ax[1].scatter(tsne[max_items, 0], tsne[max_items, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot: {} Clusters; {}-word n-grams'.format(cluster_num, ngram_size))

    plt.savefig('output/clstr-sz-{}-ngrm-sz{}.png'.format(cluster_num, ngram_size))


def get_top_keywords(clusters_df, cluster_index, labels, n_terms):
    r = clusters_df.iloc[cluster_index]
    return ','.join([labels[t] for t in np.argsort(r)[-n_terms:]])


def print_clusters_report(f, num_of_clusters, ngram_size, main_df, clusters_df, cluster_array, tfidf_obj):
    f.write('\n{} Clusters, {} 2-word n-grams'.format(num_of_clusters, ngram_size))
    people_cluster_dict = {key: [] for key in range(num_of_clusters)}
    for person_index in main_df['people'].index:
        person_name = re.sub(r' \d{1,3}$', '', main_df['people'][person_index])
        person_cluster = cluster_array[person_index]
        people_cluster_dict[person_cluster].append(person_name)

    for cluster_index in people_cluster_dict:
        f.write('\n\tCluster ' + str(cluster_index + 1))
        f.write('\n\t\tResearchers (' + str(len(people_cluster_dict[cluster_index])) + '): ' + ','
                .join(people_cluster_dict[cluster_index]))
        f.write(
            '\n\t\tTop 20 n-grams: ' + get_top_keywords(clusters_df, cluster_index, tfidf_obj.get_feature_names(), 20))


def build_corpus(use_cache):
    if use_cache and exists(PKL_CACHE_FN):
        return pd.read_pickle(PKL_CACHE_FN)

    people = pd.read_json('data/people.json')
    papers = pd.read_json('data/papers.json')

    if DEBUG:
        papers = papers.head(1000)

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


def silhouette_k(distance_matrix, linkage_matrix, title, max_k=40):
    scores = []
    for i in range(2, max_k + 1):
        clusters = fcluster(linkage_matrix, i, criterion='maxclust')
        score = silhouette_score(distance_matrix, clusters, metric='precomputed')
        #print("Silhouette score with {} clusters:".format(i), score)
        scores.append(score)
    plt.clf()
    plt.figure(figsize=(15, 15), dpi=100)
    plt.title("Silhouette score vs. number of clusters")
    plt.xlabel("# of clusters")
    plt.ylabel("Score (higher is better)")
    plt.plot(np.arange(2, max_k + 1), scores)
    plt.savefig("output/"+title)
    return scores


def main():
    abstracts_df = build_corpus(use_cache=True)
    f = open('output/cluster_info.txt', 'w')
    for ngram_size in range(1, 7):
        plt.clf()
        ngram_text_size = str(ngram_size)
        if ngram_size == 6:
            ngram_size1 = 1
            ngram_size2 = 5
            ngram_text_size = "-rng-" + str(ngram_size1) + "-" + str(ngram_size2)
        else:
            ngram_size1 = ngram_size
            ngram_size2 = ngram_size
        tfidf = TfidfVectorizer(
            min_df=5,
            max_df=0.95,
            max_features=8000,
            ngram_range=(ngram_size1, ngram_size2),
            analyzer='word',
        )
        tfidf.fit(abstracts_df.text)
        text = tfidf.transform(abstracts_df.text)

        dist = 1 - cosine_similarity(text)
        dist = dist - dist.min()  # get rid of some pesky floating point errors that give neg. distance
        linkage_matrix = ward(dist)  # replace with complete, single,

        data = abstracts_df
        max_title_len = 70
        max_cophenetic_dist = max(
            linkage_matrix[:, 2]) * 0.39  # max distance between points to be considered together. can be tuned.

        plt.subplots(figsize=(15, 80))  # set size
        dendrogram(linkage_matrix,
                   orientation="right",
                   color_threshold=max_cophenetic_dist,
                   leaf_font_size=4,
                   labels=data.people.apply(
                       lambda x: x if len(x) < max_title_len else x[:max_title_len - 3] + "...").tolist()
                   )

        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.tight_layout()  # show plot with tight layout
        plt.savefig('output/agg_clstrs-ngrm-sz' + ngram_text_size + ".png", dpi=300)
        np.fill_diagonal(dist, 0)

        _ = silhouette_k(dist, linkage_matrix, title="silhouette-ngrm-sz" + ngram_text_size)

    quit()


if __name__ == "__main__":
    main()
