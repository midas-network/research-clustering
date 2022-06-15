import os
import string
import sys
import warnings
import matplotlib
import nltk
import re

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from matplotlib import cm
from nltk import PorterStemmer, ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
warnings.filterwarnings("ignore")
stop_words = set(stopwords.words('english'))

# set DEBUG in your environment variables to enable debug mode
DEBUG = os.getenv("DEBUG", 'False').lower() in ('true', '1', 't')


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
        f.write('\n\t\tTop 20 n-grams: ' + get_top_keywords(clusters_df, cluster_index, tfidf_obj.get_feature_names(), 20))


def build_corpus():
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
            #else:
                #   print("Missing paper: " + paperIdx)
        all_person_text = title + all_person_text
        person = people['name'][personIdx] + " " + str(personIdx)
        people_list.append(person)
        list_of_words = remove_stop_words_and_do_stemming(all_person_text)
        text_list.append(''.join(list_of_words))

    df = pd.DataFrame({
        'people': people_list,
        'text': text_list
    })
    return df


def main():
    abstracts_df = build_corpus()
    f = open('output/cluster_info.txt', 'w')
    for ngram_size in range(2, 4):
        tfidf = TfidfVectorizer(
            min_df=5,
            max_df=0.95,
            max_features=8000,
            ngram_range=(ngram_size, ngram_size),
            analyzer='word',
        )
        tfidf.fit(abstracts_df.text)
        text = tfidf.transform(abstracts_df.text)
        matplotlib.pyplot.ion()



        if DEBUG:
            range_max = 4
        else:
            range_max = 12

        for num_cluster in range(3, range_max):
            mbk = MiniBatchKMeans(n_clusters=num_cluster, init_size=1024, batch_size=2048, random_state=10)
            clusters = mbk.fit_predict(text)
            df_clusters = pd.DataFrame(text.todense()).groupby(clusters).mean()
            print_clusters_report(f, num_cluster, ngram_size, abstracts_df, df_clusters, clusters, tfidf)
            plot_tsne_pca(num_cluster, ngram_size, text, clusters)

    quit()


if __name__ == "__main__":
    main()

"""
Uncomment if you want to calculate and plot optimal clusters

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)

    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))

    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    plt.show()


find_optimal_clusters(text, 25)
"""
