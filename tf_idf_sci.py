import os
import re
import threading
import time
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from yellowbrick.cluster import SilhouetteVisualizer

from util.update_data import update_data
from utils import build_corpus

# current date and time as a string in the format YYYYMMDD-HHMMSS
DATE_TIME = os.popen('date +"%Y%m%d-%H%M%S"').read().strip()
# the output directory

# OUTPUT_DIR is based on the date and time the script is run
OUTPUT_DIR = "output" + os.path.sep + "tf_idf_sci" + os.path.sep + DATE_TIME

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
warnings.filterwarnings("ignore")

# set DEBUG in your environment variables to enable debug mode
DEBUG = os.getenv("DEBUG", 'False').lower() in ('true', '1', 't')

FIELDS = [
          # {"pubmedKeywords"},
          # {"meshTerms"},
          # {"paperAbstract"},
          # {"pubmedKeywords", "meshTerms"},
          # {"pubmedKeywords", "paperAbstract"},
          # {"meshTerms", "paperAbstract"},
          {"pubmedKeywords", "meshTerms", "paperAbstract"}]


def plot_tsne_pca(output_dir, algo_name, cluster_num, ngram_size, data, labels):
    max_label = max(labels)
    max_items = np.arange(0, data.shape[0], 1, dtype=int)

    pca = PCA(n_components=2).fit_transform(data[max_items, :].todense())
    tsne = TSNE().fit_transform(
        PCA(n_components=min(50, int(data.shape[1] / 3))).fit_transform(data[max_items, :].todense()))

    label_subset = labels[max_items]
    label_subset = [cm.hsv(i / max_label) for i in label_subset[max_items]]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(pca[max_items, 0], pca[max_items, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot: {} Clusters; {}-word n-grams'.format(cluster_num, ngram_size))

    ax[1].scatter(tsne[max_items, 0], tsne[max_items, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot: {} Clusters; {}-word n-grams'.format(cluster_num, ngram_size))

    plt.savefig(
        output_dir + os.path.sep + "tnse_pca" + os.path.sep + '{}-{}clstrs-{}grm.png'.format(algo_name, cluster_num,
                                                                                             ngram_size))


def get_top_keywords(clusters_df, cluster_index, labels, n_terms):
    if len(clusters_df.iloc[:]) - 1 >= cluster_index:
        r = clusters_df.iloc[cluster_index]
        return ','.join([labels[t] for t in np.argsort(r)[-n_terms:]])
    else:
        return "none"


def print_clusters_report(f, algo_name, num_of_clusters, ngram_size, main_df, clusters_df, cluster_array, tfidf_obj,
                          sil_score):
    f.write('\n\n\n{} - {}-clusters, {}-grams, avg-sil-sc={}'.format(algo_name, num_of_clusters, ngram_size, sil_score))
    people_cluster_dict = {key: [] for key in range(num_of_clusters)}
    for person_index in main_df['people'].index:
        person_name = re.sub(r' \d{1,3}$', '', main_df['people'][person_index])
        person_cluster = cluster_array[person_index]
        people_cluster_dict[person_cluster].append(person_name)

    for cluster_index in people_cluster_dict:
        f.write('\n\tCluster ' + str(cluster_index + 1) + " -- " + str(len(people_cluster_dict[cluster_index])) +
                ' members')
        # f.write('\n\t\tResearchers (' + str(len(people_cluster_dict[cluster_index])) + '): ' + ','
        #      .join(people_cluster_dict[cluster_index]))
        f.write(
            '\n\tTop 20 n-grams: ' + get_top_keywords(clusters_df, cluster_index, tfidf_obj.get_feature_names_out(),
                                                      20))
    #flush the file
    f.flush()


def plot_sil(alg, algo_name, sample_silhouette_values, cluster_labels, n_clusters, ngram_size,
             silhouette_avg, clusters,
             text):
    plt.clf()
    visualizer = SilhouetteVisualizer(alg, colors='yellowbrick')  #
    visualizer.fit(text)
    visualizer.show(
        outpath=OUTPUT_DIR + os.path.sep + "tnse_pca" + os.path.sep + + 'sil-{}-{}clstrs-{}grm.png'.format(algo_name,
                                                                                                           n_clusters,
                                                                                                           ngram_size))
    # x = text
    # sc = SpectralClustering(n_clusters=4).fit(x)
    # labels = sc.labels_
    #
    # plt.scatter(x[:, 0], x[:, 1], c=labels)
    # plt.show()


def evaluate(clusterer, alg_name, f, X, ngram_size, abstracts_df, tfidf, num_clusters, fields, want_graphs):
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_score_text = str(silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    if want_graphs:
        try:
            plot_sil(clusterer, n_clusters=num_clusters, sample_silhouette_values=sample_silhouette_values,
                     silhouette_avg=silhouette_avg, clusters=clusterer, text=X, cluster_labels=cluster_labels,
                     ngram_size=ngram_size, algo_name=alg_name)
        except ValueError as e:
            print("Error making silhouette chart for {} with {} and {}-grams for {}".format(alg_name, num_clusters,
                                                                                            ngram_size,
                                                                                            " ".join(fields)))
            print(e)
    df_clusters = pd.DataFrame(X.todense()).groupby(cluster_labels).mean()
    print_clusters_report(f, alg_name, cluster_labels.max() + 1, ngram_size, abstracts_df, df_clusters, cluster_labels,
                          tfidf, silhouette_score_text)
    # this works, we just aren't using it right now
    # plot_tsne_pca(output_dir, alg_name, cluster_labels.max() + 1, ngram_size, X, cluster_labels)


def clean_output_directory():
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.endswith(".png"):
                os.remove(os.path.join(root, file))
            if file.endswith(".txt"):
                os.remove(os.path.join(root, file))


def main():
    print("Updating datasources...")
    update_data()
    print("Done updating datasources")

    for field_set in FIELDS:
        output_dir = OUTPUT_DIR + "/" + "-".join(field_set) + "/"
        os.makedirs(output_dir, exist_ok=True)
        print("Building corpus for fields: {}".format(field_set))
        # time how long this call takes

        start_time = time.time()
        corpus_df = build_corpus(field_set, do_stemming=True, do_remove_common=True)
        stop_time = time.time()
        print("Seconds to build corpus: ", stop_time - start_time)

        f = open(output_dir + "-".join(field_set) + '-cluster-info.txt', 'w')

        corpus_df.drop(columns="people")

        for ngram_size in range(1, 5):
            print("Building TF-IDF for {}-grams".format(ngram_size))
            tfidf_vectorizer = TfidfVectorizer(
                min_df=5,
                max_df=0.95,
                max_features=8000,
                ngram_range=(ngram_size, ngram_size),
                analyzer='word',
                token_pattern=r'(?u)\b[A-Za-z]+\b')

            tfidf_vectorizer.fit(corpus_df.text)
            print("Transforming text to TF-IDF")
            tf_idf = tfidf_vectorizer.transform(corpus_df.text)

            try:
                for num_cluster in range(5, 6):
                    print("Evaluating cluster of size {}".format(num_cluster))
                    mbk = MiniBatchKMeans(init="k-means++", n_clusters=num_cluster, init_size=1024, batch_size=2048,
                                          random_state=10)
                    k = KMeans(init="k-means++", n_clusters=num_cluster, n_init=10)
                    evaluate(mbk, "MiniBatchKMeans", f, tf_idf, ngram_size, corpus_df, tfidf_vectorizer,
                             num_cluster, field_set, want_graphs=False)
                    evaluate(k, "KMeans", f, tf_idf, ngram_size, corpus_df, tfidf_vectorizer,
                             num_cluster, field_set, want_graphs=False)
            except ValueError as e:
                print("Error occurred during evaluation of cluster of size {} ".format(num_cluster))
                print(e)
    quit()


if __name__ == "__main__":
    main()
