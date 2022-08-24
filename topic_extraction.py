# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause
import os
from time import time
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, MiniBatchNMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

from tf_idf_sci import build_corpus, FIELDS, STEMDICT

n_samples = 2000
n_features = 100000
n_components = 10
n_top_words = 20
batch_size = 128
init = "nndsvda"


def process_field(fields):
    def plot_top_words(model, nmf_features, feature_names, n_top_words, title):

        fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
        axes = axes.flatten()
        document_count = pd.DataFrame(nmf_features).idxmax(axis=1).value_counts()
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
            top_features = [feature_names[i] for i in top_features_ind]
            top_unstemmed_features = []
            weights = topic[top_features_ind]

            for features in top_features:
                feature = ""
                for feature_word in features.split(" "):
                    feature += STEMDICT[feature_word] + " "
                top_unstemmed_features.append(feature.rstrip())

            ax = axes[topic_idx]
            ax.barh(top_unstemmed_features, weights, height=0.7)
            doc_count = document_count.get(topic_idx)
            if doc_count is None:
                doc_count = 0
            ax.set_title("Topic {} ({} docs)".format(topic_idx + 1, doc_count), fontdict={"fontsize": 30})
            ax.invert_yaxis()
            ax.tick_params(axis="both", which="major", labelsize=20)
            for i in "top right left".split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=40)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)

        plt.ioff()
        plt.savefig(output_dir + '{}.png'.format(title))
        plt.close(fig)





    # Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
    # to filter out useless terms early on: the posts are stripped of headers,
    # footers and quoted replies, and common English words, words occurring in
    # only one document or in at least 95% of the documents are removed.

    print("Loading dataset...")
    t0 = time()


    abstracts_df = build_corpus(fields, do_stemming=True, do_remove_common=True)
    data = abstracts_df["text"].tolist()

    # data, _ = fetch_20newsgroups(
    #     shuffle=True,
    #     random_state=1,
    #     remove=("headers", "footers", "quotes"),
    #     return_X_y=True,
    # )
    data_samples = data[:n_samples]
    print("done in %0.3fs." % (time() - t0))

    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95, min_df=2, max_features=n_features, stop_words="english", ngram_range=(1, 4)
    )
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))

    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(
        max_df=0.95, min_df=2, max_features=n_features, stop_words="english", ngram_range=(1, 4)
    )
    t0 = time()
    tf = tf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))
    print()

    # Fit the NMF model
    print(
        "Fitting the NMF model (Frobenius norm) with tf-idf features, "
        "n_samples=%d and n_features=%d..." % (n_samples, n_features)
    )
    t0 = time()
    nmf = NMF(
        n_components=n_components,
        random_state=1,
        init=init,
        beta_loss="frobenius",
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=1,
    ).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))

    nmf_features = nmf.transform(tfidf)

    output_dir = "output-te/" + "-".join(fields) + "/"
    os.makedirs(output_dir, exist_ok=True)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(
        nmf, nmf_features, tfidf_feature_names, n_top_words, "Topics in NMF model (Frobenius norm)"
    )

    # Fit the NMF model
    print(
        "\n" * 2,
        "Fitting the NMF model (generalized Kullback-Leibler "
        "divergence) with tf-idf features, n_samples=%d and n_features=%d..."
        % (n_samples, n_features),
    )
    t0 = time()
    nmf = NMF(
        n_components=n_components,
        random_state=1,
        init=init,
        beta_loss="kullback-leibler",
        solver="mu",
        max_iter=1000,
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=0.5,
    ).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))
    nmf_features = nmf.transform(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(
        nmf,
        nmf_features,
        tfidf_feature_names,
        n_top_words,
        "Topics in NMF model (generalized Kullback-Leibler divergence)",
    )

    # Fit the MiniBatchNMF model
    print(
        "\n" * 2,
        "Fitting the MiniBatchNMF model (Frobenius norm) with tf-idf "
        "features, n_samples=%d and n_features=%d, batch_size=%d..."
        % (n_samples, n_features, batch_size),
    )
    t0 = time()
    mbnmf = MiniBatchNMF(
        n_components=n_components,
        random_state=1,
        batch_size=batch_size,
        init=init,
        beta_loss="frobenius",
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=0.5,
    ).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))

    nmf_features = mbnmf.transform(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(
        mbnmf,
        nmf_features,
        tfidf_feature_names,
        n_top_words,
        "Topics in MiniBatchNMF model (Frobenius norm)",
    )

    # Fit the MiniBatchNMF model
    print(
        "\n" * 2,
        "Fitting the MiniBatchNMF model (generalized Kullback-Leibler "
        "divergence) with tf-idf features, n_samples=%d and n_features=%d, "
        "batch_size=%d..." % (n_samples, n_features, batch_size),
    )
    t0 = time()
    mbnmf = MiniBatchNMF(
        n_components=n_components,
        random_state=1,
        batch_size=batch_size,
        init=init,
        beta_loss="kullback-leibler",
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=0.5,
    ).fit(tfidf)
    nmf_features = mbnmf.transform(tfidf)
    print("done in %0.3fs." % (time() - t0))

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(
        mbnmf,
        nmf_features,
        tfidf_feature_names,
        n_top_words,
        "Topics in MiniBatchNMF model (generalized Kullback-Leibler divergence)",
    )

    print(
        "\n" * 2,
        "Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
        % (n_samples, n_features),
    )
    # lda = LatentDirichletAllocation(
    #     n_components=n_components,
    #     max_iter=5,
    #     learning_method="online",
    #     learning_offset=50.0,
    #     random_state=0,
    # )
    # t0 = time()
    # lda.fit(tf)
    # print("done in %0.3fs." % (time() - t0))
    #
    # tf_feature_names = tf_vectorizer.get_feature_names_out()
    # plot_top_words(lda, tf_feature_names, n_top_words, "Topics in LDA model")


def main():
    for field_set in FIELDS:
        process_field(field_set)


if __name__ == "__main__":
    main()
    quit()
