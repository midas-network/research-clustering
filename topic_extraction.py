import os
from time import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import NMF, MiniBatchNMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from tf_idf_sci import build_corpus, FIELDS
from utils import STEMDICT, write_cluster_to_json, get_output_dir_name

n_samples = 2000
n_features = 100000
n_components = 5
n_top_words = 20
batch_size = 128
init = "nndsvda"

OUTPUT_DIR = get_output_dir_name("midas_viz_data")


def get_people_for_topic(people, series):
    print("Getting people for topic", people, series)
    people_per_topic = {}
    for i in range(0, len(series)):
        people_per_topic[''.join([j for j in people[i] if not j.isdigit()]).strip()] = str(series[i])

    return people_per_topic


def plot_top_words(model, nmf_features, feature_names, title, people, fields, output_dir,
                   want_graph=False):
    print("ptw - getting document count")
    document_count = pd.DataFrame(nmf_features).idxmax(axis=1).value_counts()
    feature_dict = {}
    print("ptw - number of topics " + str(len(model.components_)))
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
        top_features = [feature_names[i] for i in top_features_ind]
        print("top_features", top_features)
        top_unstemmed_features = []
        weights = topic[top_features_ind]

        for features in top_features:
            feature = ""
            for feature_word in features.split(" "):
                # feature += STEMDICT[feature_word] + " "
                words = ""
                for word, count in STEMDICT[feature_word].items():
                    top_unstemmed_features.append(word.rstrip())

        print("top_unstemmed_features", top_unstemmed_features)
        # feature_dict[topic_idx] = ",".join(top_unstemmed_features)
        feature_dict[topic_idx] = ",".join(top_features)
        print("writing cluster to json: ", title + "-" + "-".join(fields))
        write_cluster_to_json(output_dir, title + "-" + "-".join(fields),
                              get_people_for_topic(people, pd.DataFrame(nmf_features).idxmax(axis=1)), feature_dict)

    if want_graph:
        fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
        axes = axes.flatten()
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

    if want_graph:
        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)

        plt.ioff()
        plt.savefig(output_dir + '{}.png'.format(title))
        plt.close(fig)


def fit_nmfs_frobenius(tfidf_vectorizer, tfidf, tf_vectorizer, tf, people, field):
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

    output_dir = "output-te/" + "-".join(field) + "/"
    os.makedirs(output_dir, exist_ok=True)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(
        nmf, nmf_features, tfidf_feature_names, n_top_words, "Topics in NMF model (Frobenius norm)", people, field,
        output_dir
    )


def fit_nmfc_kl(tfidf_vectorizer, tfidf, people, fields, output_dir):
    t0 = time()
    nmf_c = NMF(
        n_components=n_components,
        random_state=1,
        init=init,
        beta_loss="kullback-leibler",
        solver="mu",
        max_iter=1000,
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=0.5,
    )

    nmf = nmf_c.fit(tfidf)

    print("done in %0.3fs." % (time() - t0))
    nmf_features = nmf.transform(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(
        nmf,
        nmf_features,
        tfidf_feature_names,
        n_top_words,
        "Topics in NMF model (generalized Kullback-Leibler divergence)",
        people,
        fields,
        output_dir
    )


def fit_mb_nmf(tfidf_vectorizer, tfidf, people, fields, output_dir):
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
        "Topics in MiniBatchNMF model (Frobenius norm)",
        people,
        fields,
        output_dir
    )


def fit_mb_kl(tfidf_vectorizer, tfidf, people, fields, output_dir):
    t0 = time()
    print("Fitting the MiniBatchNMF model (generalized Kullback-Leibler divergence) with tf-idf features...")
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
    print("done in %0.3fs." % (time() - t0))

    print("Transforming data")
    t0 = time()
    nmf_features = mbnmf.transform(tfidf)
    print("done in %0.3fs." % (time() - t0))

    print("output-dir: " + output_dir)

    print("Getting feature names")
    t0 = time()
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    print("done in %0.3fs." % (time() - t0))

    print("Plotting top words")
    t0 = time()
    plot_top_words(
        mbnmf,
        nmf_features,
        tfidf_feature_names,
        "Topics in MiniBatchNMF model (generalized Kullback-Leibler divergence)",
        people,
        fields,
        output_dir,
        want_graph=False
    )
    print("done in %0.3fs." % (time() - t0))


def process_field(field, bfit_nmfs_frobenius, bfit_nmfc_kl, bfit_mb_nmf, bfit_mb_kl, output_dir):
    print("Processing field", field)
    print("Building corpus")
    t0 = time()
    corpus_dfs = build_corpus(field, do_stemming=True, do_remove_common=True)

    print("done in %0.3fs." % (time() - t0))

    print("Loading dataset...")
    t0 = time()

    data = corpus_dfs["text"].tolist()
    # texts = abstracts_df["text"]
    # dictionary = Dictionary(texts)
    people = corpus_dfs["people"].tolist()

    def convert(lst):
        return ''.join(lst).split()

    data_samples = data[:n_samples]
    print("done in %0.3fs." % (time() - t0))
    data_samnples_list = [convert(item) for item in data_samples]
    # Use tf-idf features for NMF.
    ##if tfidf is already saved to disk, load it

    print("Extracting tf-idf features...")
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95, min_df=2, max_features=n_features, stop_words="english", ngram_range=(1, 4)
    )
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))

    t0 = time()
    print("Running count vectorizer")
    # Use tf (raw term count) features for LDA.
    # print("Extracting tf features for LDA...")
    # THIS IS THE DICT FOLLOWED BY THE COUNTS!
    tf_vectorizer = CountVectorizer(
        max_df=0.95, min_df=2, max_features=n_features, stop_words="english", ngram_range=(1, 4)
    )
    print("done in %0.3fs." % (time() - t0))
    print("transforming fitting/transforming data samples")
    t0 = time()
    tf = tf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))

    if bfit_nmfs_frobenius:
        fit_nmfs_frobenius(tfidf_vectorizer, tfidf, tf_vectorizer, tf, people, field)
    if bfit_nmfc_kl:
        fit_nmfc_kl(tfidf_vectorizer, tfidf, people, field)
    if bfit_mb_nmf:
        fit_mb_nmf(tfidf_vectorizer, tfidf, people, field, output_dir)
    if bfit_mb_kl:
        fit_mb_kl(tfidf_vectorizer, tfidf, people, field, output_dir)


def main():
    for field_set in FIELDS:
        output_dir = OUTPUT_DIR + "/" + "-".join(field_set) + "/"
        os.makedirs(output_dir, exist_ok=True)
        process_field(field_set, False, False, False, True, output_dir)


if __name__ == "__main__":
    main()
    quit()
