import math
import os
import re
import string
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from matplotlib import cm
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from yellowbrick.cluster import SilhouetteVisualizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
warnings.filterwarnings("ignore")
stop_words = set(stopwords.words('english'))

# set DEBUG in your environment variables to enable debug mode
DEBUG = os.getenv("DEBUG", 'False').lower() in ('true', '1', 't')


FIELDS = [{"pubmedKeywords"},
          {"pubmedKeywords", "meshTerms"},
          {"pubmedKeywords", "paperAbstract"},
          {"pubmedKeywords", "meshTerms", "paperAbstract"},
          {"pubmedKeywords", "paperAbstract"},
          {"meshTerms"},
          {"meshTerms", "paperAbstract"},
          {"paperAbstract"}]


def remove_stop_words_and_do_stemming(unfiltered_text, do_stemming, do_remove_common):
    unfiltered_text = remove_common(unfiltered_text.translate(str.maketrans("", "", string.punctuation)),
                                    do_remove_common)
    word_tokens = word_tokenize(unfiltered_text.lower())

    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    stem_words = []
    ps = PorterStemmer()
    for w in filtered_sentence:
        # don't add single letters
        if len(w) != 1:
            root_word = w
            if do_stemming:
                root_word = ps.stem(root_word)
            stem_words.append(root_word)
    return ' '.join(stem_words)


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

    plt.savefig(output_dir + '{}-{}clstrs-{}grm.png'.format(algo_name, cluster_num, ngram_size))


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
            '\n\tTop 20 n-grams: ' + get_top_keywords(clusters_df, cluster_index, tfidf_obj.get_feature_names(), 20))


def remove_common(text, bypass):
    if not bypass:
        return text

    text = re.sub(r'(©|copyright|Copyright|FUNDING|Funding Statement|This article is protected).*$', '', text)
    text = text.lower()
    text = re.sub(r'(infectious disease)', '', text)
    text = re.sub(r'mathematical model?', '', text)
    text = re.sub(r'policy|policies', '', text)
    text = re.sub(r'public health', '', text)
    text = re.sub(r'effective reproduction number', '', text)
    text = re.sub(r'public health interventions?', '', text)
    text = re.sub(r'formula: see text', '', text)
    text = re.sub(r'basic reproduction numbers?', '', text)
    text = re.sub(r'confidence intervals?', '', text)
    text = re.sub(r'transmission models?', '', text)
    text = re.sub(r'transmission dynam[a-z]*', '', text)
    text = re.sub(r'comp[a-z]* models?', '', text)
    text = re.sub(r'results? (show[a-z]*|sugg[a-z]*)', '', text)
    text = re.sub(r'attack rates?', '', text)
    text = re.sub(r'control strat[a-z]*', '', text)
    text = re.sub(r'population[a-z]*', '', text)
    text = re.sub(r'common[a-z]*', '', text)
    text = re.sub(r'case[a-z]*', '', text)
    text = re.sub(r'spread[a-z]*', '', text)
    text = re.sub(r'infectious[a-z]*', '', text)
    text = re.sub(r'computat[a-z]*', '', text)
    text = re.sub(r'susceptib[a-z]*', '', text)
    text = re.sub(r'sensitivi[a-z]*', '', text)
    text = re.sub(r'transmitt[a-z]*', '', text)
    text = re.sub(r'fatality[a-z]*', '', text)
    text = re.sub(r'vector[a-z]*', '', text)
    text = re.sub(r'strateg[a-z]*', '', text)
    text = re.sub(r'observ[a-z]*', '', text)
    text = re.sub(r'specific[a-z]*', '', text)
    text = re.sub(r'incubation period[a-z]*', '', text)
    text = re.sub(r'world health org[a-z]*', '', text)
    text = re.sub(r'vaccine eff[a-z]*', '', text)
    text = re.sub(r'illnes[a-z]*', '', text)
    text = re.sub(r'qualityadjusted[a-z]*', '', text)
    text = re.sub(r'forecast[a-z]*', '', text)
    text = re.sub(r'models? predic[a-z]*', '', text)
    text = re.sub(r'centers for disease control and prevention', '', text)
    text = re.sub(r'social distanc[a-z]* (meas[a-z]*)?', '', text)
    text = re.sub(r'clin[a-z]* tria[a-z]*', '', text)
    text = re.sub(r'reproduct[a-z]* numbers?', '', text)
    text = re.sub(r'machine learn[a-z]*', '', text)
    text = re.sub(r'disease[a-z]* trans[a-z]*', '', text)
    text = re.sub(r'cohor[a-z]* stud[a-z]*', '', text)
    text = re.sub(r'vacc[a-z]* strat[a-z]*', '', text)
    text = re.sub(r'receiv[a-z]* operat[a-z]* charact[a-z]* curv[a-z]*', '', text)
    text = re.sub(r'environmental health', '', text)
    text = re.sub(r'although', '', text)
    text = re.sub(r'infection[a-z]*', '', text)
    text = re.sub(r'monitoring', '', text)
    text = re.sub(r'theoretical', '', text)
    text = re.sub(r'model[a-z]*', '', text)
    text = re.sub(r'disease', '', text)
    text = re.sub(r'communicable', '', text)
    text = re.sub(r'golbal health', '', text)
    text = re.sub(r'virus', '', text)
    text = re.sub(r'infection[a-z]*', '', text)
    text = re.sub(r'biological', '', text)
    text = re.sub(r'disease outbreak[a-z]*', '', text)
    text = re.sub(r'human[a-z]*', '', text)
    text = re.sub(r'pandemic[a-z]*', '', text)
    text = re.sub(r'paper[a-z]*', '', text)
    text = re.sub(r'probabilistic', '', text)
    text = re.sub(r'statu', '', text)
    text = re.sub(r'united states', '', text)
    text = re.sub(r'past', '', text)
    text = re.sub(r'stillhigh', '', text)
    text = re.sub(r'methods', '', text)
    text = re.sub(r'introduction', '', text)
    text = re.sub(r'million', '', text)
    text = re.sub(r'informed', '', text)
    text = re.sub(r'national institutes of health', '', text)
    text = re.sub(r'decades', '', text)
    text = re.sub(r'disease outbreak[a-z]*', '', text)
    text = re.sub(r'decision mak[a-z]*', '', text)
    text = re.sub(r'cente[a-z]* for dis[a-z]* cont[a-z]*', '', text)
    text = re.sub(r'data interpretation', '', text)
    text = re.sub(r'tignificancethis', '', text)
    text = re.sub(r'model evaulation[a-z]*', '', text)
    text = re.sub(r'communicable', '', text)
    text = re.sub(r'disease[a-z]*', '', text)
    text = re.sub(r'incidence', '', text)
    text = re.sub(r'risk factor[a-z]*', '', text)
    text = re.sub(r'usa', '', text)
    text = re.sub(r'health planner[a-z]*', '', text)
    text = re.sub(r'ensemble', '', text)
    text = re.sub(r'paper compare[a-z]*', '', text)
    text = re.sub(r'mortality', '', text)
    text = re.sub(r'probabil[a-z]*', '', text)
    text = re.sub(r'epidemi[a-z]*', '', text)
    text = re.sub(r'nan', '', text)
    text = re.sub(r'evaluation', '', text)
    text = re.sub(r'vaccin[a-z]*', '', text)
    text = re.sub(r'season[a-z]*', '', text)
    text = re.sub(r'decisi[a-z]*', '', text)
    text = re.sub(r'global health', '', text)
    text = re.sub(r'prediction[a-z]*', '', text)
    text = re.sub(r'expos[a-z]*', '', text)
    text = re.sub(r'outbreak[a-z]*', '', text)
    text = re.sub(r'data[a-z]*', '', text)
    text = re.sub(r'simulat[a-z]*', '', text)
    text = re.sub(r'middleincome[a-z]*', '', text)
    text = re.sub(r'health care[a-z]*', '', text)
    text = re.sub(r'qualityadj[a-z]*', '', text)
    text = re.sub(r'review[a-z]*', '', text)
    text = re.sub(r'patient[a-z]*', '', text)
    text = re.sub(r'information[a-z]*', '', text)
    text = re.sub(r'surv[a-z]*', '', text)
    text = re.sub(r'result[a-z]*', '', text)
    text = re.sub(r'estimate[a-z]*', '', text)
    text = re.sub(r'algorithim[a-z]*', '', text)
    text = re.sub(r'stochastic[a-z]*', '', text)
    text = re.sub(r'processes[a-z]*', '', text)
    text = re.sub(r'intervent[a-z]*', '', text)
    text = re.sub(r'theoretical', '', text)
    text = re.sub(r'compare[a-z]*', '', text)
    text = re.sub(r'studies', '', text)
    text = re.sub(r'computer', '', text)
    text = re.sub(r'analysis', '', text)
    text = re.sub(r'healthcare', '', text)
    text = re.sub(r'testing', '', text)
    text = re.sub(r'vaccine[a-z]*', '', text)
    text = re.sub(r'test[a-z]*', '', text)
    text = re.sub(r'stud[a-z]*', '', text)
    text = re.sub(r'preval[a-z]*', '', text)
    text = re.sub(r'mitigati[a-z]*', '', text)
    text = re.sub(r'vaccine[a-z]*', '', text)
    text = re.sub(r'crosssectional', '', text)
    text = re.sub(r'distribution[a-z]*', '', text)
    text = re.sub(r'significancethis', '', text)
    text = re.sub(r'evaluation[a-z]*', '', text)
    text = re.sub(r'assessment', '', text)
    text = re.sub(r'background', '', text)
    text = re.sub(r'mitigate', '', text)
    text = re.sub(r'trasnsmiss[a-z]*', '', text)
    text = re.sub(r'pattern[a-z]*', '', text)
    text = re.sub(r'spatial', '', text)
    text = re.sub(r'mortality', '', text)
    text = re.sub(r'agentbased', '', text)
    text = re.sub(r'emergency', '', text)
    text = re.sub(r'contact tracing', '', text)
    text = re.sub(r'mathema[a-z]* mode', '', text)
    text = re.sub(r'determine', '', text)
    text = re.sub(r'united', '', text)
    text = re.sub(r'research', '', text)
    text = re.sub(r'agent', '', text)
    text = re.sub(r'experimen[a-z]*', '', text)
    text = re.sub(r'evidence', '', text)
    text = re.sub(r'health', '', text)
    text = re.sub(r'factors', '', text)
    text = re.sub(r'public heal[a-z]*', '', text)
    text = re.sub(r'mathemati[a-z]*', '', text)
    text = re.sub(r'estima[a-z]*', '', text)
    return text


def build_corpus(field_set, do_stemming, do_remove_common):
    people = pd.read_json('data/people.json')
    papers = pd.read_json('data/papers.json')
    people = people[people['publications'].map(lambda d: len(d) > 0)]
    people_list = []
    text_list = []

    for personIdx in people.index:
        all_person_text = ""
        title = ""
        for paperIdx in people['publications'][personIdx]:
            row = papers.loc[papers['uri'] == paperIdx]

            if not row.empty and len(paperIdx) > 32:
                title = remove_common(row['title'].values[0], do_remove_common)
                for field in field_set:
                    abstract = row[field]
                    if isinstance(abstract, pd.Series):
                        if not isinstance(row[field].values[0], list) and not isinstance(row[field].values[0],
                                                                                         str) and math.isnan(
                                row[field].values[0]):
                            continue;
                        abstract = " ".join(row[field].values[0])
                    if isinstance(abstract, str):
                        abstract = remove_common(abstract, do_remove_common)

                    all_person_text += " " + abstract

        all_person_text = title + all_person_text
        all_person_text = remove_common(all_person_text, do_remove_common)
        person = people['name'][personIdx] + " " + str(personIdx)
        people_list.append(person)
        list_of_words = remove_stop_words_and_do_stemming(all_person_text, do_stemming, do_remove_common)
        text_list.append(''.join(list_of_words))

    df = pd.DataFrame({
        'people': people_list,
        'text': text_list
    })
    return df


def plot_sil(output_dir, alg, algo_name, sample_silhouette_values, cluster_labels, n_clusters, ngram_size, silhouette_avg, clusters,
             text):
    plt.clf()
    visualizer = SilhouetteVisualizer(alg, colors='yellowbrick')  #
    visualizer.fit(text)
    visualizer.show(outpath=output_dir + 'sil-{}-{}clstrs-{}grm.png'.format(algo_name, n_clusters, ngram_size))
    # x = text
    # sc = SpectralClustering(n_clusters=4).fit(x)
    # labels = sc.labels_
    #
    # plt.scatter(x[:, 0], x[:, 1], c=labels)
    # plt.show()


def evaluate(output_dir, clusterer, alg_name, f, X, ngram_size, abstracts_df, tfidf, num_clusters, fields):
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_score_text = str(silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    try:
        plot_sil(output_dir, clusterer, n_clusters=num_clusters, sample_silhouette_values=sample_silhouette_values,
                 silhouette_avg=silhouette_avg, clusters=clusterer, text=X, cluster_labels=cluster_labels,
                 ngram_size=ngram_size, algo_name=alg_name)
    except ValueError as e:
        print("Error making silhouette chart for {} with {} and {}-grams for {}".format(alg_name, num_clusters,
                                                                                       ngram_size, " ".join(fields)))
        print(e)
    df_clusters = pd.DataFrame(X.todense()).groupby(cluster_labels).mean()
    print_clusters_report(f, alg_name, cluster_labels.max() + 1, ngram_size, abstracts_df, df_clusters, cluster_labels,
                          tfidf, silhouette_score_text)
    #this works, we just aren't using it right now
    #plot_tsne_pca(output_dir, alg_name, cluster_labels.max() + 1, ngram_size, X, cluster_labels)


def main():
    range_max = 8
    for field_set in FIELDS:
        output_dir = "output/" + "-".join(field_set) + "/"
        os.makedirs(output_dir, exist_ok=True)
        abstracts_df = build_corpus(field_set, do_stemming=True, do_remove_common=True)
        f = open(output_dir + "-".join(field_set) + '-cluster-info.txt', 'w')

        abstracts_df.drop(columns="people")
        text_arr = abstracts_df.iloc[:, 1:].values
        for ngram_size in range(2, 5):
            tfidf_vectorizer = TfidfVectorizer(
                min_df=5,
                max_df=0.95,
                max_features=8000,
                ngram_range=(ngram_size, ngram_size),
                analyzer='word',
                token_pattern=r'(?u)\b[A-Za-z]+\b')
            tfidf_vectorizer.fit(abstracts_df.text)
            tf_idf = tfidf_vectorizer.transform(abstracts_df.text)

            try:
                for num_cluster in range(2, range_max):
                    mbk = MiniBatchKMeans(init="k-means++", n_clusters=num_cluster, init_size=1024, batch_size=2048,
                                          random_state=10)
                    k = KMeans(init="k-means++", n_clusters=num_cluster, n_init=10)
                    evaluate(output_dir, mbk, "MiniBatchKMeans", f, tf_idf, ngram_size, abstracts_df, tfidf_vectorizer,
                             num_cluster, field_set)
                    evaluate(output_dir, k, "KMeans", f, tf_idf, ngram_size, abstracts_df, tfidf_vectorizer,
                             num_cluster, field_set)
            except ValueError as e:
                print("Error occurred during evaluation of cluster of size {} ".format(num_cluster))
                print(e)
    quit()


if __name__ == "__main__":
    main()

