import string
import warnings
import matplotlib.pyplot
import matplotlib.pyplot as plt
import nltk
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

warnings.filterwarnings("ignore")

DEBUG = False

people = pd.read_json('data/people.json')
papers = pd.read_json('data/papers.json')
if DEBUG:
    papers = papers.head(1000)

people_list = []
text_list = []

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def generate_ngrams(list_of_single_words):
    if len(list_of_single_words):
        list_of_single_words = word_tokenize(list_of_single_words)
        n_grams = ngrams(list_of_single_words, 2)
        return [' '.join(g) for g in n_grams]
    else:
        return []


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


for idx in people.index:
    all_person_text = ""
    title = ""
    for paperIdx in people['publications'][idx]:
        row = papers.loc[papers['uri'] == paperIdx]

        if not row.empty and len(paperIdx) > 32:
            title = row['title'].values[0]
            abstract = row['paperAbstract']
            if len(abstract) == 1:
                abstract = abstract.values.item()
            if isinstance(abstract, str):
                all_person_text += " " + abstract
        else:
            if not DEBUG:
                print("Missing paper: " + paperIdx)
    all_person_text = title + all_person_text
    person = people['name'][idx] + " " + str(idx)
    people_list.append(person)
    list_of_words = remove_stop_words_and_do_stemming(all_person_text)
    # bigram_list = generate_ngrams(list_of_words)
    text_list.append(''.join(list_of_words))

df = pd.DataFrame({
    'people': people_list,
    'text': text_list
})

tfidf = TfidfVectorizer(
    min_df=5,
    max_df=0.95,
    max_features=8000,
    ngram_range=(2, 2),
    analyzer='word',

)
tfidf.fit(df.text)
text = tfidf.transform(df.text)
matplotlib.pyplot.ion()
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


def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=500, replace=False)

    pca = PCA(n_components=2).fit_transform(data[max_items, :].todense())
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items, :].todense()))

    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i / max_label) for i in label_subset[idx]]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    f.show()


for x in range(3, 12):
    mbk = MiniBatchKMeans(n_clusters=x, init_size=1024, batch_size=2048, random_state=10)
    clusters = mbk.fit_predict(text)
    df_clusters = pd.DataFrame(text.todense()).groupby(clusters).mean()


    def get_top_keywords(cluster_index, labels, n_terms):
        r = df_clusters.iloc[cluster_index]
        return ','.join([labels[t] for t in np.argsort(r)[-n_terms:]])


    def print_clusters_report(cluster_array):
        print('\n{} Clusters'.format(x))
        people_cluster_dict = {key: [] for key in range(x)}
        for person_index in df['people'].index:
            person_name = df['people'][person_index]
            person_cluster = cluster_array[person_index]
            people_cluster_dict[person_cluster].append(person_name)

        for cluster_index in people_cluster_dict:
            print('\tCluster ' + str(cluster_index + 1))
            print('\tResearchers (' + str(len(people_cluster_dict[cluster_index])) + '): ' + ','
                  .join(people_cluster_dict[cluster_index]))
            print('\tTop N-grams: ' + get_top_keywords(cluster_index, tfidf.get_feature_names(), 20))


    print_clusters_report(clusters)

    plot_tsne_pca(text, clusters)
