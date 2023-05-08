from pandas import read_csv
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import pandas as pd
MAX_NUM_CLUSTERS = 5
MAX_NUM_NGRAMS = 5

if __name__ == "__main__":
    data = read_csv("infectious_disease_modelling.csv")




    # converting column data to list
    people = data['Name'].tolist()
    text = data["Interests"].tolist()
    dataframe = pd.DataFrame({
        'people': people,
        'text': text
    })

    #create dataframe
    # topic, important words
    # mike, document2_text
    # document3, document3_text
    # ...
    # documentN,documentN_text


    for ngram_size in range(2, MAX_NUM_NGRAMS):

        tfidf_vectorizer = TfidfVectorizer(
            min_df=5,
            max_df=0.95,
            max_features=8000,
            ngram_range=(ngram_size, ngram_size))
            # ,
            # analyzer='word',
            # token_pattern=r'(?u)\b[A-Za-z]+\b')

        document_term_matrix = tfidf_vectorizer.fit_transform(dataframe.text)

        for num_cluster in range(2, MAX_NUM_CLUSTERS):
            mbk = MiniBatchKMeans(init="k-means++", n_clusters=num_cluster, init_size=1024, batch_size=2048,
                                  random_state=10)
            nmf = NMF(n_components=10,random_state=5)
            nmf.fit(document_term_matrix)
            components_df = pd.DataFrame(nmf.components_, columns=tfidf_vectorizer.get_feature_names())
            for topic in range(components_df.shape[0]):
                tmp = components_df.iloc[topic]
                print(f'For topic {topic + 1} the words with the highest value are:')
                print(tmp.nlargest(10))
                print('\n')


            #cluster_labels = nmf.fit_transform(document_term_matrix)

            #silhouette_avg = silhouette_score(document_term_matrix, cluster_labels)
            #silhouette_score_text = str(silhouette_avg)
