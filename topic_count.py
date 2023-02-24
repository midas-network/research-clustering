import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from utils import build_corpus_words_only

n_features = 100000
n_top_words = 20


def process_field(fields):
    print("Loading dataset...")
    corpus_df = build_corpus_words_only(fields, do_stemming=True, do_remove_common=True)
    data = corpus_df["text"].tolist()
    count_vectorizer = CountVectorizer(
        max_df=0.95, min_df=2, max_features=n_features, stop_words="english", ngram_range=(1, 4)
    )
    count_data = count_vectorizer.fit_transform(data)
    print(count_data)
    counts = pd.DataFrame(
        list(zip(count_vectorizer.get_feature_names_out(),
                 count_data.sum(axis=0).tolist()[0])))\
        .sort_values(1,ascending=False)

    for i in range(1, 20):
        s = counts.iloc[:, i]
        counts['tick' + str(i)] = np.random.normal(s, 10)
        counts['tick' + str(i)] = counts['tick' + str(i)].astype(int)

    counts.to_csv('counts.csv', index=False)


def main():
    process_field({"meshTerms"})


if __name__ == "__main__":
    main()
    quit()
