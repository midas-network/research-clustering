import pandas as pd
import numpy as np
import sys
import itertools

from sklearn.feature_extraction.text import CountVectorizer

from utils import build_corpus_words_only, build_corpus_words_only_by_year

n_features = 100000
n_top_words = 20


def fill_topic_df(full_count):
    year_range = [*range(min(full_count['year']),max(full_count['year'])+1)]
    topic_list = pd.unique(full_count['topic'])

    all_topics_df = pd.DataFrame(columns=['year', 'topic'], data=list(itertools.product(year_range, topic_list)))
    all_topics_df.insert(2, 'count', 0)

    # all_topics_df['count'] = all_topics_df['count'].map(full_count.set_index('year')['year'])
    all_topics_df.set_index(['year','topic'],inplace=True)
    all_topics_df.update(full_count.set_index(['year','topic']))
    all_topics_df.reset_index(drop=False,inplace=True)
    all_topics_df['count'] = all_topics_df['count'].astype(int)
    all_topics_df['year'] = all_topics_df['year'].astype(str) + "/1/1"

    all_topics_df = all_topics_df.sort_values(by=['topic', 'year'], ascending=[True, True])
    print(all_topics_df)

    return all_topics_df

def process_field(fields, count_type):
    print("Loading dataset...")
    if count_type== '-a':
        corpus_df = build_corpus_words_only(fields, do_stemming=True, do_remove_common=True)
        import pdb; pdb.set_trace()
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

    elif count_type== '-b':
        corpus_dfs, word_to_paper_dict = build_corpus_words_only_by_year(fields, do_stemming=True, do_remove_common=True)
        # import pdb; pdb.set_trace()
        year_counts = []
        for year, df in corpus_dfs.items():
            if int(year) < 2010:
                 continue
            data = df["text"].tolist()
            try:
                count_vectorizer = CountVectorizer(
                    max_df=0.95, min_df=2, max_features=n_features, stop_words="english", ngram_range=(1, 4)
                )
                count_data = count_vectorizer.fit_transform(data)
            except ValueError:
                count_vectorizer = CountVectorizer(
                    max_features=n_features, stop_words="english", ngram_range=(1, 4)
                )
                count_data = count_vectorizer.fit_transform(data)
            counts = pd.DataFrame(
                list(zip(count_vectorizer.get_feature_names_out(),
                        count_data.sum(axis=0).tolist()[0])))\
                .sort_values(1,ascending=False)
            counts.insert(0, 'year', year)
            year_counts.append(counts[0:19])
            # year_counts.append(counts)

        full_count = year_counts[0].copy()
        for i in range(1,len(year_counts)-1):
            full_count = pd.concat([full_count, year_counts[i]])

        full_count = full_count.rename(columns={0:'topic',1:'count'})
        full_count = full_count.sort_values(by=['year', 'topic'], ascending=[True, True])
        all_topics_df = fill_topic_df(full_count)
        print(all_topics_df)
        all_topics_df.to_csv('year_counts_full.csv', index=False)


def main(count_type):
    process_field({"meshTerms"}, count_type)


if __name__ == "__main__":
    main(sys.argv[1])
    quit()
