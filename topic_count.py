import pandas as pd
import numpy as np
import itertools
import json

from sklearn.feature_extraction.text import CountVectorizer

from utils import filter_mesh_terms, build_corpus_words_by_year, get_papers_per_word, unstemword
from enums.fields import Fields

n_features = 100000
n_top_words = 20


def fill_topic_df(full_count):
    year_range = [*range(min(full_count['date']),max(full_count['date'])+1)]
    topic_list = pd.unique(full_count['topic'])

    all_topics_df = pd.DataFrame(columns=['date', 'topic'], data=list(itertools.product(year_range, topic_list)))
    all_topics_df.insert(2, 'count', 0)

    # all_topics_df['count'] = all_topics_df['count'].map(full_count.set_index('year')['year'])
    all_topics_df.set_index(['date','topic'],inplace=True)
    all_topics_df.update(full_count.set_index(['date','topic']))
    all_topics_df.reset_index(drop=False,inplace=True)
    all_topics_df['count'] = all_topics_df['count'].astype(int)

    return all_topics_df

def process_field(field, ngram_count, min_year, max_year):
    print("Loading dataset...")

    corpus_dfs = build_corpus_words_by_year(field, ngram_count, min_year, max_year, do_stemming=True, do_remove_common=True)
    full_topic_list = set()
    select_topic_list = set()
    # year_counts = []

    if field == Fields.MESH_TERM:
        for year, df in corpus_dfs.items():
            if int(year) >= min_year or int(year) <= max_year:
                full_topic_list.update(df.keys())

        everything_everywhere = pd.DataFrame(columns=['date', 'topic'], data=list(itertools.product(corpus_dfs.keys(), full_topic_list)))
        everything_everywhere.insert(2, 'count', 0)
    else:
        everything_everywhere = pd.DataFrame(columns=['date', 'topic', 'count'])

    year_counts = []
    for year, df in corpus_dfs.items():
        if int(year) < min_year or int(year) > max_year:
                continue
        if field == Fields.MESH_TERM:
            filtered_data = filter_mesh_terms(df)
            for term in filtered_data.keys():
                everything_everywhere.loc[np.logical_and(everything_everywhere['date']==year, everything_everywhere['topic']==term), 'count'] = filtered_data[term]

            sorted_word_list = sorted(filtered_data.items(), key=lambda x:x[1], reverse=True)
            data = dict(sorted_word_list[0:20])

            select_topic_list.update(list(data.keys()))
            counts = pd.DataFrame.from_dict(data.items())
            counts.insert(0,'date',year)
            year_counts.append(counts)
            
        else:
            data = df["text"].tolist()

            count_vectorizer = CountVectorizer(
                max_df=1.0, min_df=1, max_features=n_features, stop_words="english", ngram_range=(ngram_count, ngram_count), binary=True
            )
            count_data = count_vectorizer.fit_transform(data)


            counts = pd.DataFrame(
                list(zip(count_vectorizer.get_feature_names_out(),
                        count_data.sum(axis=0).tolist()[0])))\
                .sort_values(1,ascending=False)
            counts = counts.rename(columns={0:'topic', 1: 'count'})
            counts.insert(0, 'date', year)
            select_topic_list.update(list(counts[0:20]['topic']))

            everything_everywhere = pd.concat([everything_everywhere, counts], ignore_index=True)
            year_counts.append(counts[0:20])

    # if field == Fields.MESH_TERM:
    all_topics_df = everything_everywhere[everything_everywhere['topic'].isin(list(select_topic_list))]

    if field != Fields.MESH_TERM:
        all_topics_df = fill_topic_df(all_topics_df)

    for word_set in year_counts:
        for term in select_topic_list:
            if field == Fields.MESH_TERM:
                if term not in word_set[0].values.tolist():
                    all_topics_df.loc[np.logical_and(all_topics_df['topic']==term,all_topics_df['date']==word_set['date'][0]), 'count'] = 0
            else:
                if term not in word_set['topic'].values.tolist():
                    all_topics_df.loc[np.logical_and(all_topics_df['topic']==term,all_topics_df['date']==word_set['date'].iloc[0]), 'count'] = 0


    all_topics_df['date'] = all_topics_df['date'].astype(str) + "/1/1"
    all_topics_df = all_topics_df.sort_values(by=['topic', 'date'], ascending=[True, True])
    
    # full_count = year_counts[0].copy()
    # for i in range(1,len(year_counts)-1):
    #     full_count = pd.concat([full_count, year_counts[i]])

    # full_count = full_count.rename(columns={0:'topic',1:'count'})
    # final_word_list = full_count.copy()
    # full_count = full_count.sort_values(by=['date', 'topic'], ascending=[True, True])
    # all_topics_df = fill_topic_df(full_count, corpus_dfs)
    # print(all_topics_df)

    final_word_list = all_topics_df.copy()
    paper_dict = get_papers_per_word(field, ngram_count, final_word_list, min_year, max_year, do_stemming=True, do_remove_common=True)

    # topic_list = full_count['topic'].tolist()
    if field != Fields.MESH_TERM:
        unstemmed_paper_dict = {}
        for topic in select_topic_list:
            if len(topic.split(' ')) > 1:
                unstemmed = ''
                for single_word in topic.split(' '):
                    unstemmed += unstemword(single_word) + ' '
                unstemmed_topic = unstemmed[:-1]
            else:
                unstemmed_topic = unstemword(topic)

            all_topics_df.loc[all_topics_df['topic']==topic, 'topic']=unstemmed_topic
            for year, words in paper_dict.items():
                for word in words.keys():
                    if word == topic:
                        if year not in unstemmed_paper_dict.keys():
                            unstemmed_paper_dict[year] = {}
                        if unstemmed_topic not in unstemmed_paper_dict[year].keys():
                            unstemmed_paper_dict[year][unstemmed_topic] = []

                        unstemmed_paper_dict[year][unstemmed_topic] = words[word]
    else:
        unstemmed_paper_dict = paper_dict

    count_filename = field.value + '-ngram_' + str(ngram_count) + '-counts.csv'
    paper_filename = field.value + '-ngram_' + str(ngram_count) + '-papers.json'
    all_topics_df.to_csv('output/' + count_filename, index=False)
    with open('output/' + paper_filename, 'w') as fp:
        fp.write(json.dumps(unstemmed_paper_dict, indent=4))


def main():
    do_all = False
    min_year = 2013
    max_year = 2022
    if do_all:
        fields = [Fields.ABSTRACT, Fields.PUBMED_KEYWORD]
        for field in fields:
           process_field(field, 1, min_year, max_year)
           process_field(field, 2, min_year, max_year)
           process_field(field, 3, min_year, max_year)
        process_field(Fields.MESH_TERM, 1, min_year, max_year)
    else:
        field = Fields.ABSTRACT
        ngram_count = 2
        process_field(field, ngram_count, min_year, max_year)


if __name__ == "__main__":
    main()
    quit()
