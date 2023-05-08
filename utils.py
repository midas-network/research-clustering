import json
from enum import Enum

import math
import re
import string

from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pandas as pd;

from UnstemMethod import UnstemMethod

STEMDICT = {}
stop_words = set(stopwords.words('english'))


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
    text = re.sub(r',', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def write_cluster_to_json(title, bins, features):
    with open("data/people-with-clusters-" + title + ".json", "w") as outfile:
        json.dump(bins, outfile)

    with open("data/cluster-info-" + title + ".json", "w") as outfile2:
        json.dump(features, outfile2)


def build_corpus(field_set, do_stemming, do_remove_common):
    people = pd.read_json('data_sources/people.json')
    papers = pd.read_json('data_sources/papers.json')
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
        person = people['name'][personIdx] + "#" + people['uri'][personIdx]
        people_list.append(person)
        list_of_words = remove_stop_words_and_do_stemming(all_person_text, do_stemming, do_remove_common)
        text_list.append(''.join(list_of_words))

    df = pd.DataFrame({
        'people': people_list,
        'text': text_list
    })
    return df


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
                stem_word = ps.stem(root_word)
                if stem_word in STEMDICT:
                    indiv_stem_word_dict = STEMDICT[stem_word]
                else:
                    indiv_stem_word_dict = {}
                if root_word in indiv_stem_word_dict:
                    indiv_stem_word_dict[root_word] += 1;
                else:
                    indiv_stem_word_dict[root_word] = 1;

                STEMDICT[stem_word] = indiv_stem_word_dict
            stem_words.append(stem_word)

    return ' '.join(stem_words)


def unstemword(stemmed_word, unstem_method=UnstemMethod.SELECT_MOST_FREQUENT):
    if stemmed_word in STEMDICT:
        indiv_stem_word_dict = STEMDICT[stemmed_word]
        if unstem_method == UnstemMethod.SELECT_MOST_FREQUENT:
            return max(indiv_stem_word_dict.items(), key=lambda x: x[1])[0]
        elif unstem_method == UnstemMethod.SELECT_SHORTEST:
            return (min(indiv_stem_word_dict, key=len))

        elif unstem_method == UnstemMethod.SELECT_LONGEST:
            return (max(indiv_stem_word_dict, key=len))
        else:
            raise Exception("unsupported UnstemMethod")

    else:
        # raise Exception("stemmed_word does not exist in STEMDICT")
        print("stemmed_word does not exist in STEMDICT")
        return stemmed_word


def build_corpus_words_only(field_set, do_stemming, do_remove_common):
    papers = pd.read_json('data_sources/papers.json')
    text_list = []
    all_person_text = ""
    for index, row in papers.iterrows():
        if index > 100:
            df = pd.DataFrame({
                'text': text_list
            })
            return df
        title = remove_common(row['title'], do_remove_common)
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
        list_of_words = remove_stop_words_and_do_stemming(all_person_text, do_stemming, do_remove_common)
        text_list.append(''.join(list_of_words))

    df = pd.DataFrame({
        'text': text_list
    })
    return df


def build_corpus_words_only_by_year(field_set, do_stemming, do_remove_common):
    papers = pd.read_json('data_sources/papers.json')
    text_dict = {}
    word_to_paper_dict = {}
    # import pdb;pdb.set_trace()
    for index, row in papers.iterrows():
        all_person_text = ""
        # if index > 100:
        #     dfs = {}
        #     for key, value in text_dict.items():
        #         dfs[key] = pd.DataFrame({'text': value})
        #     return dfs
        title = remove_common(row['title'], do_remove_common)
        # TODO: maybe try earliest/latest date, see which is most filled, etc.
        try:
            year = int(row['datePublished'][-4:])
        except TypeError:
            try:
                year = int(row['articleDate'][-4:])
            except TypeError:
                year = 1993
        if year < 2010:
            continue
        for field in field_set:
            abstract = row[field]
            if isinstance(abstract, pd.Series):
                if not isinstance(row[field].values[0], list) and not isinstance(row[field].values[0],
                                                                                 str) and math.isnan(row[field].values[0]): continue;
                abstract = " ".join(row[field].values[0])
            if isinstance(abstract, list):
                abstract = " ".join([item for item in row[field]])
            if isinstance(abstract, str):
                abstract = remove_common(abstract, do_remove_common)
                all_person_text += " " + abstract

        all_person_text = title + all_person_text
        all_person_text = remove_common(all_person_text, do_remove_common)
        list_of_words = remove_stop_words_and_do_stemming(all_person_text, do_stemming, do_remove_common)

        if year in text_dict.keys():
            text_dict[year].append(''.join(list_of_words))
        else:
            text_dict[year] = [list_of_words]

    dfs = {}
    for key, value in text_dict.items():
        dfs[key] = pd.DataFrame({'text': value})
    return dfs

def get_papers_per_word(field_set, final_word_list, do_stemming, do_remove_common):
    papers = pd.read_json('data_sources/papers.json')
    paper_dict = {}

    for index, row in papers.iterrows():
        all_person_text = ""
        # if index > 100:
        #     dfs = {}
        #     for key, value in text_dict.items():
        #         dfs[key] = pd.DataFrame({'text': value})
        #     return paper_dict
        title = remove_common(row['title'], do_remove_common)
        try:
            year = int(row['datePublished'][-4:])
        except TypeError:
            try:
                year = int(row['articleDate'][-4:])
            except TypeError:
                year = 1993
        if year < 2010:
            continue

        for field in field_set:
            abstract = row[field]
            if isinstance(abstract, pd.Series):
                if not isinstance(row[field].values[0], list) and not isinstance(row[field].values[0],
                                                                                 str) and math.isnan(row[field].values[0]): continue;
                abstract = " ".join(row[field].values[0])
            if isinstance(abstract, list):
                abstract = " ".join([item for item in row[field]])
            if isinstance(abstract, str):
                abstract = remove_common(abstract, do_remove_common)
                all_person_text += " " + abstract

        all_person_text = title + all_person_text
        all_person_text = remove_common(all_person_text, do_remove_common)
        list_of_words = remove_stop_words_and_do_stemming(all_person_text, do_stemming, do_remove_common)

        # for word in list_of_words.split(' '):
        #     if not final_word_list.loc[(final_word_list['year']==year) & (final_word_list['topic']==word)].empty:
        for word in final_word_list.loc[final_word_list['year']==year, 'topic'].tolist():
            if word in list_of_words:
                if year in paper_dict.keys():
                    if word in paper_dict[year].keys():
                        if index not in paper_dict[year][word]:
                            paper_dict[year][word].append({'title': row['title'], 'uri': row['uri']})
                    else:
                        paper_dict[year][word] = [{'title': row['title'], 'uri': row['uri']}]
                else:
                    paper_dict[year] = {}
                    paper_dict[year][word] = [{'title': row['title'], 'uri': row['uri']}]

    return paper_dict
