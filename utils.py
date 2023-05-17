import json
import os
from enum import Enum

import math
import re
import string

from nltk import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords

import pandas as pd;

from UnstemMethod import UnstemMethod
from fields import Fields

STEMDICT = {}
stop_words = set(stopwords.words('english'))


def remove_common(text, ngram_count, bypass):
    if not bypass:
        return text

    text = re.sub(r'(©|copyright|Copyright|FUNDING|Funding Statement|This article is protected).*$', '', text)
    text = text.lower()
    text = re.sub(r'\b[a-z]\b', '', text)
    text = re.sub(r'\b[a-z][a-z]\b', '', text)
    if ngram_count <= 3:
        text = re.sub(r'effective reproduction number', '', text)
        text = re.sub(r'public health interventions?', '', text)
        text = re.sub(r'formula: see text', '', text)
        text = re.sub(r'basic reproduction numbers?', '', text)
        text = re.sub(r'world health org[a-z]*', '', text)
        text = re.sub(r'centers for disease control and prevention', '', text)
        text = re.sub(r'social distanc[a-z]* (meas[a-z]*)?', '', text)
        text = re.sub(r'receiv[a-z]* operat[a-z]* charact[a-z]* curv[a-z]*', '', text)
        text = re.sub(r'national institutes of health', '', text)
        text = re.sub(r'cente[a-z]* for dis[a-z]* cont[a-z]*', '', text)


    if ngram_count <= 2:
        text = re.sub(r'(infectious disease)', '', text)
        text = re.sub(r'mathematical model?', '', text)
        text = re.sub(r'public health', '', text)
        text = re.sub(r'confidence intervals?', '', text)
        text = re.sub(r'transmission models?', '', text)
        text = re.sub(r'transmission dynam[a-z]*', '', text)
        text = re.sub(r'comp[a-z]* models?', '', text)
        text = re.sub(r'results? (show[a-z]*|sugg[a-z]*)', '', text)
        text = re.sub(r'attack rates?', '', text)
        text = re.sub(r'control strat[a-z]*', '', text)
        text = re.sub(r'incubation period[a-z]*', '', text)
        text = re.sub(r'vaccine eff[a-z]*', '', text)
        text = re.sub(r'qualityadjusted[a-z]*', '', text)
        text = re.sub(r'models? predic[a-z]*', '', text)
        text = re.sub(r'clin[a-z]* tria[a-z]*', '', text)
        text = re.sub(r'reproduct[a-z]* numbers?', '', text)
        text = re.sub(r'machine learn[a-z]*', '', text)
        text = re.sub(r'disease[a-z]* trans[a-z]*', '', text)
        text = re.sub(r'cohor[a-z]* stud[a-z]*', '', text)
        text = re.sub(r'vacc[a-z]* strat[a-z]*', '', text)
        text = re.sub(r'environmental health', '', text)
        text = re.sub(r'golbal health', '', text)
        text = re.sub(r'disease outbreak[a-z]*', '', text)
        text = re.sub(r'united states', '', text)
        text = re.sub(r'disease outbreak[a-z]*', '', text)
        text = re.sub(r'decision mak[a-z]*', '', text)
        text = re.sub(r'data interpretation', '', text)
        text = re.sub(r'model evaulation[a-z]*', '', text)
        text = re.sub(r'risk factor[a-z]*', '', text)
        text = re.sub(r'health planner[a-z]*', '', text)
        text = re.sub(r'paper compare[a-z]*', '', text)
        text = re.sub(r'health care[a-z]*', '', text)
        text = re.sub(r'contact tracing', '', text)
        text = re.sub(r'mathema[a-z]* mode', '', text)
        text = re.sub(r'public heal[a-z]*', '', text)
    

    if ngram_count <=1:
        text = re.sub(r'policy|policies', '', text)
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
        text = re.sub(r'illnes[a-z]*', '', text)
        text = re.sub(r'forecast[a-z]*', '', text)
        text = re.sub(r'although', '', text)
        text = re.sub(r'infection[a-z]*', '', text)
        text = re.sub(r'monitoring', '', text)
        text = re.sub(r'theoretical', '', text)
        text = re.sub(r'model[a-z]*', '', text)
        text = re.sub(r'disease[a-z]*', '', text)
        text = re.sub(r'communicable', '', text)
        text = re.sub(r'virus', '', text)
        text = re.sub(r'infection[a-z]*', '', text)
        text = re.sub(r'biological', '', text)
        text = re.sub(r'human[a-z]*', '', text)
        text = re.sub(r'pandemic[a-z]*', '', text)
        text = re.sub(r'paper[a-z]*', '', text)
        text = re.sub(r'probabilistic', '', text)
        text = re.sub(r'statu', '', text)
        text = re.sub(r'past', '', text)
        text = re.sub(r'stillhigh', '', text)
        text = re.sub(r'methods', '', text)
        text = re.sub(r'introduction', '', text)
        text = re.sub(r'million', '', text)
        text = re.sub(r'informed', '', text)
        text = re.sub(r'decades', '', text)
        text = re.sub(r'tignificancethis', '', text)
        text = re.sub(r'communicable', '', text)
        text = re.sub(r'disease[a-z]*', '', text)
        text = re.sub(r'incidence', '', text)
        text = re.sub(r'usa', '', text)
        text = re.sub(r'ensemble', '', text)
        text = re.sub(r'mortality', '', text)
        text = re.sub(r'probabil[a-z]*', '', text)
        text = re.sub(r'epidemi[a-z]*', '', text)
        text = re.sub(r'nan', '', text)
        text = re.sub(r'evaluat[a-z]*', '', text)
        text = re.sub(r'vaccin[a-z]*', '', text)
        text = re.sub(r'season[a-z]*', '', text)
        text = re.sub(r'decisi[a-z]*', '', text)
        text = re.sub(r'prediction[a-z]*', '', text)
        text = re.sub(r'expos[a-z]*', '', text)
        text = re.sub(r'outbreak[a-z]*', '', text)
        text = re.sub(r'data[a-z]*', '', text)
        text = re.sub(r'simulat[a-z]*', '', text)
        text = re.sub(r'middleincome[a-z]*', '', text)
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
        text = re.sub(r'determin[a-z]*', '', text)
        text = re.sub(r'united', '', text)
        text = re.sub(r'research[a-z]*', '', text)
        text = re.sub(r'agent', '', text)
        text = re.sub(r'experimen[a-z]*', '', text)
        text = re.sub(r'evidence', '', text)
        text = re.sub(r'health', '', text)
        text = re.sub(r'factors', '', text)
        text = re.sub(r'mathemati[a-z]*', '', text)
        text = re.sub(r'estima[a-z]*', '', text)
        text = re.sub(r'additional', '', text)
        text = re.sub(r'affect', '', text)
        text = re.sub(r'age', '', text)
        text = re.sub(r'approach', '', text)
        text = re.sub(r'assess', '', text)
        text = re.sub(r'associated', '', text)
        text = re.sub(r'association', '', text)
        text = re.sub(r'available', '', text)
        text = re.sub(r'based', '', text)
        text = re.sub(r'cause[d]', '', text)
        text = re.sub(r'change[a-z]*', '', text)
        text = re.sub(r'clinical', '', text)
        text = re.sub(r'collect[a-z]*', '', text)
        text = re.sub(r'combine[a-z]*', '', text)
        text = re.sub(r'conclusion[a-z]*', '', text)
        text = re.sub(r'condition[a-z]*', '', text)
        text = re.sub(r'conduct[a-z]*', '', text)
        text = re.sub(r'consider[a-z]*', '', text)
        text = re.sub(r'control[a-z]*', '', text)
        text = re.sub(r'current', '', text)
        text = re.sub(r'demonstrat[a-z]*', '', text)
        text = re.sub(r'design[a-z]*', '', text)
        text = re.sub(r'develop[a-z]*', '', text)
        text = re.sub(r'differen[a-z]*', '', text)
        text = re.sub(r'dynamic[a-z]*', '', text)
        text = re.sub(r'effect[a-z]*', '', text)
        text = re.sub(r'evalutat[a-z]*', '', text)
        text = re.sub(r'general[a-z]*', '', text)
        text = re.sub(r'group', '', text)
        text = re.sub(r'high[a-z]*', '', text)
        text = re.sub(r'however', '', text)
        text = re.sub(r'identif[a-z]*', '', text)
        text = re.sub(r'implement[a-z]*', '', text)
        text = re.sub(r'important', '', text)
        text = re.sub(r'improve[a-z]*', '', text)
        text = re.sub(r'includ[a-z]*', '', text)
        text = re.sub(r'increase[a-z]*', '', text)
        text = re.sub(r'indicate', '', text)
        text = re.sub(r'individual[a-z]*', '', text)
        text = re.sub(r'investigat[a-z]*', '', text)
        text = re.sub(r'large', '', text)
        text = re.sub(r'level', '', text)
        text = re.sub(r'likely', '', text)
        text = re.sub(r'limit[a-z]*', '', text)
        text = re.sub(r'many', '', text)
        text = re.sub(r'measure[a-z]*', '', text)
        text = re.sub(r'multiple', '', text)
        text = re.sub(r'need', '', text)
        text = re.sub(r'new', '', text)
        text = re.sub(r'number', '', text)
        text = re.sub(r'occur[a-z]*', '', text)
        text = re.sub(r'outcome[a-z]*', '', text)
        text = re.sub(r'parameter[a-z]*', '', text)
        text = re.sub(r'performance', '', text)
        text = re.sub(r'potential', '', text)
        text = re.sub(r'predict', '', text)
        text = re.sub(r'present', '', text)
        text = re.sub(r'provide[a-z]*', '', text)
        text = re.sub(r'range', '', text)
        text = re.sub(r'rate[a-z]*', '', text)
        text = re.sub(r'recent', '', text)
        text = re.sub(r'reduce', '', text)
        text = re.sub(r'region[a-z]*', '', text)
        text = re.sub(r'relative', '', text)
        text = re.sub(r'remain[a-z]*', '', text)
        text = re.sub(r'report[a-z]*', '', text)
        text = re.sub(r'response', '', text)
        text = re.sub(r'sample[a-z]*', '', text)
        text = re.sub(r'setting[a-z]*', '', text)
        text = re.sub(r'severe', '', text)
        text = re.sub(r'significant', '', text)
        text = re.sub(r'suggest[a-z]*', '', text)
        text = re.sub(r'time', '', text)
        text = re.sub(r'understand[a-z]*', '', text)
        text = re.sub(r'use', '', text)
        text = re.sub(r'using', '', text)
        text = re.sub(r'variable[a-z]*', '', text)
        text = re.sub(r'variation', '', text)
        text = re.sub(r'year[a-z]*', '', text)
        text = re.sub(r'absolute', '', text)
        text = re.sub(r'acute', '', text)
        text = re.sub(r'adult[a-z]*', '', text)
        text = re.sub(r'algorithm', '', text)
        text = re.sub(r'amplification', '', text)
        text = re.sub(r'animal', '', text)
        text = re.sub(r'autocorrelation', '', text)
        text = re.sub(r'based', '', text)
        text = re.sub(r'behavior', '', text)
        text = re.sub(r'big', '', text)
        text = re.sub(r'borne', '', text)
        text = re.sub(r'burden', '', text)
        text = re.sub(r'care', '', text)
        text = re.sub(r'cluster[a-z]*', '', text)
        text = re.sub(r'community', '', text)
        text = re.sub(r'conditonal', '', text)
        text = re.sub(r'demographic', '', text)
        text = re.sub(r'detection', '', text)
        text = re.sub(r'drug', '', text)
        text = re.sub(r'emerg[a-z]*', '', text)
        text = re.sub(r'function', '', text)
        text = re.sub(r'individualbased', '', text)
        text = re.sub(r'inference', '', text)
        text = re.sub(r'maximum', '', text)
        text = re.sub(r'mean', '', text)
        text = re.sub(r'minimum', '', text)
        text = re.sub(r'multivariate', '', text)
        text = re.sub(r'process[a-z]*', '', text)
        text = re.sub(r'quantitative', '', text)
        text = re.sub(r'randomize[a-z]*', '', text)
        text = re.sub(r'rate', '', text)
        text = re.sub(r'selection', '', text)
        text = re.sub(r'series', '', text)
        text = re.sub(r'targeted', '', text)
        text = re.sub(r'theory', '', text)
        text = re.sub(r'trial', '', text)
        text = re.sub(r'variant', '', text)
        text = re.sub(r'examin[a-z]*', '', text)
        text = re.sub(r'activit[a-z]*', '', text)
        text = re.sub(r'\baim[a-z]*', '', text)
        text = re.sub(r'consist[a-z]*', '', text)
        text = re.sub(r'countr[a-z]*', '', text)
        text = re.sub(r'\bday[a-z]*', '', text)
        text = re.sub(r'depend[a-z]*', '', text)
        text = re.sub(r'describ[a-z]*', '', text)
        text = re.sub(r'follow[a-z]*', '', text)
        text = re.sub(r'impact[a-z]*', '', text)
        text = re.sub(r'\bing\b', '', text)
        text = re.sub(r'\blow\b', '', text)
        text = re.sub(r'\blower\b', '', text)
        text = re.sub(r'\bkey\b', '', text)
        text = re.sub(r'major', '', text)
        text = re.sub(r'method[s]?', '', text)
        text = re.sub(r'network', '', text)
        text = re.sub(r'novel', '', text)
        text = re.sub(r'object[a-z]*', '', text)
        text = re.sub(r'overall', '', text)
        text = re.sub(r'participant[s]?', '', text)
        text = re.sub(r'perform[a-z]*', '', text)
        text = re.sub(r'period', '', text)
        text = re.sub(r'positive', '', text)
        text = re.sub(r'ratio', '', text)
        text = re.sub(r'relat[a-z]*', '', text)
        text = re.sub(r'require[a-z]*', '', text)
        text = re.sub(r'respective[a-z]*', '', text)
        text = re.sub(r'risk[s]?', '', text)
        text = re.sub(r'role[s]?', '', text)
        text = re.sub(r'several', '', text)
        text = re.sub(r'similar', '', text)
        text = re.sub(r'statistic[a-z]*', '', text)
        text = re.sub(r'structure', '', text)
        text = re.sub(r'substancial', '', text)
        text = re.sub(r'support', '', text)
        text = re.sub(r'type[s]?', '', text)
        text = re.sub(r'vary[a-z]*', '', text)

    # all
    text = re.sub(r',', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b[a-z]\b', '', text)
    text = re.sub(r'\b[a-z][a-z]\b', '', text)

    # add standalone numbers- not sure the correct regex for this
    return text

def get_field_value(val):
    text = ""
    if isinstance(val, pd.Series):
        if not isinstance(val.values[0], list) and not isinstance(val.values[0],
                    str) and math.isnan(val.values[0]):
            pass
        else:
            text = " ".join(val.values[0])
    if isinstance(val, list):
        text = " ".join([item for item in val])
    if isinstance(val, str):
        text = val
    
    return text


def inc_topic_count(text_dict, year, topic):
    if year in text_dict.keys():
        if topic not in text_dict[year].keys():
            text_dict[year][topic] = 0
    else:
        text_dict[year] = {topic: 0}
    
    text_dict[year][topic] += 1

    return text_dict

def post_process_text(field, text, title, ngram_count, do_remove_common, do_stemming):
    if field == Fields.ABSTRACT:
        text += title
        remove_common(text, ngram_count, do_remove_common)

    return remove_stop_words_and_do_stemming(text, ngram_count, do_stemming, do_remove_common)

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
                title = remove_common(row['title'].values[0], ngram_count, do_remove_common)
                for field in field_set:
                    abstract = row[field]
                    if isinstance(abstract, pd.Series):
                        if not isinstance(row[field].values[0], list) and not isinstance(row[field].values[0],
                                                                                         str) and math.isnan(
                            row[field].values[0]):
                            continue;
                        abstract = " ".join(row[field].values[0])
                    if isinstance(abstract, str):
                        abstract = remove_common(abstract, ngram_count, do_remove_common)

                    all_person_text += " " + abstract

        all_person_text = title + all_person_text
        all_person_text = remove_common(all_person_text, ngram_count, do_remove_common)
        person = people['name'][personIdx] + "#" + people['uri'][personIdx]
        people_list.append(person)
        list_of_words = remove_stop_words_and_do_stemming(all_person_text, do_stemming, do_remove_common)
        text_list.append(''.join(list_of_words))

    df = pd.DataFrame({
        'people': people_list,
        'text': text_list
    })
    return df


def remove_stop_words_and_do_stemming(unfiltered_text, ngram_count, do_stemming, do_remove_common):
    unfiltered_text = remove_common(unfiltered_text.translate(str.maketrans("", "", string.punctuation)),ngram_count,
                                    do_remove_common)
    word_tokens = wordpunct_tokenize(unfiltered_text.lower())

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

def make_list(obj):
    if isinstance(obj, pd.Series):
        list_obj = obj.tolist()
    elif isinstance(obj, str):
        list_obj = obj.split(" ")
    else:
        list_obj = obj
    return list_obj

def build_corpus_words_by_year(field, ngram_count, min_year, max_year, do_stemming, do_remove_common):
    papers = pd.read_json('data_sources/papers.json')
    text_dict = {}

    for paper_idx, row in papers.iterrows():
        if paper_idx==157:
            pass
        # if index > 100:
        #     dfs = {}
        #     for key, value in text_dict.items():
        #         dfs[key] = pd.DataFrame({'text': value})
        #     return dfs
        title = remove_common(row['title'], ngram_count, do_remove_common)
        # TODO: maybe try earliest/latest date, see which is most filled, etc.
        try:
            year = int(row['datePublished'][-4:])
        except TypeError:
            try:
                year = int(row['articleDate'][-4:])
            except TypeError:
                year = 1993
        if year < min_year or year > max_year:
            continue

        abstract = row[field.value]
        if field == Fields.MESH_TERM:
            if isinstance(abstract, float):
                continue
            else:
                for term in abstract:
                    text_dict = inc_topic_count(text_dict, year, term)
        else:
            text = get_field_value(abstract)
            if len(text) == 0:
                continue

            list_of_words = post_process_text(field, text, title, ngram_count, do_remove_common, do_stemming)

            if year in text_dict.keys():
                text_dict[year].append(''.join(list_of_words))
            else:
                text_dict[year] = [list_of_words]

    if field == Fields.MESH_TERM:
        return text_dict
    else:
        dfs = {}
        for key, value in text_dict.items():
            dfs[key] = pd.DataFrame({'text': value})
        return dfs

def get_papers_per_word(field, final_word_list, min_year, max_year, do_stemming, do_remove_common):
    papers = pd.read_json('data_sources/papers.json')
    paper_dict = {}

    search_year = 2012
    search_word = 'ed'
    filename = str(search_year) + '_w_' + search_word + '.txt'
    ## delete the file if it exists
    if os.path.exists(filename):
        os.remove(filename)

    for paper_idx, row in papers.iterrows():
        all_content_text_for_paper = ""
        ngram_count = 1
        title =  row['title']
        processed_title = remove_common(title, ngram_count, do_remove_common)
        uri = row['uri']
        # if index > 100:
        #     dfs = {}
        #     for key, value in text_dict.items():
        #         dfs[key] = pd.DataFrame({'text': value})
        #     return paper_dict

        try:
            year = int(row['datePublished'][-4:])
        except TypeError:
            try:
                year = int(row['articleDate'][-4:])
            except TypeError:
                year = 1993
        if year < min_year or year > max_year:
            continue

        field_content = row[field.value]
        if field == Fields.MESH_TERM:
            if isinstance(field_content, float):
                continue
            else:
                list_of_words = field_content
        else:
            text = get_field_value(field_content)
            if len(text) == 0:
                continue
            
            list_of_words = post_process_text(field, text, processed_title, ngram_count, do_remove_common, do_stemming)

        # Iterate through words for particular year
        keep_list = make_list(final_word_list.loc[final_word_list['date'] == year, 'topic'])
        word_list = make_list(list_of_words)

        for word in keep_list:
            if word in word_list:
                if year in paper_dict.keys():
                    if word in paper_dict[year].keys():
                        paper_dict[year][word].append({'title': title, 'uri': uri})
                    else:
                        paper_dict[year][word] = [{'title': title, 'uri': uri}]
                else:
                    paper_dict[year] = {}
                    paper_dict[year][word] = [{'title': title, 'uri': uri}]



        if search_year == year and search_word in word_list:
            with open(filename, 'a') as f:
                f.write(str(paper_idx) + ": " + list_of_words + '\n')

    return paper_dict
