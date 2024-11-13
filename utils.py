import json
import os
from enum import Enum

import math
import re
import string

from nltk import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords

import pandas as pd
import numpy as np

from UnstemMethod import UnstemMethod
from fields import Fields

STEMDICT = {}
stop_words = set(stopwords.words('english'))


def get_stemdict_cache():
    ##if the stemdict file exists
    if os.path.exists("cache/stemdict.json"):
        with open("cache/stemdict.json", "r") as f:
            return json.load(f)

def remove_common_no_ngram(text, bypass):
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
    text = re.sub(r',', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def remove_common(text, ngram_count, bypass):
    if not bypass:
        return text

    text = text.lower()
    # text = re.sub(r'published by.*on behalf of.*', '', text)
    text = re.sub(r'published by.*$', '', text)
    text = re.sub(r'(©|copyright|funding|this article is protected).*$', '', text)
    text = re.sub(r'\b[a-z]\b', '', text)
    text = re.sub(r'\b[a-z][a-z]\b', '', text)
    filter_words_3gram = []
    if ngram_count <= 3:
        for string in filter_words_3gram:
            text = re.sub(r'' + string, '', text)
        text = re.sub(r'\beffective reproduction number', '', text)
        text = re.sub(r'\bpublic health interventions?', '', text)
        text = re.sub(r'\bformula: see text', '', text)
        text = re.sub(r'\bbasic reproduction numbers?', '', text)
        text = re.sub(r'\bworld health org[a-z]*', '', text)
        text = re.sub(r'\bcenters for disease control and prevention', '', text)
        text = re.sub(r'\bsocial distanc[a-z]* (meas[a-z]*)?', '', text)
        text = re.sub(r'\breceiv[a-z]* operat[a-z]* charact[a-z]* curv[a-z]*', '', text)
        text = re.sub(r'\bnational institutes of health', '', text)
        text = re.sub(r'\bcente[a-z]* for dis[a-z]* cont[a-z]*', '', text)


    filter_words_2gram = []
    if ngram_count <= 2:
        for string in filter_words_2gram:
            text = re.sub(r'\b' + string + '[a-z]*\b', '', text)
        text = re.sub(r'\b(infectious disease)', '', text)
        text = re.sub(r'\bmathematical model?', '', text)
        text = re.sub(r'\bpublic health', '', text)
        text = re.sub(r'\bconfidence intervals?', '', text)
        text = re.sub(r'\btransmission models?', '', text)
        text = re.sub(r'\btransmission dynam[a-z]*', '', text)
        text = re.sub(r'\bcomp[a-z]* models?', '', text)
        text = re.sub(r'\bresults? (show[a-z]*|sugg[a-z]*)', '', text)
        text = re.sub(r'\battack rates?', '', text)
        text = re.sub(r'\bcontrol strat[a-z]*', '', text)
        text = re.sub(r'\bincubation period[a-z]*', '', text)
        text = re.sub(r'\bvaccine eff[a-z]*', '', text)
        text = re.sub(r'\bqualityadjusted[a-z]*', '', text)
        text = re.sub(r'\bmodels? predic[a-z]*', '', text)
        text = re.sub(r'\bclin[a-z]* tria[a-z]*', '', text)
        text = re.sub(r'\breproduct[a-z]* numbers?', '', text)
        text = re.sub(r'\bmachine learn[a-z]*', '', text)
        text = re.sub(r'\bdisease[a-z]* trans[a-z]*', '', text)
        text = re.sub(r'\bcohor[a-z]* stud[a-z]*', '', text)
        text = re.sub(r'\bvacc[a-z]* strat[a-z]*', '', text)
        text = re.sub(r'\benvironmental health', '', text)
        text = re.sub(r'\bgolbal health', '', text)
        text = re.sub(r'\bdisease outbreak[a-z]*', '', text)
        text = re.sub(r'\bunited states', '', text)
        text = re.sub(r'\bdisease outbreak[a-z]*', '', text)
        text = re.sub(r'\bdecision mak[a-z]*', '', text)
        text = re.sub(r'\bdata interpretation', '', text)
        text = re.sub(r'\bmodel evaulation[a-z]*', '', text)
        text = re.sub(r'\brisk factor[a-z]*', '', text)
        text = re.sub(r'\bhealth planner[a-z]*', '', text)
        text = re.sub(r'\bpaper compare[a-z]*', '', text)
        text = re.sub(r'\bhealth care[a-z]*', '', text)
        text = re.sub(r'\bcontact tracing', '', text)
        text = re.sub(r'\bmathema[a-z]* mode[a-z]*', '', text)
        text = re.sub(r'\bpublic heal[a-z]*', '', text)
    

    filter_words_1gram = ['polic', 'population', 'common', 'case', 'spread', 'infectious',
                          'computat', 'susceptib', 'sensitivi', 'transmitt', 'fatality',
                          'vector', 'strateg', 'observ', 'specific', 'illnes', 'forecast',
                          'although', 'infection', 'monitoring', 'theoretical', 'model',
                          'disease', 'communicable', 'virus', 'infection', 'biological',
                          'human', 'pandemic', 'paper', 'probabilistic', 'statu', 'past',
                          'stillhigh', 'methods', 'introduction', 'million', 'informed',
                          'decades', 'tignificancethis', 'communicable', 'disease', 'incidence',
                          'usa', 'ensemble', 'mortality', 'probabil', 'epidemi', 'nan',
                          'evaluat', 'vaccin', 'season', 'decisi', 'prediction', 'expos',
                          'outbreak', 'data', 'simulat', 'middleincome', 'qualityadj', 'review',
                          'patient', 'information', 'surv', 'result', 'estimate', 'algorithim',
                          'stochastic', 'processes', 'intervent', 'theoretical', 'compare',
                          'studies', 'computer', 'analysis', 'healthcare', 'testing', 'vaccine',
                          'test', 'stud', 'preval', 'mitigati', 'vaccine', 'crosssectional',
                          'distribution', 'significancethis', 'evaluation', 'assessment',
                          'background', 'mitigate', 'trasnsmiss', 'pattern', 'spatial',
                          'mortality', 'agentbased', 'emergency', 'determin', 'united',
                          'research', 'agent', 'experimen', 'evidence', 'health', 'factors',
                          'mathemati', 'estima', 'additional', 'affect', 'age', 'approach',
                          'assess', 'associated', 'association', 'available', 'based', 'cause',
                          'change', 'clinical', 'collect', 'combine', 'conclusion', 'condition',
                          'conduct', 'consider', 'control', 'current', 'demonstrat', 'design',
                          'develop', 'differen', 'dynamic', 'effect', 'evalutat', 'general',
                          'give', 'group', 'high', 'however', 'identif', 'implement', 'important',
                          'improve', 'includ', 'increase', 'indicate', 'individual', 'inform',
                          'initial', 'insight', 'interpret', 'investigat', 'involve', 'large',
                          'level', 'likely', 'limit', 'many', 'measure', 'multiple', 'need', 'new',
                          'number', 'occur', 'outcome', 'parameter', 'performance', 'potential',
                          'predict', 'present', 'provide', 'range', 'rate', 'recent', 'reduce',
                          'region', 'relative', 'remain', 'report', 'response', 'reveal', 'sample',
                          'setting', 'set', 'severe', 'significant', 'suggest', 'time', 'understand',
                          'use', 'using', 'variable', 'variation', 'year', 'absolute', 'acute',
                          'adult', 'algorithm', 'amplification', 'animal', 'autocorrelation', 'based',
                          'behavior', 'big', 'borne', 'burden', 'care', 'cluster', 'community',
                          'conditonal', 'demographic', 'detection', 'drug', 'emerg', 'function',
                          'individualbased', 'inference', 'maximum', 'mean', 'minimum', 'multivariate',
                          'process', 'quantitative', 'randomize', 'rate', 'selection', 'series',
                          'targeted', 'theory', 'trial', 'variant', 'examin', 'activit', 'aim',
                          'consist', 'countr', 'day', 'depend', 'describ', 'follow', 'impact', 'low',
                          'lower', 'key', 'major', 'method', 'network', 'novel', 'object', 'overall',
                          'participant', 'perform', 'period', 'positive', 'ratio', 'relat', 'require',
                          'respective', 'risk', 'role', 'several', 'similar', 'statistic', 'structure',
                          'substancial', 'support', 'type', 'vary', 'account', 'allow', 'analys',
                          'analyz', 'appli', 'apply', 'area', 'better', 'challeng', 'character',
                          'child', 'compar', 'complex', 'confirm', 'contact', 'continu', 'contribut',
                          'cost', 'death', 'decreas', 'detect', 'direct', 'discuss', 'early', 'effort',
                          'exist', 'explor', 'framework', 'future', 'global', 'great', 'help', 'increas',
                          'infect', 'interact', 'know', 'local', 'mechanism', 'month', 'national',
                          'obtain', 'optimal', 'possib', 'primary', 'program', 'proportion', 'propos',
                          'scale', 'single', 'size', 'small', 'source', 'state', 'substantial', 'target',
                          'thus', 'tool', 'total', 'value', 'wide', 'work', 'world', 'working', 'publish',
                          'explain', 'correlate', 'annual', 'address', 'average', 'event', 'generate',
                          'importan', 'previous', 'represent']
    if ngram_count <=1:
        for string in filter_words_1gram:
           text = re.sub(r'\b' + string + '[a-z]*', '', text)
        
        text = re.sub(r'\b[0-9]*\b', '', text)

    # all
    text = re.sub(r'\b,', ' ', text)
    text = re.sub(r'\b\s+', ' ', text)
    text = re.sub(r'\b[a-z]\b', '', text)
    text = re.sub(r'\b[a-z][a-z]\b', '', text)

    return text

def filter_mesh_terms(data):
    filter_terms = ['Asolescent', 'Adult', 'Aged', 'Animals', 'Child', 'Female',
                    'Humans', 'Infant', 'Male', 'Middle Aged', 'Young Adult',
                    'Adolescent', 'Child, Preschool', 'Computer Simulation',
                    'Incidence', 'Prevalence', 'Aged, 80 and over']

    
    for term in filter_terms:
        if term in data.keys():
            del data[term]

    return data

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
    with open("output/clusters/people-with-clusters-" + title + ".json", "w") as outfile:
        json.dump(bins, outfile)

    with open("output/clusters/cluster-info-" + title + ".json", "w") as outfile2:
        json.dump(features, outfile2)

def word_is_present(word, list_of_words, ngram_count):
    if ngram_count == 1:
        word_list = make_list(list_of_words)
        if word in word_list:
            return True
        else:
            return False
    else:
        if word in list_of_words:
            return True
        else:
            return False

def remove_stop_words_and_do_stemming_no_ngrams(unfiltered_text, do_stemming, do_remove_common):
    unfiltered_text = remove_common_no_ngram(unfiltered_text.translate(str.maketrans("", "", string.punctuation)),
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

    ##write stemdict to disk
    with open('cache/stemdict.json', 'w') as fp:
        json.dump(STEMDICT, fp)

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
        if paper_idx == 157:
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

def process_text(text, title, field, do_remove_common):
    if field in [Fields.ABSTRACT]:
        text = remove_common_no_ngram(text + title, do_remove_common)
    return text

def build_corpus(field_set, do_stemming, do_remove_common, want_cache=False):
    if want_cache and os.path.exists('data_sources/people_text.json'):
        return pd.read_json('data_sources/people_text.json')
    else:
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
                    title = remove_common_no_ngram(row['title'].values[0], do_remove_common)
                    for field in field_set:
                        field_value = row[field]
                        if isinstance(field_value, pd.Series):
                            if not isinstance(row[field].values[0], list) and not isinstance(row[field].values[0],
                                                                                             str) and math.isnan(
                                row[field].values[0]):
                                continue;
                            field_value = row[field].values[0]
                        if isinstance(field_value, str):
                            field_value = remove_common_no_ngram(field_value, do_remove_common)
                        if isinstance(field_value, list):
                            field_value = " ".join(field_value)

                        all_person_text += " " + field_value

            all_person_text = process_text(all_person_text, title, Fields.ABSTRACT, do_remove_common)
            person = people['name'][personIdx] + "#" + people['uri'][personIdx]
            people_list.append(person)
            list_of_words = remove_stop_words_and_do_stemming_no_ngrams(all_person_text, do_stemming, do_remove_common)
            text_list.append(''.join(list_of_words))

        df = pd.DataFrame({
            'people': people_list,
            'text': text_list
        })

        #save dataframe to file
        df.to_json('data_sources/people_text.json')
        return df

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

def get_papers_per_word(field, ngram_count, final_word_list, min_year, max_year, do_stemming, do_remove_common):
    papers = pd.read_json('data_sources/papers.json')
    paper_dict = {}

    search_year = 2021
    search_word = 'oxford'
    filename = str(search_year) + '_w_' + search_word + '.txt'
    ## delete the file if it exists
    if os.path.exists(filename):
        os.remove(filename)

    words_per_year = {}
    for year in range(min_year,max_year+1):
        date = str(year) + '/1/1'
        words_per_year[date] = make_list(final_word_list.loc[np.logical_and(final_word_list['date'] == date, final_word_list['count'] != 0), 'topic'])

    for paper_idx, row in papers.iterrows():
        all_content_text_for_paper = ""
        # ngram_count = 1
        title =  row['title']
        processed_title = remove_common(title, ngram_count, do_remove_common)
        uri = row['uri']
        abstract = row['paperAbstract']
        if type(abstract) != str:
            abstract = ''
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

        for word in words_per_year[str(year)+'/1/1']:
            if word_is_present(word, list_of_words, ngram_count):
                if year in paper_dict.keys():
                    if word in paper_dict[year].keys():
                        paper_dict[year][word].append({'title': title, 'uri': uri, 'abstract': abstract})
                    else:
                        paper_dict[year][word] = [{'title': title, 'uri': uri, 'abstract': abstract}]
                else:
                    paper_dict[year] = {}
                    paper_dict[year][word] = [{'title': title, 'uri': uri, 'abstract': abstract}]



        # if search_year == year and search_word in list_of_words:
            # with open(filename, 'a') as f:
            #     f.write(str(paper_idx) + ": " + list_of_words + '\n')

    return paper_dict
