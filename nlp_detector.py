import spacy

import pandas as pd

import re
from utils import build_corpus_words_only

searchTerms = {

               "anaplasma",
               "anaplasmosis",
               "anthrax",
               "argentine hemorrhagic fever",
               "avian (influenza|flu)",
               "babesia",
               "babesiosis"
               "bacillary dysentery",
               "bacillus anthracis",
               "bacterial vaginosis",
               "blinding trachoma",
               "bolivian hemorrhagic fever",
               "brazilian hemorrhagic fever",
               "bv",
               "c\.? diff",
               "campylobacter",
               "campylobacteriosis",
               "candida",
               "candidiasis",
               "cat scratch disease",
               "chancroid",
               "chickenpox",
               "chikungunya fever",
               "chikungunya",
               "chlamydia",
               "cholera",
               "cjd",
               "clostridioides difficile",
               "clostridium difficile",

               "coronavirus",
               "covid ?19",
               "creutzfeldt-jakob disease",
               "crimean-congo hemorrhagic fever",
               "(cryptosporidiosis|cryptosporidium( infection)?)",
               "csd",
               "cyclospora",
               "cyclosporiasis",
               "(cytomegalovirus( infection)?|cmv)",
               "cytomegalovirus",
               "dengue fever",
               "dengue",
               "dhf",
               "diphtheria",
               "dysentery",
               "(escherichia|e\.? coli) infection",
               "eastern equine encephalitis",
               "ebola",
               "ebv",
               "ehec",
               "ehrlichia",
               "ehrlichiosis",
               "enterohemorrhagic (e\.?|escherichia) coli",
               "enterotoxic e\.? coli",
               "epstein-barr virus infection",
               "escherichia",
               "etec",
               "fever and rash illnesses",
               "fungal infection",
               "gastroenteritis",
               "german measles",
               "giardia infection",
               "giardiasis",
               "glandular fever",
               "gonorrhea",
               "gonorrhoea",
               "hai",
               "hand, foot, and mouth disease",
               "hansen disease",
               "hantavirus infection",
               "hantavirus pulmonary syndrome",
               "healthcare acquired infection",
               "healthcare associated infection",
               "heartland virus",
               "hemolytic uremic syndrome",
               "hemorrhagic fever with renal syndrome",
               "hendra virus infection",
               "hep a",
               "hep b",
               "hep c",
               "hep e",
               "hepatitis a",
               "hepatitis b",
               "hepatitis c",
               "hepatitis e",
               "(herpes|herpes simplex virus)",
               "hfmd",
               "hfmd",
               "histoplasmosis",
               "hpv infection",
               "hsv",
               "(hiv|human immunodeficiency virus|aids|acquired immune deficiency syndrome)",
               "human papillomavirus infection",
               "hus",
               "(ili|influenza-like illness)",
               "infectious mononucleosis",
               "(influenza|flu)",
               "invasive candidiasis",
               "japanese encephalitis",
               "kyasanur forest disease",
               "lassa fever",
               "lassa hemorrhagic fever",
               "legionella",
               "legionellosis",
               "legionnaires' disease",
               "leishmania",
               "leishmaniasis",
               "leprosy",
               "leptospirosis",
               "listeriosis",
               "lockjaw",
               "lyme disease",
               "lymphocytic choriomeningitis",
               "malaria",
               "marburg virus disease",
               "measles",
               "meningitis",
               "meningococcal disease",
               "meningococcal",
               "meningococcemia",
               "(mers|middle east respiratory syndrome)",
               "methicillin-resistant staphylococcus aureus",
               "(monkeypox|mpox|monkey pox)",
               "mononucleosis",
               "(mrsa|methicillin-resistant staphylococcus aureus)",
               "mumps",
               "nipah virus infection",
               "norovirus",
               "norwegian scabies",
               "o157:h7",
               "omsk hemorrhagic fever",
               "pertussis",
               "plague",
               "plasmodium",
               "pneumonia",
               "polio",
               "poliomyelitis",
               "powassan virus",
               "q fever",
               "rabies",
               "rickettsia",
               "rickettsial",
               "rift valley fever",

               "(rmsf|rocky mountain spotted fever)",
               "rotavirus infection",
               "rotavirus",
               "rubella",
               "salmonella",
               "salmonellosis",
               "(sars|severe acute respiratory syndrome)",
               "scabies",
               "scrub typhus",
               "shiga toxin-producing (e\.?|escherichia) coli",
               "shigella",
               "shigellosis",
               "(smallpox|small pox)",
               "st\.? louis encephalitis",
               "staph infections",
               "staphylococcal infections",
               "staphylococcus",
               "stec",
               "streptococcal",
               "streptococcus",
               "syphilis",
               "(tb|tuberculosis)",
               "tetanus",
               "toxic shock syndrome",
               "toxo",
               "toxoplasma",
               "toxoplasmosis",
               "trachoma",
               "trich",
               "trichomonas",
               "trichomoniasis",
               "trichonella",
               "trichonellosis",
               "tss",
               "tularemia",
               "typhoid fever",
               "typhoid",
               "typhus fever",
               "vancomycin resistant staphylococcus aureus",
               "varicella",
               "vcjd",
               "venezuelan hemorrhagic fever",
               "visa",
               "vrsa",
               "west nile fever",
               "west nile neuroinvasive disease",
               "(wnv|west nile virus)",
               "western equine encephalitis",
               "whooping cough",
               "yeast infection",
               "yellow fever",
               "zika fever",
               "zika virus",
               "zoonoses",
               "zoonosis",
               "zoonotic infections",
               "zoonotic influenza"}


def do_nlp():
    med_transcript = pd.read_json('data/papers.json')
    med_transcript.info()
    med_transcript = med_transcript[med_transcript.index.notnull()]
    med_transcript.head()
    nlp = spacy.load("en_ner_bc5cdr_md")

    med_transcript_small = med_transcript.dropna()
    sample_transcription_list = (med_transcript_small['paperAbstract'].values.tolist())
    # sample_transcription = nlp("This paper is about influenza in people with asthma, and covid-19, and measles.")

    counts = {}
    i = 0
    for abs in sample_transcription_list:
        doc = nlp(abs)
        i += 1
        # if i > 100:
        #     break
        for ent in doc.ents:
            key = ent.text + " " + ent.label_
            if key in counts.keys():
                counts[key] += 1
            else:
                counts[key] = 1
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    f = open('counts.csv', 'w')
    for item in counts:
        if (item[1] > 5):
            f.write(item[0] + "," + str(item[1]) + "\n")
    f.close()

    exit(0
         )


def do_manual():
    documents = build_corpus_words_only(["pubmedKeywords", "meshTerms", "paperAbstract"], do_stemming=False,
                                     do_remove_common=False)
    counts = {}
    tpp_f = open('disease_document_pairs.csv', 'w')
    for searchTerm in searchTerms:
        print(searchTerm)
        for docText in documents["text"]:
            idx = re.search(r"\b" + r"{}".format(searchTerm.lower()) + r"s?\b", docText)
            if idx is not None:
                start_pos = max(0, idx.span()[0] - 20)
                end_pos = min(len(docText), idx.span()[1] + 20)
                tpp_f.write(searchTerm + "," + docText[start_pos:end_pos].strip() + "," + docText + "\n")

                if searchTerm in counts.keys():
                    counts[searchTerm] += 1
                else:
                    counts[searchTerm] = 1

    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    f = open('disease_counts.csv', 'w')
    for item in counts:
        # if (item[1] > 5):
        f.write(item[0] + "," + str(item[1]) + "\n")
    f.close()


if __name__ == '__main__':
    do_manual()
    exit(0)
