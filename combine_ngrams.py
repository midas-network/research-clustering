import pandas as pd

from fields import Fields


def combine(field):
    n1 = pd.read_csv("output/" + field.value + "-ngram_1-counts.csv")
    n2 = pd.read_csv("output/" + field.value + "-ngram_2-counts.csv")
    n3 = pd.read_csv("output/" + field.value + "-ngram_3-counts.csv")

    combined_df = pd.concat([n1, n2, n3])

    years = combined_df['date'].unique()

    terms = []
    for year in years:
        year_df = combined_df.loc[combined_df['date']==year].sort_values('count', ascending=False)
        top20 = year_df[0:19]['topic'].tolist()
        for term in top20:
            if term not in terms:
                terms.append(term)

    limited_df = combined_df[combined_df['topic'].isin(terms)].sort_values(by=['date', 'topic'], ascending=[True, True])
    return limited_df

def print_file(field, limited_df):
    combo_filename = field.value + '-combo-counts.csv'
    limited_df.to_csv('output/' + combo_filename, index = False)

def main():
    fields = [Fields.ABSTRACT, Fields.PUBMED_KEYWORD]
    for field in fields:
        limited_df = combine(field)
        print_file(field, limited_df)

if __name__ == "__main__":
    main()
    quit()

