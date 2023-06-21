`conda create --name topic-counter python=3.9 -y`

`conda activate topic-counter`

`conda config --add channels conda-forge`

`conda install scikit-learn pandas nltk -y`

`python topic_count.py`

To make adjustments to the generated files, edit the variables in the main() function of topic_count.py.

- do_all: Boolean which controls if you generate the full set of files or just one
- min_year, max_year: The minimum and maximum year of the date range you are looking at
- field: The single field you are generating for a non-complete run.
- ngram_count: The single ngram number you are generating for a non-complete run.