conda create --name topic-counter python=3.9 -y
conda activate topic-counter
conda config --add channels conda-forge
conda install scikit-learn pandas nltk -y
python topic_count.py
