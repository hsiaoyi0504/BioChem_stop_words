import gzip
import shutil
import re
from collections import Counter
import xml.etree.ElementTree as ET
from os import mkdir, remove
from os.path import dirname, abspath, exists, join
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from functools import lru_cache
from six.moves.urllib import request


PROJECT_ROOT = dirname(abspath(__file__))

stop_words = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()
lemmatize = lru_cache(maxsize=50000)(wordnet_lemmatizer.lemmatize)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


data_path = join(PROJECT_ROOT, 'data')
if not exists(data_path):
    mkdir(data_path)

word_counts = Counter()
for i in range(1, 928):
    num = str(i).zfill(4)
    print('Downloading pubmed18n{}.xml.gz ...'.format(num))
    request.urlretrieve(
        'ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed18n{}.xml.gz'.format(num),
        join(data_path, 'pubmed18n{}.xml.gz'.format(num)))
    print('Extracting pubmed18n{}.xml.gz ...'.format(num))
    file_name_in = join(data_path, 'pubmed18n{}.xml.gz'.format(num))
    file_name_out = join(data_path, 'pubmed18n{}.xml'.format(num))
    with gzip.open(file_name_in, 'rb') as f_in, open(file_name_out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    tree = ET.parse(file_name_out)
    root = tree.getroot()
    for article in root.findall('PubmedArticle'):
        try:
            abstract = article.find('MedlineCitation').find('Article').find('Abstract').find('AbstractText').text
            try:
                sentences = sent_tokenize(abstract)
            except TypeError:
                continue
            for s in sentences:
                temp = s.sub(r'\d+', '', s)
                text = word_tokenize(temp)
                tagged_text = pos_tag(text)
                temp_words = []
                for t, tag in tagged_text:
                    temp_words.append(lemmatize(t.lower(), get_wordnet_pos(tag)))
                word_counts.update(temp_words)
        except AttributeError:
            continue
    print('Finished prcessing the pubmed18n{}'.format(num))
    remove(file_name_in)
    remove(file_name_out)

for s in stop_words + [',', '.', '?', ';', '/', '\\', '!', '*', '(', ')', '|', '{', '}', '[', ']', '-', '>', '<', "'", '%', '=', ':', '+/-', "''", '``', '+', '--']:
    if s in word_counts:
        del word_counts[s]

with open('stop_words_1000.csv', 'w') as f:
    for w, count in word_counts.most_common(1000):
        f.write(w + ',' + str(count) + '\n')
