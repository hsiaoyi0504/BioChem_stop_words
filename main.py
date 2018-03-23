from glob import glob
import xml.etree.ElementTree as ET
from os.path import dirname, abspath, join
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from functools import lru_cache

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

word_counts = Counter()
xml_files = glob(join(PROJECT_ROOT, 'data', '*.xml'))
for f in xml_files:
    tree = ET.parse(f)
    root = tree.getroot()
    for article in root.findall('PubmedArticle'):
        try:
            abstract = article.find('MedlineCitation').find('Article').find('Abstract').find('AbstractText').text
            try:
                sentences = sent_tokenize(abstract)
            except TypeError:
                continue
            for s in sentences:
                text = word_tokenize(s)
                tagged_text = pos_tag(text)
                temp_words = []
                for t, tag in tagged_text:
                    temp_words.append(lemmatize(t.lower(), get_wordnet_pos(tag)))
                word_counts.update(temp_words)
        except AttributeError:
            continue


for s in stop_words + [',', '.', '?', ';', '/', '\\', '!', '*', '(', ')', '|', '{', '}', '[', ']', '-', '>', '<', "'"]:
    if s in word_counts:
        del word_counts[s]

with open('stop_words_1000.csv', 'w') as f:
    for w, count in word_counts.most_common(1000):
        f.wrte(w + ',' + str(count) + '\n' )

