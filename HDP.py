import re
import gensim
import pprint
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from textblob import TextBlob
import spacy
import json
import os
import glob
import time

start_time = time.time()

doc_set = []
filtered = []
texts = []
tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()
spa = spacy.load("en_core_web_sm")
dir_path = 'C:\\Users\\Dino\\Desktop\\User-Review-Clustering\\app_reviews'

for filename in glob.glob(os.path.join(dir_path, '*.JSON')):
  with open(filename, 'r') as f:
        for element in f:   
            data = json.loads(element)
            doc_set.append(data['comment'])
        f.close()


def decontract(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


for doc in doc_set:
    raw = doc.lower()

    correct = TextBlob(raw).correct()

    long_words = decontract(str(correct))

    tagged = spa(long_words)

    for w in tagged:
        if w.pos_ == 'NOUN' or w.pos_ == 'VERB':
            filtered.append(w)

    singular = TextBlob(str(filtered)).words.singularize()

    tokens = tokenizer.tokenize(str(singular))

    stopped_tokens = [doc for doc in tokens if doc not in en_stop]

    stemmed_tokens = [p_stemmer.stem(doc) for doc in stopped_tokens]

    final_tokens = list(dict.fromkeys(stemmed_tokens))

    for t in final_tokens:
        if len(t) < 3:
            stemmed_tokens.remove(t)

    if len(final_tokens) > 3:
        texts.append(stemmed_tokens)

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
hdp_model = gensim.models.hdpmodel.HdpModel(corpus, dictionary)

pprint.pprint(hdp_model.print_topics())


coherence = gensim.models.coherencemodel.CoherenceModel(hdp_model, corpus=corpus, coherence='u_mass')
print(coherence.get_coherence())
print("--- %s seconds ---" % (time.time() - start_time))