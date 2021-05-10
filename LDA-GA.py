import gensim
from gensim import corpora
from nltk import RegexpTokenizer, PorterStemmer
from stop_words import get_stop_words
import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Crossover, Mutations, Selection
from textblob import TextBlob
import re
import spacy
import json
import os
import glob

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
    print('corrected')

    long_words = decontract(str(correct))
    print('longed')

    tagged = spa(long_words)
    print('tagged')

    for w in tagged:
        if w.pos_ == 'NOUN' or w.pos_ == 'VERB':
            filtered.append(w)

    print('filtered')

    singular = TextBlob(str(filtered)).words.singularize()
    print('singularized')

    tokens = tokenizer.tokenize(str(singular))
    print('tokenized')

    stopped_tokens = [doc for doc in tokens if doc not in en_stop]
    print('stop removed')

    stemmed_tokens = [p_stemmer.stem(doc) for doc in stopped_tokens]
    print('stemmed')

    final_tokens = list(dict.fromkeys(stemmed_tokens))

    for t in final_tokens:
        if len(t) < 3:
            stemmed_tokens.remove(t)

    if len(final_tokens) > 3:
        texts.append(stemmed_tokens)

print('preprocessing is finished')

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# ga
population = []
pop_size = 100
crossover_prob = 0.6
mutation_prob = 0.01


def fitness(c):
    ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=int(c[0]),
                                               passes=int(c[1]), alpha=float(c[2]),
                                               eta=float(c[2]))
    cm = gensim.models.ldamodel.CoherenceModel(model=ldamodel, corpus=corpus, coherence='u_mass')
    coherence = cm.get_coherence()
    print(coherence)
    return -coherence


vartypes = np.array(['int', 'int', 'real', 'real'])
varbounds = np.array([[2, 10], [10, 20], [0, 1], [0, 1]])
alg_param = {'max_num_iteration': 100,
             'population_size': pop_size,
             'mutation_probability': mutation_prob,
             'elit_ratio': 0.02,
             'crossover_probability': crossover_prob,
             'parents_portion': 0.02,
             'crossover_type': Crossover.arithmetic(),
             'mutation_type': Mutations.uniform_by_center(),
             'selection_type': Selection.roulette(),
             'max_iteration_without_improv': 10}

print('premodel')
model = ga(fitness, 4, variable_type_mixed=vartypes, variable_boundaries=varbounds, algorithm_parameters=alg_param)
print('postmodel, prerun')
model.run(no_plot=True)
model.plot_results()
