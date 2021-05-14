import gensim
from gensim import corpora
from nltk import RegexpTokenizer, PorterStemmer
from stop_words import get_stop_words
from nltk import pos_tag
import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Crossover, Mutations, Selection
from textblob import TextBlob
import re, json, os, glob, time


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


def main():
    start_time = time.time()
    doc_set = []
    filtered = []
    texts = []
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    dir_path = 'C:\\Users\\Dino\\Desktop\\User-Review-Clustering\\app_reviews'

    # input
    for filename in glob.glob(os.path.join(dir_path, '*.JSON')):
        with open(filename, 'r') as f:
            for element in f:   
                data = json.loads(element)
                doc_set.append(data['comment'])
            f.close()

    # input preprocessing        
    for doc in doc_set:

        correct = TextBlob(doc).correct()

        long_words = decontract(str(correct))

        tokens = tokenizer.tokenize(long_words)

        stopped_tokens = [w for w in tokens if w not in en_stop]

        stemmed_tokens = [p_stemmer.stem(w) for w in stopped_tokens]

        tagged = pos_tag(stemmed_tokens)

        filtered = [w[0] for w in tagged if  w[1] == 'NN' or w[1] == 'NNP' or w[1] == 'VB']

        final_tokens = list(dict.fromkeys(filtered))

        final_tokens_longer_than_3 = [t for t in final_tokens if len(t) >= 3]

        if len(final_tokens_longer_than_3) > 3:
            texts.append(final_tokens_longer_than_3)

    print(len(texts))

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # ga
    pop_size = 100
    crossover_prob = 0.6
    mutation_prob = 0.01


    def fitness(c):
        ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=int(c[0]),
                                                passes=int(c[1]), alpha=float(c[2]), eta=float(c[2]))
        coherence = gensim.models.ldamodel.CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence= 'c_v').get_coherence()
        return -coherence


    varbounds = np.array([[2, 20], [10, 100], [0, 1], [0, 1]])
    alg_param = {'max_num_iteration': 100,
                'population_size': 10,
                'mutation_probability': mutation_prob,
                'elit_ratio': 0.2,
                'crossover_probability': crossover_prob,
                'crossover_type': Crossover.arithmetic(),
                'mutation_type': Mutations.uniform_by_center(),
                'selection_type': Selection.roulette(),
                'max_iteration_without_improv': 10}

    model = ga(fitness, 4, function_timeout=60.0 , variable_type='real', variable_boundaries=varbounds, algorithm_parameters=alg_param)
    model.run()

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()