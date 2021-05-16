from gensim import corpora
from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Crossover, Mutations, Selection
import time,pickle, gensim, numpy as np

def main():
    start_time = time.time()
    file = open('preprocessedData.txt', 'rb')
    texts = pickle.load(file)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # ga
    pop_size = 100
    crossover_prob = 0.6
    mutation_prob = 0.01


    def fitness(c):
        ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=int(c[0]),
                                                passes=int(c[1]), alpha=float(c[2]), eta=float(c[2]))
        coherence = gensim.models.ldamodel.CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence= 'u_mass').get_coherence()
        return abs(coherence)


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
    model.run(set_function=ga.set_function_multiprocess(fitness, n_jobs=5))

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()