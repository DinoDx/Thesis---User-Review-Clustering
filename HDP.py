from re import template
from gensim.models import coherencemodel

import numpy as np
import HDPUtils, pickle, time
import tomotopy as tp
from gensim import corpora
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    start_time = time.time()
    file = open('preprocessedData.txt', 'rb')
    texts = pickle.load(file)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    results = []

    #clustering
    for t in range (10, 100, 10):
        hdp_model = tp.HDPModel(tw= tp.TermWeight.ONE, initial_k=2)
        hdp_model_trained = HDPUtils.train_HDPmodel(hdp=hdp_model, word_list=texts, mcmc_iter=t)
        topics = HDPUtils.get_hdp_topics(hdp_model_trained, top_n=10)
        score = HDPUtils.eval_coherence(topics_dict=topics, word_list=texts)
        print("with {} iterations score : {} with {} topics".format(t, score, hdp_model_trained.live_k))
        tup = t, score
        results.append(tup)

    results = pd.DataFrame(results, columns=['iteration', 'score'])
    s = pd.Series(results.score.values, index=results.iteration.values)
    _ = s.plot()
    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
