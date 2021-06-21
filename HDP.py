import  pickle, time
import tomotopy as tp
from gensim import corpora
import pandas as pd
import matplotlib.pyplot as plt 
import HDPUtils

if __name__ == "__main__":
    start_time = time.time()
    file = open('preprocessing\\preprocessedData.txt', 'rb')
    texts = pickle.load(file)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    results = []

    #clustering
    hdp_model = tp.HDPModel(tw= tp.TermWeight.ONE,min_df = 5,initial_k=2)
    hdp_model_trained = HDPUtils.train_HDPmodel(hdp=hdp_model, word_list=texts, mcmc_iter=90)
    topics = HDPUtils.get_hdp_topics(hdp_model_trained, top_n=10)
    score = HDPUtils.eval_coherence(topics_dict=topics, word_list=texts)

    top_words_per_topic = []
    for t in topics:
        for i in range (0, 10):
            top_words_per_topic.extend([(t, ) + topics[t][i]])

    pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("output\\top_words_hdp.csv")

    print("--- %s seconds ---" % (time.time() - start_time))
