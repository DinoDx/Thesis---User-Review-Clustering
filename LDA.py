from gensim import corpora
import time, pickle, os, csv
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    start_time = time.time()
    file = open('preprocessing\\preprocessedData.txt', 'rb')
    texts = pickle.load(file)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    results = []

    # clustering
    lda_model = LdaModel(corpus, id2word=dictionary, num_topics=5)
    coherence = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence= 'c_v')
    score = coherence.get_coherence()

    # save the topics
    top_words_per_topic = []
    for t in range(lda_model.num_topics):
        top_words_per_topic.extend([(t, ) + x for x in lda_model.show_topic(t, topn = 10)])

    pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("output\\top_words_lda.csv")

    print("--- %s seconds ---" % (time.time() - start_time))
