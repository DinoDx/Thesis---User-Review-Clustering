from gensim import corpora
import time, pickle
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
    for t in range(2, 10):
        lda_model = LdaModel(corpus, id2word=dictionary, num_topics=t)
        coherence = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence= 'c_v')
        score = coherence.get_coherence()
        tup = t, score
        results.append(tup)

    results = pd.DataFrame(results, columns=['topic', 'score'])
    s = pd.Series(results.score.values, index=results.topic.values)
    _ = s.plot()
    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
