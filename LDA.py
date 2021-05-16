from gensim import corpora
import time, pyLDAvis.gensim_models, pyLDAvis, pickle, gensim

if __name__ == "__main__":
    start_time = time.time()
    file = open('preprocessedData.txt', 'rb')
    texts = pickle.load(file)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # clustering
    ldamodel = gensim.models.ldamodel.LdaModel(corpus= corpus, num_topics=4, id2word=dictionary, passes=100)
    coherence = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence= 'c_v')
    print(coherence.get_coherence())

    vis = pyLDAvis.gensim_models.prepare(ldamodel, corpus= corpus, dictionary=dictionary)
    pyLDAvis.save_html(vis, 'lda.html')

    print("--- %s seconds ---" % (time.time() - start_time))
