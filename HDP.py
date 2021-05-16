from gensim import corpora
import time, pyLDAvis.gensim_models, pyLDAvis, gensim, pickle

def main():
    start_time = time.time()
    file = open('preprocessedData.txt', 'rb')
    texts = pickle.load(file)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    #clustering
    hdp_model = gensim.models.hdpmodel.HdpModel(corpus, dictionary, T= 20)
    coherence = gensim.models.coherencemodel.CoherenceModel(model=hdp_model, texts=texts, dictionary=dictionary, coherence= 'c_v')
    print(coherence.get_coherence())

    vis = pyLDAvis.gensim_models.prepare(hdp_model, corpus= corpus, dictionary=dictionary)
    pyLDAvis.save_html(vis, 'hdp.html')

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
