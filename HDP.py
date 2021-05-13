import gensim
from gensim.models import hdpmodel
from stop_words import get_stop_words
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora
from textblob import TextBlob
import re, json, pprint, os, glob, time
import pandas as pd

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

    #input
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

    #clustering
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    hdp_model = gensim.models.hdpmodel.HdpModel(corpus, dictionary)
    print(hdp_model.show_topics())

    coherence = gensim.models.coherencemodel.CoherenceModel(model=hdp_model, texts=texts, dictionary=dictionary, coherence= 'c_v')
    print(coherence.get_coherence())
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
