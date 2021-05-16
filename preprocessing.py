from stop_words import get_stop_words
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
import json, os, glob, time, re, pickle
from multiprocessing import Pool 
from multiprocessing import cpu_count

start_time = time.time()
doc_set = []
texts = []
tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()
dir_path = 'C:\\Users\\Dino\\Desktop\\User-Review-Clustering\\app_reviews'


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


def preprocess(doc):

    correct = TextBlob(doc).correct()

    long_words = decontract(str(correct))

    tokens = tokenizer.tokenize(long_words)

    stopped_tokens = [w for w in tokens if w not in en_stop]

    stemmed_tokens = [p_stemmer.stem(w) for w in stopped_tokens]

    tagged = pos_tag(stemmed_tokens)

    filtered = [w[0] for w in tagged if w[1] ==
                    'NN' or w[1] == 'NNP' or w[1] == 'VB']

    final_tokens = list(dict.fromkeys(filtered))

    final_tokens_longer_than_3 = [t for t in final_tokens if len(t) >= 3]

    if len(final_tokens_longer_than_3) > 3:
        global texts
        texts.append(final_tokens_longer_than_3)


if __name__ == '__main__':
    for file in glob.glob(os.path.join(dir_path, '*.json')):
        with open(file, 'r') as f:
            for element in f:
                data = json.loads(element)
                doc_set.append(data['comment'])
            f.close()

    pool = Pool(cpu_count()-1)
    pool.map(preprocess, doc_set)
    pool.close()
    pool.join()
    file = open('preprocessedData.txt', 'wb')
    print(texts)
    pickle.dump(texts, file)

    file.close()

    print("--- %s seconds ---" % (time.time() - start_time))

