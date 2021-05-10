import re
import gensim
import pprint
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from textblob import TextBlob
import spacy

# some fake documents
doc_a = "Broccoli is good to eat. My brother likes to eat good broccoli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that broccoli is good for your health."

# put documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]
filtered = []
texts = []
tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()
spa = spacy.load("en_core_web_sm")


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


for doc in doc_set:
    raw = doc.lower()

    correct = TextBlob(raw).correct()

    long = decontract(str(correct))

    tagged = spa(doc)

    for w in tagged:
        if w.pos_ == 'NOUN' or w.pos_ == 'VERB':
            filtered.append(w)

    singular = TextBlob(str(filtered)).words.singularize()

    tokens = tokenizer.tokenize(str(singular))

    stopped_tokens = [doc for doc in tokens if doc not in en_stop]

    stemmed_tokens = [p_stemmer.stem(doc) for doc in stopped_tokens]

    final_tokens = list(dict.fromkeys(stemmed_tokens))

    for t in final_tokens:
        if len(t) < 3:
            stemmed_tokens.remove(t)

    if len(final_tokens) > 3:
        texts.append(stemmed_tokens)

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
hdp_model = gensim.models.hdpmodel.HdpModel(corpus, dictionary)

pprint.pprint(hdp_model.print_topics())


coherence = gensim.models.coherencemodel.CoherenceModel(hdp_model, corpus=corpus, coherence='u_mass')
print(coherence.get_coherence())