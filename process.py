import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import numpy as np
import re

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def processText(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text) 
    text = re.sub(r'[^\w\s]', '', text)
    
    doc = nlp(text)
    tokens = []
    for token in doc:
        if (token.text not in stop_words and not token.is_punct
            and token.pos_ in ["ADJ", "ADV", "VERB", "NOUN"]):
            tokens.append(token.lemma_)
    
    return " ".join(tokens)


def featureExtraction(data, ngram_range):
    vectorizer = TfidfVectorizer(ngram_range = ngram_range)
    features = vectorizer.fit_transform(data['processed_text'])
    labels = data['sentiment']
    return features, labels, vectorizer

#def featureExtractionSvd(data, ngram_range):
#    vectorizer = TfidfVectorizer(ngram_range = ngram_range)
#    tfidf_features = vectorizer.fit_transform(data['processed_text'])
#    svd = TruncatedSVD(n_components=100)
#    features = svd.fit_transform(tfidf_features)
#    labels = data['sentiment']
#    return features, labels, vectorizer

