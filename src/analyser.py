#!/bin/env python

from argparse import *
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.externals import joblib

stemmer = PorterStemmer()

class StemmedTfIdfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfIdfVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

# Extracted and adapted from: https://github.com/mkfs/misc-text-mining/blob/master/R/wordcloud.R
def expand_contractions(doc):
    # "won't" is a special case as it does not expand to "wo not"
    doc = doc.replace("won't", "will not")
    doc = doc.replace("n't", " not")
    doc = doc.replace("'ll", " will")
    doc = doc.replace("'re", " are")
    doc = doc.replace("'ve", " have")
    doc = doc.replace("'m", " am")
    # 's could be 'is' or could be possessive: it has no expansion
    doc = doc.replace("'s", "")
    return doc


def main():
    parser = ArgumentParser(description = "This program evaluates a movie review based on the 200 feature svm extractor")
    parser.add_argument("review", help = "Movie review in text format")

    args = parser.parse_args()

    # load tf-idf vectorizer
    vectorizer = joblib.load('data/vectorizer.pkl')
    vectorizer.set_params(input = 'content')

    # process data
    data = vectorizer.transform([args.review])

    # load feature selector
    feature_selector = joblib.load('data/selector.pkl')
    clean_data       = feature_selector.transform(data)

    # load SVM
    svm_classifier = joblib.load('data/classifier.pkl')
    print("Label: {}".format(svm_classifier.predict(clean_data)[0]))

if __name__ == '__main__':
    main()

