#!/bin/env python

from glob import glob
from argparse import *
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

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

# This program receives the method extractor and the data directory as params
def main():
    parser = ArgumentParser(description = "This program extract text features from data in a given directory")

    parser.add_argument("data_directory", help = "Directory (whitout the last trailing slash) from which data will be read")

    parser.add_argument("-p", "--positive-label", default = "positive", help = "Label to be used in positive use cases")
    parser.add_argument("-n", "--negative-label", default = "negative", help = "Label to be used in negative use cases")

    parser.add_argument("-f", "--features", type = int, default = 10, help = "Number of features to extract")
    parser.add_argument("-g", "--n-grams", type = int, default = 2, help = "N-Grams max range to extract")
    parser.add_argument("-t", "--test-size", type = float, default = 0.10, help = "Test size in relation to total data")

    args = parser.parse_args()

    vectorizer = StemmedTfIdfVectorizer(input = 'filename', strip_accents = 'unicode',
                                        #norm = None, use_idf = False,
                                        analyzer = 'word',
                                        preprocessor = expand_contractions, stop_words = 'english',
                                        ngram_range = (1, args.n_grams))

    # load data
    positive_data  = glob("{}/pos/*.txt".format(args.data_directory))
    negative_data  = glob("{}/neg/*.txt".format(args.data_directory))
    total_raw_data = positive_data + negative_data
    # data length
    len_positive = len(positive_data)
    len_negative = len(negative_data)
    len_total    = len_positive + len_negative

    # example labels
    y = ([args.positive_label] * len_positive) + ([args.negative_label] * len_negative)

    # processed data
    data = vectorizer.fit_transform(total_raw_data, y)
    feature_names = np.array(vectorizer.get_feature_names())

    # build feature selector
    feature_selector = SelectKBest(chi2, k = args.features)
    clean_data       = feature_selector.fit_transform(data, y)

    # get best K features
    best_features = feature_names[feature_selector.get_support()].tolist()

    # print best features
    print("Best features:\n{}".format(', '.join(best_features)))

    # split data into train and test
    train_X, test_X, train_Y, test_Y = train_test_split(clean_data.toarray(), y,
                                                        test_size = args.test_size, random_state = 42)

    # Data set to data frame
    df = {}
    df['train'] = pd.DataFrame(train_X, columns = best_features)
    df['train'] = df['train'].assign(label = train_Y)

    df['test'] = pd.DataFrame(test_X, columns = best_features)
    df['test'] = df['test'].assign(label = test_Y)

    # save data
    for key in df.keys():
        file_name = "{}/{}-{}-{}-features.csv".format(args.data_directory, key, args.features, args.n_grams)
        print ("\n\nSaving dataset into '{}'".format(file_name))
        # saving...
        df[key].to_csv(file_name, index = False)


if __name__ == '__main__':
    main()

