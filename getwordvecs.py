import numpy as np
from nltk import word_tokenize
from os import listdir
from os.path import isfile, join, splitext, basename
from math import log, sqrt
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

import argparse

def get_all_files(directory):
    return [join(directory,f) for f in listdir(directory) if isfile(join(directory,f))]

def standardize(rawexcerpt):
    return word_tokenize(rawexcerpt.decode('utf8').lower())

def load_file_excerpts(filepath):
    with open(filepath, 'r') as f:
#        return [line.decode('utf8').lower().strip() for line in f]
        return [standardize(line) for line in f]

def load_directory_excerpts(dirpath):
    return flatten([load_file_excerpts(f) for f in get_all_files(dirpath)])

def flatten(listoflists):
    return [elem for sublist in listoflists for elem in sublist]

def getvocab(sentences):
    vocab = {}
    vocabid = 0
    for s in sentences:
        for w in s:
            if w not in vocab:
                vocab[w] = vocabid
                vocabid += 1
    return vocab

def count_cooccur(vocab, sentences):
    cooccurmat = np.zeros((len(vocab),len(vocab)))
    for n, s in enumerate(sentences):
        for i, w in enumerate(s):
            for j, x in enumerate(s):
                if i != j:
                    cooccurmat[vocab[w],vocab[x]] += 1
    return cooccurmat

def populate_cooccurmat(vocab, cooccurdict):
    cooccurmat = np.zeros((len(vocab),len(vocab)))
    for w, context in cooccurdict.iteritems():
        row = vocab[w]
        for c in context:
            col = vocab[c]
            cooccurmat[row,col] += 1
    return cooccurmat

def reduce_dim(cooccurmat, k):
    svd = TruncatedSVD(n_components=k)
    dimred = svd.fit_transform(cooccurmat)
    return dimred

def create_cooccurmat(dirpath):
    sentences = flatten([load_file_excerpts(f) for f in get_all_files(dirpath)])
    vocab = getvocab(sentences)
    cooccurmat = count_cooccur(vocab, sentences)

    dimredmat = reduce_dim(cooccurmat, 50)
    print dimredmat.shape
#    vectorizer = CountVectorizer(ngram_range=(100,100))
#    cooccurs = vectorizer.fit_transform(samples)
#    sums = np.sum(cooccurs.todense(),axis=0)
#    print zip(vectorizer.get_feature_names(),np.array(sums)[0].tolist())
#    return cooccurs.todense()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Get Word Vectors")
#    parser.add_argument("inputdir", help="directory containing corpus")

    text = create_cooccurmat("/home1/c/cis530/hw1/data/train")
    

#    corpus = load_directory_excerpts("/home1/c/cis530/hw1/data/train")
#    sample = load_file_excerpts("/home1/c/cis530/hw1/data/train/background.txt")
#    print corpus
 #   print sample

    #assume bag of words
