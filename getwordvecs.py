import numpy as np
from nltk import word_tokenize
from os import listdir
from os.path import isfile, join, splitext, basename
from math import log, sqrt
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import lil_matrix
import pickle
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

def get_sentencevocab(sentences):
    vocab = {}
    vocabid = 0
    for s in sentences:
        for w in s:
            if w not in vocab:
                vocab[w] = vocabid
                vocabid += 1
    return vocab


def get_intersectionvocab(lexfile, sentvocab):
    vocab = {}
    vocabid = 0
    with open(lexfile, "r") as f:
        for line in f:
            word = line.decode("utf8").split("\t")[0].strip()
            if word in sentvocab and word not in vocab:
                vocab[word] = vocabid
                vocabid += 1
    return vocab

def count_cooccur(lexvocab, sentvocab, sentences, k):
    cooccurmat = lil_matrix((len(lexvocab),len(sentvocab)))
    for n, s in enumerate(sentences):
        for i, w in enumerate(s):
            if w not in lexvocab:
                continue
            for j, x in enumerate(s):
                if i != j:
                    cooccurmat[lexvocab[w],sentvocab[x]] += 1
    return reduce_dim(cooccurmat, k)


def reduce_dim(cooccurmat, k):
    svd = TruncatedSVD(n_components=k)
    dimred = svd.fit_transform(cooccurmat)
    return dimred


def create_cooccurmat(dirpath, lexiconfile):
    sentences = flatten([load_file_excerpts(f) for f in get_all_files(dirpath)])#[0:20]
    sentvocab = get_sentencevocab(sentences)
    intvocab = get_intersectionvocab(lexiconfile, sentvocab)
    print len(sentvocab)
    print len(intvocab)
    cooccurmat = count_cooccur(intvocab, sentvocab, sentences, 50)
    print cooccurmat.shape
    print type(cooccurmat)
    
#    vectorizer = CountVectorizer(ngram_range=(100,100))
#    cooccurs = vectorizer.fit_transform(samples)
#    sums = np.sum(cooccurs.todense(),axis=0)
#    print zip(vectorizer.get_feature_names(),np.array(sums)[0].tolist())
#    return cooccurs.todense()


if __name__ == "__main__":
    # python getwordvecs.py /home1/c/cis530/hw1/data/train wordlist.2010.eng.utf8.txt --savemat cooccur.pickle
    # python getwordvecs.py /home1/c/cis530/hw1/data/train wordlist.2010.eng.utf8.txt --loadmat cooccur.pickle

    parser = argparse.ArgumentParser(description = "Get Word Vectors")
    parser.add_argument("inputdir", help="directory containing sentences")
    parser.add_argument("lexfile", help="lexicon file")
    parser.add_argument("--savemat", help="pickle the SVD'd cooccurrence matrix")
    parser.add_argument("--loadmat", help="unpickle the SVD'd coccurrence matrix")

    args = parser.parse_args()

    if args.loadmat:
        with open(args.loadmat, 'r') as instream:
            cooccurmat = pickle.load(instream)
    else:
        cooccurmat = create_cooccurmat(args.inputdir, args.lexfile)
    if args.savemat:
        with open(args.savemat, 'wb') as outstream:
            pickle.dump(cooccurmat, outstream)

#    corpus = load_directory_excerpts("/home1/c/cis530/hw1/data/train")
#    sample = load_file_excerpts("/home1/c/cis530/hw1/data/train/background.txt")
#    print corpus
 #   print sample

    #assume bag of words
