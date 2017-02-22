import numpy as np
from nltk import word_tokenize
import re
import os
from os import listdir
from os.path import isfile, join, splitext, basename
from math import log, sqrt
from collections import defaultdict
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import lil_matrix
import pickle
import argparse

FINDWORD = re.compile(r"\(\w+\s([a-z]+)\)")

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
    vocabfreqs = defaultdict(int)
    vocabid = 0
    for s in sentences:
        for w in s:
            vocabfreqs[w] += 1
            if w not in vocab:
                vocab[w] = vocabid
                vocabid += 1
    return vocab, dict(vocabfreqs)


def get_intersectionvocab(lexfile, sentvocabfreqs, minfreq=1):
    vocab = {}
    vocabid = 0
    with open(lexfile, "r") as f:
        for line in f:
            word = line.decode("utf8").split("\t")[0].strip()
            if word in sentvocabfreqs and sentvocabfreqs[word] >= minfreq and word not in vocab:
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

def read_brown_cmb(filename):
    wcount = 0
    sentences = []
    with open(filename, "r") as f:
        sentence = []
        for line in f:
            if "END_OF_TEXT_UNIT" in line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            words = FINDWORD.findall(line.lower().replace("'",""))
            for word in words:
                sentence.append(word)
    return sentences


def read_brown(dirpath):
    allsentences = []
    for subdir, dirs, files in os.walk(dirpath):
        if os.path.basename(subdir)[0] != "c":
            continue
        for f in files:
            if f.split(".")[-1] != "cmb":
                continue

            sentences = read_brown_cmb(os.path.join(subdir,f))
            allsentences.extend(sentences)
    return allsentences

                

def create_cooccurmat(dirpath, lexiconfile, filetype=None):
    sentences = []
    if filetype == "BROWN":
        sentences = read_brown(dirpath)
    else:
        sentences = flatten([load_file_excerpts(f) for f in get_all_files(dirpath)])#[0:20]
    sentvocab, sentvocabfreqs = get_sentencevocab(sentences)
    intvocab = get_intersectionvocab(lexiconfile, sentvocabfreqs, minfreq=5)
#    print len(sentvocab)
#    print len(intvocab)
    cooccurmat = count_cooccur(intvocab, sentvocab, sentences, 50)
#    print cooccurmat.shape
#    print type(cooccurmat)
    return cooccurmat, intvocab

def load_pairs_by_suff(fname, minpairs=1):
    pairs_by_suff = {}
    with open(fname, "r") as f:
        for line in f:
            components = line.decode("utf8").strip().split("\t")
            suff = components[2]
            root = components[0]
            deriv = components[1]
#            suff, root, deriv = line.decode("utf8").split("\t")
            if suff not in pairs_by_suff:
                pairs_by_suff[suff] = [(root, deriv)]
            else:
                pairs_by_suff[suff].append((root, deriv))

        dels = set([])
        for suff, pairs in pairs_by_suff.iteritems():
            if len(pairs) < minpairs:
                dels.add(suff)
        for suff in dels:
            del pairs_by_suff[suff]
    return pairs_by_suff

def get_CCA_by_suff(pairs_by_suff, cooccurmat, lexicon):
    for suff, pairs in pairs_by_suff.iteritems():
        xrows = [lexicon[pair[0]] for pair in pairs if pair[0] in lexicon and pair[1] in lexicon]
        yrows = [lexicon[pair[1]] for pair in pairs if pair[0] in lexicon and pair[1] in lexicon]
        if len(xrows) < 2 or len(yrows) < 2:
            continue
        numattested = len([1 for pair in pairs if pair[0] in lexicon and pair[1] in lexicon])

        X = cooccurmat[xrows][:]
        Y = cooccurmat[yrows][:]
        cca = CCA(n_components=50)
        try:
            X_c, Y_c = cca.fit_transform(X, Y)
            print X_c
        except np.linalg.linalg.LinAlgError:
            print "singular matrix error"
            print "\t", suff
            print "\t\t", xrows
            print "\t\t", yrows
            pass
        except ValueError:  # https://github.com/scikit-learn/scikit-learn/pull/4420
            print "inf or NaN error"
            print "\t", suff
            print "\t\t", xrows
            print "\t\t", yrows
            pass

#        for pair in pairs:
#            if pair[0] in lexicon and pair[1] in lexicon:
#                print pair[0], pair[1], "\t", lexicon[pair[0]], lexicon[pair[1]]
#        numattested = len([1 for pair in pairs if pair[0] in lexicon and pair[1] in lexicon])
#        if numattested > 100:
#            print suff, numattested


if __name__ == "__main__":
    # python getwordvecs.py /mnt/pollux-new/cis/nlp/data/corpora/brown  wordlist.2010.eng.utf8.txt --loadmat cooccurbrown.pickle --pairsfile morphopairs.txt --loadlex lexicon.pickle 

    parser = argparse.ArgumentParser(description = "Get Word Vectors")
    parser.add_argument("inputdir", help="directory containing sentences")
    parser.add_argument("lexfile", help="lexicon file")
    parser.add_argument("--savemat", help="pickle the SVD'd cooccurrence matrix")
    parser.add_argument("--loadmat", help="unpickle the SVD'd coccurrence matrix")
    parser.add_argument("--savelex", help="pickle the lexicon dictionary")
    parser.add_argument("--loadlex", help="unpickle the lexicon dictionary")
    parser.add_argument("--pairsfile", help="file with pairs derived by affix")

    args = parser.parse_args()


    if args.loadmat:
        with open(args.loadmat, 'r') as instream:
            cooccurmat = pickle.load(instream)
        if args.loadlex:
            with open(args.loadlex, 'r') as instream:
                lexicon = pickle.load(instream)
    else:
        cooccurmat, lexicon = create_cooccurmat(args.inputdir, args.lexfile, filetype="BROWN")
    if args.savemat:
        with open(args.savemat, 'wb') as outstream:
            pickle.dump(cooccurmat, outstream)
    if args.savelex:
        with open(args.savelex, 'wb') as outstream:
            pickle.dump(lexicon, outstream)

    pairs_by_suff = load_pairs_by_suff(args.pairsfile, 5)

    get_CCA_by_suff(pairs_by_suff, cooccurmat, lexicon)

    #
    
