"""
Slightly modified from https://github.com/peter3125/sentence2vec with
some new methods for preprocessing sentences, calculating cosine similarity
with sklearn, and a pipeline for comparing an input sentence against a corpus
for some braindead question answering purposes
"""
from __future__ import print_function
import time

import gensim
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec
import spacy

import math
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


nlp = spacy.load('en')

gnews_model = gensim.models.KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300.bin', binary=True)


class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector


class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list
    # return the length of a sentence
    def len(self):
        return(len(self.word_list))
    def __str__(self):
        word_str_list = [word.text for word in self.word_list]
        return ' '.join(word_str_list)
    def __repr__(self):
        return self.__str__()


def get_word_frequency(word_text, vec_model):
    wf = vec_model.vocab[word_text].count
    return(wf)


def preloading_sentences(sentence_list, model):
    """
    Converts a list of sentences into a list of Sentence (and Word) objects
    Pretty similar to what peter3125/sentence2vec.git does

    input: a list of sentences, embedding_size
    output: a list of Sentence objects, containing Word objects, which contain 'text' and word vector attributes
    """
    embedding_size = 300
    all_sent_info = []
    for sentence in sentence_list:
        sent_info = []
        spacy_sentence = nlp(sentence)
        for word in spacy_sentence:
            if word.text in model.vocab:
                sent_info.append(Word(word.text, model[word.text]))
        # todo: if sent_info > 0, append, else don't
        all_sent_info.append(Sentence(sent_info))
    return(all_sent_info)


def sentence_to_vec(sentence_list, embedding_size, a=1e-3):
    """
    A SIMPLE BUT TOUGH TO BEAT BASELINE FOR SENTENCE EMBEDDINGS

    Sanjeev Arora, Yingyu Liang, Tengyu Ma
    Princeton University
    """
    sentence_set = [] # intermediary list of sentence vectors before PCA
    sent_list = [] # return list of input sentences in the output
    for sentence in sentence_list:
        this_sent = []
        vs = np.zeros(embedding_size) # add all w2v values into one vector for the sentence
        sentence_length = sentence.len()
        for word in sentence.word_list:
            this_sent.append(word.text)
            word_freq = get_word_frequency(word.text, gnews_model)
            a_value = a / (a + word_freq) # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, word.vector)) # vs += sif * word_vector
        vs = np.divide(vs, sentence_length) # weighted average, normalized by sentence length
        sentence_set.append(vs) # add to our existing re-caculated set of sentences
        sent_list.append(' '.join(this_sent))
    # calculate PCA of this sentence set
    pca = PCA(n_components=embedding_size)
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT
    # pad the vector? (occurs if we have less sentences than embeddings_size)
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)
    # resulting sentence vectors, vs = vs -u * uT * vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u, vs)
        sentence_vecs.append(np.subtract(vs, sub))
    return(sentence_vecs, sent_list)


def get_sen_embeddings(sentence_list):
    """
    Create Sentence and Word objects from a list and pass them to sentence_to_vec()
    Return a list of _sentence embeddings_ for all sentences in the list
    """
    embedding_size = 300
    all_sent_info = preloading_sentences(sentence_list, gnews_model)
    sentence_vectors, sent_list = sentence_to_vec(all_sent_info, embedding_size)
    return(sentence_vectors, sent_list)


def get_cos_distance(sentence_list):
    """
    Create Sentence and Word objects from a list and pass them to sentence_to_vec()
    Return a matrix of the _cosine distance_ of elements in the matrix
    This is used for sentence similarity functions

    input: A list of plaintext sentences
    output: A list of sentence distances
    """
    sentence_vectors, sent_list = get_sen_embeddings(sentence_list)
    cos_list = cosine_similarity(sentence_vectors, Y=None, dense_output=True)
    return(cos_list, sent_list)


def get_most_similar(utterance, sentence_list):
    """
    Takes an input utterance and a corpus sentence_list to compare it to,
    and returns a dict of the utterance, closest question, and relevant answer
    """
    sentence_list.append(utterance)
    cos_list, sent_text = get_cos_distance(sentence_list)
    tmp_list = list(cos_list[len(cos_list)-1])
    tmp_indx = tmp_list.index(max(tmp_list[0:len(tmp_list)-1]))
    return(tmp_indx)


if __name__ == "__main__":
    # load questions to compare simliarity to input/test utterances
    faq_csv = pd.read_csv("./utterances_test.csv")
    sentence_list = list(faq_csv['utterance'])
    answer_list = list(faq_csv['answer'])
    max_idx = get_most_similar(utterance, sentence_list)
    # print the relevant question and answer pair
    sentence_list[max_idx]
    answer_list[max_idx]