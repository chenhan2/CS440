# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
from collections import defaultdict
import math

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=1.0, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set
    logProb = {0: defaultdict(float), 1: defaultdict(float)}
    totalWords = {0: 0, 1: 0}
    logProb[0]["UNK"] = 0.0
    logProb[1]["UNK"] = 0.0
    for i, sen in enumerate(train_set):
        for word in sen:
            logProb[train_labels[i]][word] += 1.0
        totalWords[train_labels[i]] += len(sen)
    for i in [0, 1]:
        for word in logProb[i]:
            logProb[i][word] = math.log(lobProb[i][word] + 1) - math.log(totalWords[i] + len(logProb[i]))
    prior = {}
    prior[1] = math.sum(train_labels == 1) * 1.0 / len(train_labels)
    prior[0] = 1 - prior[1]

    dev_labels = []
    for sen in dev_set:
        prob_Pos = math.log(prior[1])
        prob_Neg = math.log(prior[0])
        for word in sen:
            if word in logProb[1]:
                prob_Pos += logProb[1][word]
            else:
                prob_pos += logProb[1]["UNK"]
            if word in logProb[0]:
                prob_Neg += logProb[0][word]
            else:
                prob_pos += logProb[1]["UNK"]
        if prob_Pos > prob_Neg:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    return dev_labels

def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=1.0, bigram_smoothing_parameter=1.0, bigram_lambda=0.5,pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smoothing_parameter - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set using a bigram model
    return []
