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

stopwords = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=0.8, pos_prior=0.8):
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

    #remove stopwords
    for i in range(len(train_set)):
        sen = []
        for word in train_set[i]:
            if word not in stopwords:
                sen.append(word)
        train_set[i] = sen
    for i in range(len(dev_set)):
        sen = []
        for word in dev_set[i]:
            if word not in stopwords:
                sen.append(word)
        dev_set[i] = sen

    #build unigram model
    logProb = buildUniGram(train_set, train_labels, smoothing_parameter)

    dev_labels = []
    for sen in dev_set:
        #calculate pos and neg log probabilities
        logPost_Pos, logPost_Neg = calculateUnigramLogProb(sen, pos_prior, logProb)
        #determine labels
        if logPost_Pos >= logPost_Neg:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    return dev_labels

def buildUniGram(train_set, train_labels, smoothing_parameter):
    logProb = {0: defaultdict(float), 1: defaultdict(float)}
    #number of words in each class
    totalWords = {0: 0, 1: 0}
    #get word count
    for i, sen in enumerate(train_set):
        for word in sen:
            logProb[train_labels[i]][word] += 1.0
        totalWords[train_labels[i]] += len(sen)
    #add UNK count
    logProb[0]["UNK"] = 0.0
    logProb[1]["UNK"] = 0.0
    #calculate word log probability based on word count
    for i in [0, 1]:
        for word in logProb[i]:
            logProb[i][word] = math.log(logProb[i][word] + smoothing_parameter) - math.log(totalWords[i] + smoothing_parameter * len(logProb[i]))
    return logProb

def calculateUnigramLogProb(sen, pos_prior, logProb):
    logPost_Pos = math.log(pos_prior)
    logPost_Neg = math.log(1 - pos_prior)
    for word in sen:
        #calculate postive log probability
        if word in logProb[1]:
            logPost_Pos += logProb[1][word]
        else:
            logPost_Pos += logProb[1]["UNK"]
        #calculate negative log probability
        if word in logProb[0]:
            logPost_Neg += logProb[0][word]
        else:
            logPost_Neg += logProb[0]["UNK"]
    return logPost_Pos, logPost_Neg

def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=0.1, bigram_smoothing_parameter=1.0 / 2 ** 2, bigram_lambda=0.1,pos_prior=0.8):
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
    for i in range(len(train_set)):
        sen = []
        for word in train_set[i]:
            if word not in stopwords:
                sen.append(word)
        train_set[i] = sen
    for i in range(len(dev_set)):
        sen = []
        for word in dev_set[i]:
            if word not in stopwords:
                sen.append(word)
        dev_set[i] = sen
    uni_logProb = buildUniGram(train_set, train_labels, unigram_smoothing_parameter)
    bi_logProb, bi_vocab = buildBiGram(train_set, train_labels, bigram_smoothing_parameter)
    dev_labels = []
    for sen in dev_set:
        uni_logPrior_Pos, uni_logPrior_Neg = calculateUnigramLogProb(sen, pos_prior, uni_logProb)
        bi_logPrior_Pos, bi_logPrior_Neg = calculateBigramLogProb(sen, pos_prior, bi_logProb, bi_vocab)
        logPrior_Pos = (1 - bigram_lambda) * uni_logPrior_Pos + bigram_lambda * bi_logPrior_Pos
        logPrior_Neg = (1 - bigram_lambda) * uni_logPrior_Neg + bigram_lambda * bi_logPrior_Neg
        if logPrior_Pos > logPrior_Neg:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    return dev_labels

def buildBiGram(train_set, train_labels, smoothing_parameter):
    logProb = {0: defaultdict(float), 1: defaultdict(float)}
    totalWords = {0: 0, 1: 0}
    vocab = {0: set(), 1: set()}
    for i, sen in enumerate(train_set):
        for j in range(len(sen) - 1):
            vocab[train_labels[i]].add(sen[j])
            logProb[train_labels[i]][(sen[j], sen[j + 1])] += 1.0
        vocab[train_labels[i]].add(sen[-1])
        totalWords[train_labels[i]] += len(sen) - 1
    for i in [0, 1]:
        logProb[i][("UNK", "UNK")] = 0.0
        for word in vocab[i]:
            logProb[i][(word, "UNK")] = 0.0
            logProb[i][("UNK", word)] = 0.0
    for i in [0, 1]:
        for pair in logProb[i]:
            logProb[i][pair] = math.log(logProb[i][pair] + smoothing_parameter) - math.log(totalWords[i] + smoothing_parameter * len(logProb[i]))
    return logProb, vocab

def calculateBigramLogProb(sen, pos_prior, logProb, vocab):
    logPost_Pos = math.log(pos_prior)
    logPost_Neg = math.log(1 - pos_prior)
    sen_pos = sen.copy()
    sen_neg = sen.copy()
    for i in range(len(sen)):
        if sen[i] not in vocab[1]:
            sen_pos[i] = "UNK"
        if sen[i] not in vocab[0]:
            sen_neg[i] = "UNK"
    for i in range(len(sen_pos) - 1):
        logPost_Pos += logProb[1][(sen_pos[i], sen_pos[i + 1])]
    for i in range(len(sen_neg) - 1):
        logPost_Neg += logProb[0][(sen_neg[i], sen_neg[i + 1])]
    return logPost_Pos, logPost_Neg
