"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import defaultdict
def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tagCount = {"UNK":defaultdict(int)}
    for sen in train:
        for word, tag in sen:
            if word not in tagCount:
                tagCount[word] = defaultdict(int)
            tagCount[word][tag] += 1
            tagCount["UNK"][tag] += 1
    wordTag = {}
    for word in tagCount:
        maxTag = None
        maxCount = 0
        for tag in tagCount[word]:
            if tagCount[word][tag] > maxCount:
                maxCount = tagCount[word][tag]
                maxTag = tag
        wordTag[word] = maxTag
    result = []
    for sen in test:
        resultSen = []
        for word in sen:
            if word in wordTag:
                resultSen.append((word, wordTag[word]))
            else:
                resultSen.append((word, wordTag["UNK"]))
        result.append(resultSen)
    return result
