"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""
from collections import defaultdict
from math import log, inf

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    initial, transition, emission, tagList = buildProbs(train)
    tagList.remove("START")
    result = []
    for sen0 in test:
        #ignore START and END
        sen = sen0[1:-1]
        # print(sen)
        v = {}
        b = {}
        v[0] = {}
        #set v(0, tags)
        for tag in initial:
            if sen[0] in emission[tag]:
                v[0][tag] = initial[tag] + emission[tag][sen[0]]
            else:
                v[0][tag] = initial[tag] + emission[tag]["UNK"]
        #iteratively set v(k, tags)
        for k in range(1, len(sen)):
            word = sen[k]
            v[k] = {}
            b[k] = {}
            #for all tag2
            for tag2 in tagList:
                v_max = -inf
                tag_max = None
                #find the best tag1 in the previous level
                for tag1 in v[k - 1]:
                    v_tmp = v[k - 1][tag1] + transition[tag1][tag2]
                    if v_tmp > v_max:
                        v_max = v_tmp
                        tag_max = tag1
                if word in emission[tag2]:
                    v[k][tag2] = v_max + emission[tag2][word]
                else:
                    v[k][tag2] = v_max + emission[tag2]["UNK"]
                #remember the path
                b[k][tag2] = tag_max
        #constuct the optimal path
        tagPath = []
        lastTag = None
        lastV = -inf
        #find the last tag
        for tag in v[len(sen) - 1]:
            if v[len(sen) - 1][tag] > lastV:
                lastV = v[len(sen) - 1][tag]
                lastTag = tag
        #backtracking
        k = len(sen) - 1
        while k > 0 and k in b:
            tagPath = [lastTag] + tagPath
            lastTag = b[k][lastTag]
            k -= 1
        tagPath = [lastTag] + tagPath
        #add tag to sentence
        tagged = [("START", "START")]
        for i in range(len(sen)):
            tagged.append((sen[i], tagPath[i]))
        tagged.append(("END", "END"))
        result.append(tagged)
        # print(tagged)
    return result

def buildProbs(train, alpha = 1e-4, beta = 1e-6):
    tagPair = {}
    totalPair = defaultdict(int)
    tagWord = {}
    countTag = defaultdict(int)
    for sen in train:
        #count tag pairs
        for i in range(len(sen) - 1):
            if i + 1 < len(sen) - 1:
                if sen[i][1] not in tagPair:
                    tagPair[sen[i][1]] = defaultdict(int)
                tagPair[sen[i][1]][sen[i + 1][1]] += 1
                totalPair[sen[i][1]] += 1

        #count tag word pairs
            if sen[i][1] not in tagWord:
                tagWord[sen[i][1]] = defaultdict(int)
                #tags other than "START" can emit "UNK"
                if sen[i][1] != "START":
                    tagWord[sen[i][1]]["UNK"] = 0
            tagWord[sen[i][1]][sen[i][0]] += 1
            countTag[sen[i][1]] += 1

    #add unseen tag pairs for smoothing purpose
    for tag1 in tagPair:
        for tag2 in countTag:
            if tag2 != "START" and tag2 not in tagPair[tag1]:
                tagPair[tag1][tag2] = 0

    #calculate initial prob
    initial = {}
    for tag in tagPair["START"]:
        initial[tag] = log(tagPair["START"][tag] + alpha) - log(totalPair["START"] + alpha * len(tagPair["START"]))
    # print(initial)
    #calculate transition prob
    transition = {}
    for tag1 in tagPair:
        if tag1 == "START":
            continue
        transition[tag1] = {}
        for tag2 in tagPair[tag1]:
            transition[tag1][tag2] = log(tagPair[tag1][tag2] + alpha) - log(totalPair[tag1] + alpha * len(tagPair[tag1]))

    hapex = countHapex(train)
    #calculate emission prob
    # print(hapex)
    emission = {}
    for tag in tagWord:
        if tag == "START" or tag == "END":
            continue
        emission[tag] = {}
        for word in tagWord[tag]:
            # if tagWord[tag][word] + hapex[tag] * beta == 0 or countTag[tag] + hapex[tag] * beta * len(tagWord[tag]) == 0:
            #     print(tag, tagWord[tag][word], hapex[tag])
            emission[tag][word] = log(tagWord[tag][word] + hapex[tag] * beta) - log(countTag[tag] + hapex[tag] * beta * len(tagWord[tag]))

    return initial, transition, emission, list(countTag.keys())


def countHapex(train, gamma = 1.0):
    once = set()
    multi = set()
    hapexCount = defaultdict(float)
    tagList = set()
    for sen in train:
        for word, tag in sen:
            if tag not in tagList:
                tagList.add(tag)
            if word not in once and word not in multi:
                once.add(word)
            elif word in once:
                once.remove(word)
                multi.add(word)
    for sen in train:
        for word, tag in sen:
            if word in once:
                hapexCount[tag] += 1
    totalHapex = 0
    # print(hapexCount)
    for tag in tagList:
        if tag not in hapexCount:
            hapexCount[tag] = 0.0
    for tag in hapexCount:
        totalHapex += hapexCount[tag]
    for tag in hapexCount:
        hapexCount[tag] = (hapexCount[tag] + gamma) / (totalHapex + gamma * len(hapexCount))
    return hapexCount
