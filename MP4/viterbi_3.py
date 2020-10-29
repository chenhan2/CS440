"""
Part 4: Here should be your best version of viterbi,
with enhancements such as dealing with suffixes/prefixes separately
"""
from collections import defaultdict
from math import log, inf
from copy import deepcopy
def viterbi_3(train, test):
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
        word = sen[0]
        for tag in initial:
            if word in emission[tag]:
                v[0][tag] = initial[tag] + emission[tag][word]
            elif word.endswith("ly"):
                v[0][tag] = initial[tag] + emission[tag]["X+ly"]
            elif word.endswith("al"):
                v[0][tag] = initial[tag] + emission[tag]["X+al"]
            elif word.endswith("ed") or word.endswith("ing"):
                v[0][tag] = initial[tag] + emission[tag]["X+ed/ing"]
            elif word.endswith("er") or word.endswith("or") or word.endswith("ers") or word.endswith("ors"):
                v[0][tag] = initial[tag] + emission[tag]["X+er/or"]
            elif word.endswith("ion") or word.endswith("ions"):
                v[0][tag] = initial[tag] + emission[tag]["X+ion"]
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
                elif word.endswith("ly"):
                    v[k][tag2] = v_max + emission[tag2]["X+ly"]
                elif word.endswith("al"):
                    v[k][tag2] = v_max + emission[tag2]["X+al"]
                elif word.endswith("ed") or word.endswith("ing"):
                    v[k][tag2] = v_max + emission[tag2]["X+ed/ing"]
                elif word.endswith("er") or word.endswith("or") or word.endswith("ers") or word.endswith("ors"):
                    v[k][tag2] = v_max + emission[tag2]["X+er/or"]
                elif word.endswith("ion") or word.endswith("ions"):
                    v[k][tag2] = v_max + emission[tag2]["X+ion"]
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
                    tagWord[sen[i][1]]["X+ly"] = 0
                    tagWord[sen[i][1]]["X+al"] = 0
                    tagWord[sen[i][1]]["X+ed/ing"] = 0
                    tagWord[sen[i][1]]["X+er/or"] = 0
                    tagWord[sen[i][1]]["X+ion"] = 0
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

    hapex, unseen = countHapex(tagWord)
    tagWord = mappingUnseen(tagWord, unseen)
    hapex, unseen = countHapex(tagWord)
    #calculate emission prob
    emission = {}
    for tag in tagWord:
        if tag == "START" or tag == "END":
            continue
        emission[tag] = {}
        for word in tagWord[tag]:
            emission[tag][word] = log(tagWord[tag][word] + hapex[tag] * beta) - log(countTag[tag] + hapex[tag] * beta * len(tagWord[tag]))

    return initial, transition, emission, list(countTag.keys())

def countHapex(tagWord, gamma = 1e-1):
    once = set()
    multi = set()
    hapexCount = defaultdict(float)
    tagList = set()
    for tag in tagWord:
        if tag not in tagList:
            tagList.add(tag)
        for word in tagWord[tag]:
            if tag not in tagList:
                tagList.add(tag)
            if word not in once and word not in multi:
                once.add(word)
            elif word in once:
                once.remove(word)
                multi.add(word)
    for tag in tagWord:
        for word in tagWord[tag]:
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
    return hapexCount, once

def mappingUnseen(tagWord, hapex, cnt = 1e-4):
    newTagWord = deepcopy(tagWord)
    for tag in tagWord:
        for word in tagWord[tag]:
            if word in hapex:
                if word.endswith("ly"):
                    newTagWord[tag]["X+ly"] += cnt
                    newTagWord[tag].pop(word)
                elif word.endswith("al"):
                    newTagWord[tag]["X+al"] += cnt
                    newTagWord[tag].pop(word)
                elif word.endswith("ed") or word.endswith("ing"):
                    newTagWord[tag]["X+ed/ing"] += cnt
                    newTagWord[tag].pop(word)
                elif word.endswith("er") or word.endswith("or") or word.endswith("ers") or word.endswith("ors"):
                    newTagWord[tag]["X+er/or"] += cnt
                    newTagWord[tag].pop(word)
                elif word.endswith("ion") or word.endswith("ions"):
                    newTagWord[tag]["X+ion"] += cnt
                    newTagWord[tag].pop(word)
    return newTagWord
