import numpy as np


def getMaxToken(train_data):
    maxLen = 0
    for t in train_data:
        if len(t[1]) > maxLen:
            maxLen = len(t[1])
    return maxLen + 1

def checkWordExist(sample, model):
    for sub_word in sample:
        if sub_word not in model:
            return False
    return True

def _checkAllWordExist(sample, model):
    if sample[0] not in model:
        return False
    for sub_word in sample[1]:
        if sub_word not in model:
            return False
    return True

def getSample(sample, model,max_length,vec_length):
    vector = []
    for word in sample:
        if word in model:
            vector.append(model[word.lower()])
    for i in range(max_length - len(sample)):
        vector.append(np.zeros(vec_length))
    return vector

def loadData(train_data, model, start, bs,max_length,vec_length):
    train = []
    label = []
    for sample in train_data[start:start+bs]:
        if _checkAllWordExist(sample,model):
            label.append(model[sample[0]])
            train.append(np.array(getSample(sample[1], model,max_length,vec_length), dtype=np.float).astype(np.float32))
    return np.array(train, dtype=np.float).astype(np.float32), np.array(label, dtype=np.float).astype(np.float32)

def genTrainingFromVocab(word_vectors):
    with open('training.txt', 'w') as fp:
        for t in word_vectors.vocab:
            if '_' in t:
                w = t.split('_')
                try:
                    fp.write(u' '.join(w) + '\t' + t + '\n')
                except UnicodeEncodeError:
                    print ('unicide error', t)

def getAverageVector(word_vectors,words):
    avgVec = []
    if not checkWordExist(words, word_vectors):
        return None
    for word in words:
        if len(avgVec) == 0:
            avgVec = word_vectors[word]
        else:
            avgVec = avgVec + word_vectors[word]
    return avgVec/len(words)

def compareVectorsDist(org,rnn,avg):
    rnn_dist = np.linalg.norm(org-rnn)
    avg_dist = np.linalg.norm(org-avg)
    if rnn_dist < avg_dist:
        return 1
    return 0

def compareVectorsSim(org,rnn,avg):
    rnn_sim = cos_sim(org,rnn)
    avg_sim = cos_sim(org,avg)
    if rnn_sim > avg_sim:
        return 1
    return 0

def cos_sim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def ranking(word_vectors,vec,target):
    rank = 0
    sim_ref = cos_sim(vec,word_vectors[target])
    for w in word_vectors.vocab:
        sim = cos_sim(word_vectors[w],vec)
        if sim > sim_ref:
            rank = rank + 1
    return rank

# test function
def test(word_vectors,wordList,predict_vec):

    #EVal Metric
    dist_result = 0.0
    sim_result =  0.0
    rank_avg = 0.0
    rank_rnn = 0.0

    for w in wordList:
        target = '_'.join(w)
        orgVec = word_vectors[target]
        rnnVec = predict_vec[target]
        avgVec = getAverageVector(word_vectors,w)

        #Check vector is available
        if orgVec is None or rnnVec is None or avgVec is None:
            continue

        dist_result = dist_result + compareVectorsDist(orgVec,rnnVec,avgVec)
        sim_result = sim_result + compareVectorsSim(orgVec,rnnVec,avgVec)

        rank_avg = rank_avg + ranking(word_vectors,avgVec,target)
        rank_rnn = rank_rnn + ranking(word_vectors, rnnVec, target)


    print('Dist Improv', str(dist_result/len(wordList)))
    print('Sim Improv', str(sim_result/len(wordList)))
    print('MRR AVG :', str(1/(rank_avg/len(wordList))))
    print('MRR RNN', str(1/(rank_rnn/len(wordList))))
