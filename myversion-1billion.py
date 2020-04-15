import os, sys

import numpy as np
import random
import pickle
import time
import queue
import math
import sys
from collections import deque
from config import *
from huffman_tree import HuffmanCoding
from sklearn.metrics.pairwise import cosine_similarity


# 미니배치일 때
def generate_batch(pwi, codedict, vocab):
    assert batch_size() % (window_size() * 2) == 0
    batch = np.ndarray(shape=(batch_size()), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size()), dtype=np.int32)
    prob = random.random()

    span = int(batch_size() / (window_size() * 2)) + 2 * window_size()

    buffer = deque(maxlen=span)
    # buffer(batch와 label 안에 들어갈 데이터들) 만들기

    cnt = 0
    buffercnt = 0

    while cnt < span:
        move = vocab[buffercnt]
        if subsampling() == True:
            if pwi[list(vocab)[cnt]] < prob:
                buffer.append(move)
                cnt += 1
        else:
            buffer.append(move)
            cnt += 1
        buffercnt += 1

    for i in range(int(batch_size() / (2 * window_size()))):
        vocab.popleft()
    # batch 만들기

    batchcnt = 0

    for i in range(window_size(), span - window_size()):
        for j in range(window_size() * 2):
            batch[batchcnt] = dictionary[buffer[i]]
            batchcnt += 1

    # label 만들기
    labelcnt = 0
    for i in range(window_size(), span - window_size()):
        for j in range(window_size() * 2 + 1):
            if j + i - window_size() != i:
                labels[labelcnt] = dictionary[buffer[j + i - window_size()]]
                labelcnt += 1

    if method() == 0:
        return batch, labels, vocab
    # else:
    #     path = []
    #     sign = []
    #     for i in range(len(labels)):
    #         huff = codedict[reverse_dictionary[i]]
    #         path.append([])
    #         path[i].append(0)
    #         sign.append([])
    #
    #         for j in range(len(huff) - 1):
    #             if huff[j + 1] == "0":
    #                 path[i].append(2 * (path[i][j] + 1) - 1)
    #             else:
    #                 path[i].append(2 * (path[i][j] + 1))
    #         for j in range(maxhuff - len(huff)):
    #             path[i].append(dic_size - 1)
    #
    #         for j in huff:
    #             if j == "0":
    #                 sign[i].append(1)
    #             else:
    #                 sign[i].append(-1)
    #         for j in range(maxhuff - len(huff)):
    #             sign[i].append(0)
    #     for q in path:
    #         for w in q:
    #             if w > dic_size:
    #                 print(path)
    #     return batch, labels, path, sign, vocab


# def batch_test(vocab):
#     if method() == 0:
#         batch_inputs, batch_labels, batch_labels_dic = generate_batch(pwi, None, vocab)
#     else:
#         imsicodedict = codedict
#         batch_inputs, batch_labels, path, sign = generate_batch(pwi, imsicodedict, vocab)
#     for i in range(batch_size()):
#         print(batch_inputs[i], reverse_dictionary[batch_inputs[i]], '->', batch_labels[i],
#               reverse_dictionary[batch_labels[i]])


def datafeed(vocab, filecnt):
    with open("/hdd/user5/word2vec/1billion/1billion-voca/voca" + str(filecnt), "rb") as f:
        thisvoca = pickle.load(f)

    for i in range(len(thisvoca)):
        vocab.append(thisvoca[i])
    filecnt += 1
    print("read", filecnt, "th file.")
    return vocab, filecnt


def generate_neg_sample():
    neg_choice = random.choices(unigram_words, weights=unigram_values, k=1000)
    neg_sample = queue.Queue()
    for i in range(1000):
        neg_sample.put(neg_choice[i])

    return neg_sample

def scalar_sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


def array_sigmoid(z):
    return 1 / (1 + np.exp(-z))


if __name__ == "__main__":

    with open("1billion-preprocessed_data.bin", "rb") as f:
        total_data = pickle.load(f)

    dictionary = total_data[0]
    count = total_data[1]
    reverse_dictionary = total_data[2]
    pwi = total_data[3]
    unigram_values = total_data[4]
    unigram_words = total_data[5]

    print("preprocessed_data loaded.")

    tot_count = sum(count.values())
    data_index = window_size()
    print("dic_size:", len(dictionary))
    dic_size = len(dictionary)

    # if method() == 0:
    #     word2vec = nsmodel_build(dic_size=dic_size)
    # else:
    #     maxhuff = 0
    #     codedict = HuffmanCoding().build(count)
    #     for i in codedict.keys():
    #         if len(codedict[i]) > maxhuff:
    #             maxhuff = len(codedict[i])
    #     print("maxhuff:", maxhuff)
    #     word2vec = hsmodel_build(dic_size=dic_size, codedict=codedict, revdic=reverse_dictionary, maxhuff=maxhuff)

    dic_keys = list(dictionary.keys())
    test_index = []
    cnt = 0
    # dic_words = np.random.choice(list(dictionary.keys()), batch_size())
    dic_words = ["one", "this", "as", "many", "who", "north", "funny", "fuck", "you", "sex"]

    for i in dic_words:
        test_index.append(dictionary[i])

    for i in range(batch_size() - len(dic_words)):
        test_index.append(0)

    if method() == 0:
        print("Negative Sampling with size", neg_num())
    else:
        print("Hierarchical Softmax")
    if subsampling() == True:
        print("with subsampling")

    vocab = deque()
    data = queue.Queue()
    filecnt = 0
    step = 0
    stepcheck = 5000
    savecheck = 10000000
    printcheck = 100000

    neg_sample = queue.Queue()
    average_loss = 0
    starting_alpha = learning_rate()
    alpha = learning_rate()
    # vocab = batch_test(vocab)
    imsi = 0
    epoch = 0

    embeddings = np.random.uniform(low=-0.5 / embedding_size(), high=0.5 / embedding_size(),
                                   size=(dic_size, embedding_size()))
    proj_weights = np.random.uniform(low=-0.5 / embedding_size(), high=0.5 / embedding_size(),
                                     size=(dic_size, embedding_size()))
    start_time = time.time()
    print("전체 단어수 : ", sum(count.values()))
    while True:
        step += 1
        if len(vocab) < batch_size() * 3:
            vocab, filecnt = datafeed(vocab, filecnt)

        if filecnt > 149:
            epoch += 1
            if epoch == epoch_num():
                print("training finished.")
                end_time = time.time()
                workingtime = (end_time - start_time) / 3600
                print("WorkingTime: {} sec".format(workingtime))
                with open("./model/embeddings_final", "wb+") as f:
                    pickle.dump(embeddings, f)
                break
            else:
                filecnt = 0

        if method() == 0:

            masklen = int(batch_size() * skip_prob())
            mask = np.random.choice(batch_size(), masklen)
            batch_inputs, batch_labels, vocab = generate_batch(pwi, None, vocab)
            neg_sample_index = [[0] for i in range(neg_num())]
            neg_cnt = 0

            # print("batch_inputs : ", batch_inputs)
            # print("batch_labels : ", batch_labels)
            # neg_sample_index = np.zeros(shape=(neg_num()))

            while neg_cnt < neg_num():
                if neg_sample.qsize() == 0:
                    neg_sample = generate_neg_sample()
                thisneg = neg_sample.get()
                if dictionary[thisneg] in batch_labels: continue
                neg_sample_index[neg_cnt] = dictionary[thisneg]
                neg_cnt += 1

            neu1e = np.zeros([masklen, embedding_size()])
            thisinput = batch_inputs[mask]
            thislabel = batch_labels[mask]

            # input 형식 만들기
            z = np.diagonal(np.dot(embeddings[thislabel], np.transpose(proj_weights[thisinput])))
            p = array_sigmoid(z)
            g = alpha * (1 - p)
            neu1e += np.dot(g, proj_weights[thisinput])
            g = np.reshape(g, [masklen, 1])
            proj_weights[thisinput] += np.multiply(g, embeddings[thislabel])

            # negative sample 계산
            neg_thislabel = [thislabel for x in range(neg_num())]
            neg_thislabel = np.transpose(np.array(neg_thislabel))
            for i in range(len(neg_thislabel)):
                z = np.diagonal(np.dot(embeddings[neg_thislabel[i]], np.transpose(proj_weights[neg_sample_index])))
                p = array_sigmoid(z)
                g = alpha * (-p)
                neu1e += np.dot(g, proj_weights[neg_sample_index])
                g = np.reshape(g, [neg_num(), 1])
                proj_weights[neg_sample_index] += np.multiply(g, embeddings[neg_thislabel[i]])

            # neg_thislabel = [thislabel for x in range(neg_num())]
            # z = np.dot(embeddings[neg_thislabel], np.transpose(proj_weights[neg_sample_index]))[0]
            # p = array_sigmoid(z)
            # g = alpha * (-p)
            # neu1e += np.dot(g, proj_weights[neg_sample_index])
            # g = np.reshape(g, [neg_num(), 1])
            # proj_weights[neg_sample_index] += np.multiply(g, embeddings[neg_thislabel])

            embeddings[thislabel] += neu1e
            average_loss += np.sum(neu1e)


        # else:
        #
        #     batch_inputs, batch_labels, path, sign, vocab = generate_batch(pwi, codedict, vocab)
        #
        #     feed_dict = {word2vec.train_inputs: batch_inputs,
        #                  word2vec.train_labels: batch_labels,
        #                  word2vec.path: path,
        #                  word2vec.sign: sign}

        if step % stepcheck == 0:
            if step > 0:
                average_loss /= stepcheck
                average_loss = abs(average_loss)
            print('Average loss at step {} : {}'.format(step, average_loss))

            average_loss = 0

        if step % printcheck == 0:

            # Recalculate alpha
            alpha = starting_alpha * (1 - float(step) / tot_count)
            if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001
            print("Current learning rate :", alpha)

            test_index = test_index[0:10]

            test_input = []
            # print(test_index)
            # print(np.shape(saveembed))
            for q in test_index:
                test_input.append(embeddings[q])

            sim = -cosine_similarity(test_input, embeddings)

            for i in range(10):
                print(dic_words[i], "와 가장 가까운 단어들 : ", end="")
                thissim = np.argsort(sim[i])[0:10]
                for i in thissim:
                    print(reverse_dictionary[i], end=", ")
                print()
            # progress?
            print((step * (batch_size() / (2 * window_size())) / sum(count.values())) * 100, "% finished.")
            if step % savecheck == 0:
                with open("./model/embeddings" + str(step), "wb+") as f:
                    pickle.dump(embeddings, f)
