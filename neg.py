# import tensorflow as tf
# import numpy as np
# from collections import Counter
# import collections
# import time
# import random
# from matplotlib import pylab
# from sklearn.manifold import TSNE
# import pickle
# import os
# import math
# os.environ["CUDA_VISIBLE_DEVICES"]= '1'
#
#
# #### step 1 : 데이터 가공
#
# def read_data(filename):
#     with open(filename,'rt', encoding = 'UTF8' ) as f:
#         result = f.read().split()
#     return result
#
#
# tokens = read_data('text8')
#
#
# def cosine(a,b):
#     # deno_1 = math.sqrt(sum_of_squares(a))
#     # p1 = [i/deno_1 for i in a]
#     # deno_2 = math.sqrt(sum_of_squares(b))
#     # p2 = [i/deno_2 for i in b]
#     # result = np.dot(p1,p2)
#     result = np.dot(a,b)
#     return result
#
#
# def make_dict(object, n_words):
#     data = []
#     voca_freq = [['UNK',-1]]
#     unk_count = 0
#     voca_dict = {}
#     voca_freq.extend(collections.Counter(object).most_common(n_words-1))
#     for idx, item in enumerate(voca_freq):
#         voca_dict[item[0]] = idx
#     reversed_voca_dict = {value : key for key , value in voca_dict.items()}
#     for token in object:
#         index = voca_dict.get(token,0)
#         if index == 0 :
#             unk_count += 1
#         data.append(index)
#     voca_freq[0][1] = unk_count
#     voca_mostcommon = [(voca_dict[k] , v) for k,v in voca_freq]
#     return voca_freq , voca_dict , reversed_voca_dict , data , voca_mostcommon
#
# voca_freq , voca_dict , reversed_voca_dict , data , voca_mostcommon = make_dict(tokens, n_words=50000)
#
# def test_dataset(filename,y2):
#     testfile = open(filename, 'r')
#     data = []
#     analogy_voca = []
#     print('Processing analogy dataset')
#     while True:
#         line = testfile.readline()
#         if line == '':
#             break
#         testline = line.split()
#         for idx, factor in enumerate(testline):
#             if factor not in y2:
#                 break
#             elif idx + 1 == len(testline):
#                 data.append(testline)
#             analogy_voca.append(factor)
#     analogy_voca = list(set(analogy_voca))
#     return data, analogy_voca
#
#
# vocab_list = list(voca_dict.keys())
# analogy_data, analogy_voca = test_dataset('analogy task.txt', vocab_list)
#
# del tokens
#
#
# test_index = 0
# test_size = 1000
# def test_analogy(norm_y1, y2, analogy_data):
#     global test_index
#     analogy_voca_vector = []
#     analogy_voca_index = []
#     for voca in analogy_voca:
#         voca_index = y2.index(voca)
#         voca_vector = norm_y1[voca_index]
#         analogy_voca_vector.append(voca_vector)
#         analogy_voca_index.append(voca_index)
#
#     target_list = []
#     prediction = 0
#     if test_index+test_size > len(analogy_data):
#         test_index = 0
#     batch = analogy_data[test_index : test_index + test_size]
#     for idx,factor in enumerate(batch):
#         target = norm_y1[y2.index(batch[idx][1])] - norm_y1[y2.index(batch[idx][0])] + norm_y1[y2.index(batch[idx][2])]
#         target_list.append(target)
#     print('analogy test Implementation')
#     for num, target_token in enumerate(target_list):
#         pred_list = {}
#         for i, vector in enumerate(analogy_voca_vector):
#             cosine_value = cosine(target_token, vector)
#             pred_list[analogy_voca_index[i]] = math.sqrt(cosine_value**2)
#         sorted_x = sorted(pred_list.items(), key=lambda kv: kv[1])
#         if sorted_x[-1][0] == y2.index(batch[num][3]) or sorted_x[-2][0] == y2.index(batch[num][3]) or sorted_x[-3][0] == y2.index(batch[num][3]) or \
#                 sorted_x[-4][0] == y2.index(batch[num][3]):
#             prediction += 1
#         if num == 0:
#             print(reversed_voca_dict[sorted_x[-1][0]],':',sorted_x[-1][1],reversed_voca_dict[sorted_x[-2][0]],':',sorted_x[-2][1],
#                   reversed_voca_dict[sorted_x[-3][0]],':',sorted_x[-3][1],reversed_voca_dict[sorted_x[-4][0]],':',sorted_x[-4][1],
#                   reversed_voca_dict[sorted_x[-5][0]],':',sorted_x[-5][1],reversed_voca_dict[sorted_x[-6][0]],':',sorted_x[-6][1])
#     target_vector = norm_y1[y2.index(batch[0][1])] - norm_y1[y2.index(batch[0][0])] + norm_y1[y2.index(batch[0][2])]
#     score = (prediction / len(target_list))*100
#     test_index += test_size
#     print(batch[0][0], batch[0][1], batch[0][2], batch[0][3], 'target :', cosine(target_vector,norm_y1[y2.index(batch[0][3])]))
#     return score
#
#
#
# def build_dataset(words, n_words):
#   """Process raw inputs into a dataset."""
#   count = [['UNK', -1]]
#   count.extend(collections.Counter(words).most_common(n_words - 1))
#   dictionary = dict()
#   for word, _ in count:
#     dictionary[word] = len(dictionary)
#   data = list()
#   unk_count = 0
#   for word in words:
#     index = dictionary.get(word, 0)
#     if index == 0:  # dictionary['UNK']
#       unk_count += 1
#     data.append(index)
#   count[0][1] = unk_count
#   reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
#   return data, count, dictionary, reversed_dictionary
#
#
# ## step 3 : subsampling 구현
# def subsampling(object_data):
#     threshold = 1e-5
#     global voca_freq # voca_freq == law_word_counts
#     voca_total_count = len(object_data)
#     voca_prob = {word: count/voca_total_count for word, count in voca_mostcommon}
#     prob = {word: 1-np.sqrt(threshold/voca_prob[word]) for word, _ in voca_mostcommon}
#     voca_sampled = [word for word in object_data if random.random() < (1-prob[word])]
#     return voca_sampled
#
#
#
#
# #### step 2 : 배치 데이터 생성
#
# def generate_data(window_size, data):
#     global data_index
#     num_skip = window_size * 2
#     span = (window_size * 2) + 1
#     buffer = collections.deque(maxlen=span)
#     batch = np.ndarray(shape=(batch_size), dtype=np.int32)
#     label = np.ndarray(shape=(batch_size,1), dtype=np.int32)
#     if data_index + span > len(data):
#         data_index = 0
#     buffer.extend(data[data_index : data_index + span])
#     data_index += span
#
#
#     for i in range(batch_size // num_skip):
#         context_words = [j for j in buffer if j != buffer[window_size]]
#         for idx , context_word in enumerate(context_words):
#             batch[i * num_skip + idx] = buffer[window_size]
#             label[i * num_skip + idx , 0] = context_word
#         if data_index == len(data):
#             buffer.extend(data[0:span])
#             data_index = span
#         else:
#             buffer.append(data[data_index])
#             data_index += 1
#     data_index = (data_index + len(data) - span) % len(data)
#     return batch , label
#
#
# data_index=0
#
#
#
#
#
# #### step 3 : 학습을 위한 초기변수 설정
#
#
# mean = 0
# std = 1
# voca_size = len(voca_dict)
#
# def init_net(voca_size , embedding_size):
#
#     weight_0_shape = (voca_size , embedding_size)
#     weight_1_shape = (voca_size , embedding_size)
#
#     weight_0 = np.random.uniform(low = -0.5 / embedding_size, high = 0.5 / embedding_size, size = weight_0_shape)
#     weight_1 = np.zeros(shape = weight_1_shape)
#
#     return weight_0 , weight_1
#
#
#
#
# def make_unigram():
#     unigram_dict = {}
#     for voca in voca_mostcommon:
#         unigram_dict[voca[0]]=int(np.ceil(voca[1]**(3/4)))
#     unigram_list = []
#     for object in unigram_dict:
#         for _ in range(unigram_dict[object]):
#             unigram_list.append(object)
#     return unigram_list
#
# unigram_list = make_unigram()
#
#
#
# def negative_sampling(list):
#     sampling_size = 5
#     sampling_list = np.random.choice(list , sampling_size)
#     return sampling_list
#
#
#
# #### step 5 : 학습모델 빌드
#
#
#
# batch_size = 60
# epochs = 10
# # batch_size = 16
# # window_size = 1
# embedding_size = 300
#
# # nearest 뽑는 것
# # valid_examples = np.array(random.sample(range(valid_window), valid_size))
#
# #인풋 레이어를 만듭니다.
# train_graph = tf.Graph()
# with train_graph.as_default():
#     with tf.name_scope('inputs'):
#         inputs = tf.placeholder(tf.int32, [batch_size], name='inputs')
#         labels = tf.placeholder(tf.int32, [batch_size,1], name='labels')
#         # valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
#
# #임베딩 레이어를 만듭니다.
#
#     with tf.device('/cpu:0'):
#         with tf.name_scope('embeddings'):
#             embedding = tf.Variable(tf.random_uniform((voca_size, embedding_size), -1, 1))
#             embed = tf.nn.embedding_lookup(embedding, inputs)
#         with tf.name_scope('weights'):
#             softmax_w = tf.Variable(tf.truncated_normal((voca_size, embedding_size), stddev=0.05))# create softmax weight matrix here
#         with tf.name_scope('biases'):
#             softmax_b = tf.Variable(tf.zeros(voca_size))# create softmax biases here
#
#
#     #Negative sampling
#
#     with tf.name_scope('loss'):
#     # softmax_w = tf.Variable(tf.truncated_normal((voca_size, embedding_size), stddev=0.05))# create softmax weight matrix here
#     # softmax_b = tf.Variable(tf.zeros(voca_size))# create softmax biases here
#         n_sampled = 5
#         loss = tf.nn.sampled_softmax_loss(weights = softmax_w, biases=softmax_b,
#                                           labels=labels, inputs=embed,
#                                           num_sampled=n_sampled, num_classes=voca_size)
#         cost = tf.reduce_mean(loss)
#
#
#     with tf.name_scope('optimizer'):
#
#         optimizer = tf.train.AdamOptimizer().minimize(cost)
#
#     # norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
#     # normalized_embedding = embedding / norm
#
#
#     save_file = './model.ckpt'
#     saver = tf.train.Saver()
#
#
#
#
# ### step 6 : 학습 시작
#
#
# sess=tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True)),graph=train_graph)
#
# with sess:
#
#     sess.run(tf.global_variables_initializer())
#     # global data_index
#     iteration = 1
#     loss = 0
#     data_index2 = 0
#     # iteration으로 해야하나? #interation으로 할 경우, data index를 0으로?
#
#     for e in range(1, epochs+1):
#         if e == 1:
#             window_size = 5
#         if e == 2:
#             window_size = 4
#         if e == 3:
#             window_size = 3
#         if e == 4:
#             window_size = 2
#         if e == 5:
#             window_size = 1
#         if e >= 6:
#             window_size = random.randint(1,5)
#         num_skip = window_size * 2
#         voca_sampled = subsampling(data)
#         start = time.time()
#         len('배치 사이즈는 %d' %len(voca_sampled))
#         num_step = int(len(voca_sampled)//(batch_size/num_skip))
#         print('num_step : %d'%num_step)
#         print('win_size',window_size)
#         for index in range(num_step):  ## 코드에서는 배치랑 라벨데이터를 한번에 뽑아놨음.
#             batch_data, label_data = generate_data(window_size, data=voca_sampled)
#             feed = {inputs: batch_data,
#                     labels: label_data}
#             train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
#             print(iteration)
#             print(train_loss)
#             loss += train_loss
#
#             if iteration % 1000 == 0:
#                 end = time.time()
#                 print("Epoch {}/{}".format(e, epochs),
#                       "Iteration: {}".format(iteration),
#                       "Avg. Training loss: {:.4f}".format(loss/100),
#                       "{:.4f} sec/batch".format((end-start)/100),
#                       "data index : %d"%data_index)
#                 loss = 0
#                 start = time.time()
#
#
#             if iteration % 20000 == 0:
#                 norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
#                 normalized_embedding = embedding / norm
#                 test_embedding = normalized_embedding.eval()
#                 print(test_analogy(test_embedding, vocab_list, analogy_data), '%')
#
#
#             iteration += 1
#
#         saver.save(sess,save_file)
#
#
#
#     final_embeddings = normalized_embedding.eval()
#
#
#
#
# num_points = 400
#
# tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
# two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
#
# def plot(embedding, labels):
#   assert embedding.shape[0] >= len(labels), 'More labels than embeddings'
#   pylab.figure(figsize=(15,15))  # in inches
#   for i, label in enumerate(labels):
#     x, y = embedding[i, :]
#     pylab.scatter(x, y)
#     pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
#                    ha='right', va='bottom')
#   pylab.show()
#
# words = [reversed_voca_dict[i] for i in range(1, num_points+1)]
# plot(two_d_embeddings, words)
#
#
#
#
# ## import variable
# with tf.Session() as sess:
#   new_saver = tf.train.import_meta_graph('model.ckpt.meta')
#   new_saver.restore(sess, tf.train.latest_checkpoint('./'))
#
#
#

import pickle

with open('./model/first_syn0.bin', 'rb') as f1:
    syn0 = pickle.load(f1)
with open('./model/first_syn1.bin','rb') as f2:
    syn1 = pickle.load(f2)
print(syn0[0])
print('----------------------------------------------------------------------------------------------------')
print(syn1[0])