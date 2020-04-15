import tensorflow as tf
import numpy as np
from collections import Counter
import collections
import time
import random
from matplotlib import pylab
from sklearn.manifold import TSNE
import pickle


#### step 1 : 데이터 가공

def read_data(filename):
    with open(filename,'rt', encoding = 'UTF8' ) as f:
        result = f.read().split()
    return result


tokens = read_data('text8')





def make_dict(object, n_words):
    data = []
    voca_freq = [['UNK',-1]]
    unk_count = 0
    voca_dict = {}
    voca_freq.extend(collections.Counter(object).most_common(n_words-1))
    for idx, item in enumerate(voca_freq):
        voca_dict[item[0]] = idx
    reversed_voca_dict = {value : key for key , value in voca_dict.items()}
    for token in object:
        index = voca_dict.get(token,0)
        if index == 0 :
            unk_count += 1
        data.append(index)
    voca_freq[0][1] = unk_count
    voca_mostcommon = [(voca_dict[k] , v) for k,v in voca_freq]
    return voca_freq , voca_dict , reversed_voca_dict , data , voca_mostcommon

voca_freq , voca_dict , reversed_voca_dict , data , voca_mostcommon = make_dict(tokens, n_words=50000)
del tokens

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary


## step 3 : subsampling 구현
def subsampling(object_data):
    threshold = 1e-5
    global voca_freq # voca_freq == law_word_counts
    voca_total_count = len(object_data)
    voca_prob = {word: count/voca_total_count for word, count in voca_mostcommon}
    prob = {word: 1-np.sqrt(threshold/voca_prob[word]) for word, _ in voca_mostcommon}
    voca_sampled = [word for word in object_data if random.random() < (1-prob[word])]
    return voca_sampled




#### step 2 : 배치 데이터 생성

def generate_data(window_size, data):
    global data_index
    span = (window_size * 2) + 1
    buffer = collections.deque(maxlen=span)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    label = np.ndarray(shape=(batch_size,1), dtype=np.int32)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index : data_index + span])
    data_index += span


    for i in range(batch_size // num_skip):
        context_words = [j for j in buffer if j != buffer[window_size]]
        for idx , context_word in enumerate(context_words):
            batch[i * num_skip + idx] = buffer[window_size]
            label[i * num_skip + idx , 0] = context_word
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    data_index = (data_index + len(data) - span) % len(data)
    return batch , label


data_index=0





#### step 3 : 학습을 위한 초기변수 설정


mean = 0
std = 1
voca_size = len(voca_dict)

def init_net(voca_size , embedding_size):

    weight_0_shape = (voca_size , embedding_size)
    weight_1_shape = (voca_size , embedding_size)

    syn0 = np.random.uniform(low = -0.5 / embedding_size, high = 0.5 / embedding_size, size = weight_0_shape)
    syn1 = np.zeros(shape = weight_1_shape)

    return syn0 , syn1




def make_unigram():
    unigram_dict = {}
    for voca in voca_mostcommon:
        unigram_dict[voca[0]]=int(np.ceil(voca[1]**(3/4)))
    unigram_list = []
    for object in unigram_dict:
        for _ in range(unigram_dict[object]):
            unigram_list.append(object)
    return unigram_list

unigram_list = make_unigram()



def negative_sampling(list):
    sampling_size = 5
    sampling_list = np.random.choice(list , sampling_size)
    return sampling_list





#### step 5 : 학습모델 빌드


def train_process(pid):


    alpha = 0.025

    word_count = 0
    last_word_count = 0
    loss = 0
            # CBOW
            if cbow:
                # Compute neu1
                neu1 = np.mean(np.array([syn0[c] for c in context]), axis=0)
                assert len(neu1) == dim, 'neu1 and dim do not agree'

                # Init neu1e with zeros
                neu1e = np.zeros(dim)

                # Compute neu1e and update syn1
                if neg > 0:
                    classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                else:
                    classifiers = zip(vocab[token].path, vocab[token].code)
                for target, label in classifiers:
                    z = np.dot(neu1, syn1[target])
                    p = sigmoid(z)
                    g = alpha * (label - p)
                    loss += g
                    neu1e += g * syn1[target]  # Error to backpropagate to syn0
                    syn1[target] += g * neu1  # Update syn1

                # Update syn0
                for context_word in context:
                    syn0[context_word] += neu1e

            # Skip-gram
            else:
                for context_word in context: # context_word가 label_data, label data를 하나 꺼냄
                    # Init neu1e with zeros
                    neu1e = np.zeros(dim)

                    # Compute neu1e and update syn1
                    if neg > 0:
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)] # token은 중심단어 , 즉 batch_data, target은 neg sample , 중심단어와 neg_sample를 같이 묶어서 classifier로 만듬
                    else:
                        classifiers = zip(vocab[token].path, vocab[token].code)
                    for target, label in classifiers:
                        z = np.dot(syn0[context_word], syn1[target]) # 주변단어(label data)의 syn0 벡터값과 중심단어 및 neg sample의 syn1 벡터값을 내적
                        p = sigmoid(z) # 그 값을 시그모이드값으로
                        g = alpha * (label - p) # 교차엔트로피 미분값 , 여기서 alpha는 학습률
                        loss += g
                        neu1e += g * syn1[target]  # Error to backpropagate to syn0 # 손실함수 미분값에 neg sample의 syn1 매개변수를 곱한 값이
                        syn1[target] += g * syn0[context_word]  # Update syn1 #

                    # Update syn0
                    syn0[context_word] += neu1e

            word_count += 1
            if word_count % 10000 == 0 :
                print('loss 값은 :', loss)
                loss = 0

    # Print progress info
    global_word_count.value += (word_count - last_word_count)
    sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                     (alpha, global_word_count.value, vocab.word_count,
                      float(global_word_count.value) / vocab.word_count * 100))
    sys.stdout.flush()
    fi.close()


window_size = 1 ## span이 윈도우 사이즈??
num_skip = window_size*2
batch_size = 128
epochs = 5
# batch_size = 16
# window_size = 1
data_index = 0
voca_size = len(voca_dict)
embedding_size = 64





### step 6 : 학습 시작




with tf.Session(graph=train_graph) as sess:

    sess.run(tf.global_variables_initializer())
    global data_index
    iteration = 1
    loss = 0
    data_index = 0
    # iteration으로 해야하나? #interation으로 할 경우, data index를 0으로?

    for e in range(1, epochs+1):
        voca_sampled = subsampling(data)
        start = time.time()
        len('배치 사이즈는 %d' %len(voca_sampled))
        num_step = int(len(voca_sampled)//(batch_size/num_skip))
        print('num_step : %d'%num_step)
        for index in range(num_step):  ## 코드에서는 배치랑 라벨데이터를 한번에 뽑아놨음.
            batch_data, label_data = generate_data(window_size, data=voca_sampled)
            train_process(pid)

            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += train_loss

            if iteration % 1000 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss/100),
                      "{:.4f} sec/batch".format((end-start)/100),
                      "data index : %d"%data_index)
                loss = 0
                start = time.time()


            iteration += 1

        saver.save(sess,save_file)



    final_embeddings = normalized_embedding.eval()




num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])

def plot(embedding, labels):
  assert embedding.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embedding[i, :]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()

words = [reversed_voca_dict[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)


