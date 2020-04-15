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

    weight_0 = np.random.uniform(low = -0.5 / embedding_size, high = 0.5 / embedding_size, size = weight_0_shape)
    weight_1 = np.zeros(shape = weight_1_shape)

    return weight_0 , weight_1




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

###



###

#### step 5 : 학습모델 빌드


window_size = 1 ## span이 윈도우 사이즈??
num_skip = window_size*2
batch_size = 128
epochs = 5
# batch_size = 16
# window_size = 1
data_index = 0
voca_size = len(voca_dict)
embedding_size = 64

# nearest 뽑는 것
# valid_examples = np.array(random.sample(range(valid_window), valid_size))

#인풋 레이어를 만듭니다.
train_graph = tf.Graph()
with train_graph.as_default():
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [batch_size], name='inputs')
        labels = tf.placeholder(tf.int32, [batch_size,1], name='labels')
        # valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

#임베딩 레이어를 만듭니다.

    with tf.device('/cpu:0'):
        with tf.name_scope('embeddings'):
            embedding = tf.Variable(tf.random_uniform((voca_size, embedding_size), -1, 1))
            embed = tf.nn.embedding_lookup(embedding, inputs)
        with tf.name_scope('weights'):
            softmax_w = tf.Variable(tf.truncated_normal((voca_size, embedding_size), stddev=0.05))# create softmax weight matrix here
        with tf.name_scope('biases'):
            softmax_b = tf.Variable(tf.zeros(voca_size))# create softmax biases here


    #Negative sampling

    with tf.name_scope('loss'):
        """Builds the loss for hierarchical softmax.
        Args:
          inputs: int tensor of shape [batch_size] (skip_gram) or
            [batch_size, 2*window_size+1] (cbow)
          labels: int tensor of shape [batch_size, 2*max_depth+1]
          weigh_0: float tensor of shape [vocab_size, embed_size], input word
            embeddings (i.e. weights of hidden layer).
          syn1: float tensor of shape [syn1_rows, embed_size], output word
            embeddings (i.e. weights of output layer).
          biases: float tensor of shape [syn1_rows], biases added onto the logits.
        Returns:
          loss: float tensor of shape [sum_of_code_len]
        """
        inputs_list = batch_data
        inputs_syn0_list = tf.unstack(tf.gather(embedding, inputs_list)) # lookup이랑 뭐가 다른거지?
        codes_points_list = tf.unstack(labels) #여기서 label은 code label인듯한데 0,1같은
        max_depth = (labels.shape.as_list()[1] - 1) // 2
        loss = []
        for inputs_syn0, codes_points in zip(inputs_syn0_list, codes_points_list):
            true_size = codes_points[-1] #맨마지막, 즉 본인 코드
            codes = codes_points[:true_size] #부모코드들
            points = codes_points[max_depth:max_depth + true_size]

            logits = tf.reduce_sum(
                tf.multiply(inputs_syn0, tf.gather(syn1, points)), 1)

            loss.append(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.to_float(codes), logits=logits))
        loss = tf.concat(loss, axis=0)


    with tf.name_scope('optimizer'):

        optimizer = tf.train.AdamOptimizer().minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
    normalized_embedding = embedding / norm

    # valid_embeddings = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    # similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embedding))
    # embed_mat = sess.run(normalized_embedding)

    save_file = './model.ckpt'
    saver = tf.train.Saver()




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
            feed = {inputs: batch_data,
                    labels: label_data}
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


