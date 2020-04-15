import tensorflow as tf
import numpy as np
from collections import Counter
import collections


#### step 1 : 데이터 가공

def read_data(filename):
    with open(filename,'rt', encoding = 'UTF8' ) as f:
        result = f.read().split()
    return result


tokens = read_data('text8')





def make_dict(object):
    data = []
    voca_freq = Counter(object)
    voca_dict = {items : idx  for idx , items in enumerate(list(voca_freq))}
    voca_mostcommon = {voca_dict[k]:v for k ,v in voca_freq.items()}
    voca_mostcommon = Counter(voca_mostcommon).most_common()
    reversed_voca_dict = {value : key for key , value in voca_dict.items()}
    for token in object :
        data.append(voca_dict[token])

    return voca_freq , voca_dict , reversed_voca_dict , data , voca_mostcommon

voca_freq , voca_dict , reversed_voca_dict , data , voca_mostcommon = make_dict(tokens)


#### step 2 : 배치 데이터 생성

def generate_data(window, data):
    global data_index
    span = (window * 2) + 1
    num_skip = window * 2
    buffer = collections.deque(maxlen=span)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    label = np.ndarray(shape=(batch_size,1), dtype=np.int32)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index : data_index + span])
    data_index += span


    for i in range(batch_size // num_skip):
        context_words = [j for j in buffer if j != buffer[window]]
        for idx , context_word in enumerate(context_words):
            batch[i * num_skip + idx] = buffer[window]
            label[i * num_skip + idx , 0] = context_word
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    print(batch)

    return batch , label


data_index = 0
window = 2
batch_size = 16



#### step 3 : 학습을 위한 초기변수 설정

embedding_size = 300
mean = 0
std = 1
voca_size = len(voca_dict)

def init_net(voca_size , embedding_size):

    weight_0_shape = (voca_size , embedding_size)
    weight_1_shape = (voca_size , embedding_size)

    weight_0 = np.random.uniform(low = -0.5 / embedding_size, high = 0.5 / embedding_size, size = weight_0_shape)
    weight_1 = np.zeros(shape = weight_1_shape)

    return weight_0 , weight_1



# def lookup_0(matrix, object):
#     result = []
#     if type(object) == list:
#         for item in object:
#             result.append(matrix[item])
#     else :
#         result.append(matrix[object])
#         result = np.array(result)
#     return result



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





def loss_cross_entropy(t, y):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#### step 4 : skip-gram 모델

def skip_gram(batch_data , label_data):
    alpha = 0.01
    global weight_0
    loss = 0



    for index in range(len(batch_data)):
        sampling_list = negative_sampling(unigram_list)
        classifier = [(batch_data[index] , 1)] + [(k , 0) for k in sampling_list]
        grad_loss_0 = np.zeros(embedding_size)
        for target, answer in classifier:
            result1 = np.dot(weight_0[label_data[index]], weight_1[target])
            result2 = sigmoid(result1)
            if answer == 1:
                loss += loss_cross_entropy(answer,result2)
            grad_loss_1 = alpha * (answer - result2)
            grad_loss_0 += grad_loss_1 * weight_1[target]
            weight_1[target] += grad_loss_1 * weight_0[label_data[index][0]]

        weight_0[label_data[index]] += grad_loss_0
    return loss





#### step 5 : 학습

weight_0 , weight_1 = init_net(voca_size, embedding_size)

def train():

    num_process = 10000

    for num in range(num_process):
        batch_data, label_data = generate_data(window, data)
        loss = skip_gram(batch_data,label_data)
        print('%d번째 수행 loss = %d' %(num, loss))


train()


## 왜 context를 word와 negative samples랑 비교를 하나? , word를 context와 negative를 비교해야하는것 아닌가?
## sigmoid값이 다 0.5로 나옴 -- 초기값 조정? 배치사이즈가 작아서 학습이 느리나? 학습률 조정?
## reference 코드에서는 fi fo, pool map을 쓰는데 연산 속도에 영향?
## 트레이닝, 테스트 데이터 나눠서 시험




def __init_process(*args):
    global vocab, syn0, syn1, table, cbow, neg, dim, starting_alpha
    global win, num_processes, global_word_count, fi

    vocab, syn0_tmp, syn1_tmp, table, cbow, neg, dim, starting_alpha, win, num_processes, global_word_count = args[:-1]
    fi = open(args[-1], 'r')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)



def train(fi, fo, cbow, neg, dim, alpha, win, min_count, num_processes, binary):
    # Read train file to init vocab
    vocab = Vocab(fi, min_count)

    # Init net
    syn0, syn1 = init_net(dim, len(vocab))

    global_word_count = Value('i', 0)
    table = None
    if neg > 0:
        print
        'Initializing unigram table'
        table = UnigramTable(vocab)
    else:
        print
        'Initializing Huffman tree'
        vocab.encode_huffman()

    # Begin training using num_processes workers
    t0 = time.time()
    ## 프로세스 풀 객체
    pool = Pool(processes=num_processes, initializer=__init_process,
                initargs=(vocab, syn0, syn1, table, cbow, neg, dim, alpha,
                          win, num_processes, global_word_count, fi))
    pool.map(train_process, range(num_processes))
    t1 = time.time()
    print
    print
    'Completed training. Training took', (t1 - t0) / 60, 'minutes'

    # Save model to file
    save(vocab, syn0, fo, binary)


if __name__ == '__main__': ## 멀티 프로세싱 시작
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Training file', dest='fi', required=True)
    parser.add_argument('-model', help='Output model file', dest='fo', required=True)
    parser.add_argument('-cbow', help='1 for CBOW, 0 for skip-gram', dest='cbow', default=1, type=int)
    parser.add_argument('-negative',
                        help='Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax',
                        dest='neg', default=5, type=int)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=100, type=int)
    parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.025, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int)
    parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=5,
                        type=int)
    parser.add_argument('-processes', help='Number of processes', dest='num_processes', default=1, type=int)
    parser.add_argument('-binary', help='1 for output model in binary format, 0 otherwise', dest='binary', default=0,
                        type=int)
    # TO DO: parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)
    args = parser.parse_args()

    train(args.fi, args.fo, bool(args.cbow), args.neg, args.dim, args.alpha, args.win,
          args.min_count, args.num_processes, bool(args.binary))


