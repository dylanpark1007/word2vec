import argparse
import math
import struct
import sys
import time
import warnings
import numpy as np
from multiprocessing import Pool, Value, Array
from matplotlib import pylab
from sklearn.manifold import TSNE
import random
import pickle



class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None  # Path (list of indices) from the root to the word (leaf)
        self.code = None  # Huffman encoding


class Vocab:
    def __init__(self, fi, min_count):
        vocab_items = []
        vocab_hash = {}
        word_count = 0
        fi = open('text8', 'r') ## fi는 파일이름, 파일이름을 담은 변수

        # Add special tokens <bol> (beginning of line) and <eol> (end of line)
        for token in ['<bol>', '<eol>']:
            vocab_hash[token] = len(vocab_items)
            vocab_items.append(VocabItem(token)) ## class '<bol>', '<eol>'을 vocab_items에 넣어놨음

        for line in fi:
            tokens = line.split()
            for token in tokens:
                if token not in vocab_hash:
                    vocab_hash[token] = len(vocab_items) # vocab_hash는 {단어 : index} 형태의 단어사전
                    vocab_items.append(VocabItem(token))  # vocab_items는 VocabItem class에 가입된 단어들을 입력순서대로 쭉 나열한 리스트(중복x) -> 단어사전을 만들기 위한것

                # assert vocab_items[vocab_hash[token]].word == token, 'Wrong vocab_hash index'
                vocab_items[vocab_hash[token]].count += 1 # vocab_hash에 있는 단어들 각각의 count를 세어줌(해당 단어가 corpus에 몇개나오는지)
                word_count += 1 # 총 단어의 개수를 세어줌

                if word_count % 10000 == 0:
                    sys.stdout.write("\rReading word %d" % word_count) # sys.stdout.write는 프린트를 하고, str개수를 반환함 / flush는 버퍼를 비워서 불필요한 메모리를 확보
                    sys.stdout.flush()

            # Add special tokens <bol> (beginning of line) and <eol> (end of line)
            vocab_items[vocab_hash['<bol>']].count += 1 # 각 tokens(=line)이 끝나면 bol과 eol을 각각 카운트해줌, 즉 line 개수를 의미
            vocab_items[vocab_hash['<eol>']].count += 1
            word_count += 2

        self.bytes = fi.tell() # fi.tell은 파일의 현재 위치를 정수로 가리키는 함수
        self.vocab_items = vocab_items  # List of VocabItem objects ## 앞에서 정리한 데이터들을 self로 저장
        self.vocab_hash = vocab_hash  # Mapping from each token to its index in vocab
        self.re_vocab_hash = dict((v, k) for k, v in vocab_hash.items())
        self.word_count = word_count  # Total number of words in train file

        # Add special token <unk> (unknown),
        # merge words occurring less than min_count into <unk>, and
        # sort vocab in descending order by frequency in train file
        self.__sort(min_count)

        # assert self.word_count == sum([t.count for t in self.vocab_items]), 'word_count and sum of t.count do not agree'
        print('Total words in training file: %d' % self.word_count)
        print('Total bytes in training file: %d' % self.bytes)
        print('Vocab size: %d' % len(self))

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash


    def __sort(self, min_count):
        tmp = []
        tmp.append(VocabItem('<unk>')) # tmp에 unk 하나 넣어주고
        unk_hash = 0

        count_unk = 0
        for token in self.vocab_items: #voab_items 리스트에서 단어 하나씩 꺼내서 그 단어의 count가 내가 정한 min_count보다 작으면 unk count(unk가 몇 종류인지)를 하나씩 올려줌, unk_hash는 unk가 총 몇개 들어있는지
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token) # unk가 아닌 애들은 tmp에 넣어줌

        tmp.sort(key=lambda token: token.count, reverse=True) # token count 순서로 내림차순 정렬 ,  tmp는 unk이 없는 리스트

        # Update vocab_hash
        vocab_hash = {}
        for i, token in enumerate(tmp):
            vocab_hash[token.word] = i

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash
        self.re_vocab_hash = dict((v, k) for k, v in vocab_hash.items())

        print('Unknown vocab size:', count_unk)

    def indices(self, tokens): ## vocab_hash를 이용해서 해당 voca의 index반환
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]

    def return_vocab(self, tokens):
        return [self.re_vocab_hash[token] for token in tokens]


    def encode_huffman(self):
        # Build a Huffman tree
        vocab_size = len(self)
        count = [t.count for t in self] + [1e15] * (vocab_size - 1)
        parent = [0] * (2 * vocab_size - 2)
        binary = [0] * (2 * vocab_size - 2)

        pos1 = vocab_size - 1
        pos2 = vocab_size

        for i in range(vocab_size - 1):
            # Find min1
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1 = pos1
                    pos1 -= 1
                else:
                    min1 = pos2
                    pos2 += 1
            else:
                min1 = pos2
                pos2 += 1

            # Find min2
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2 = pos1
                    pos1 -= 1
                else:
                    min2 = pos2
                    pos2 += 1
            else:
                min2 = pos2
                pos2 += 1

            count[vocab_size + i] = count[min1] + count[min2]
            parent[min1] = vocab_size + i
            parent[min2] = vocab_size + i
            binary[min2] = 1



        # Assign binary code and path pointers to each vocab word
        root_idx = 2 * vocab_size - 2
        for i, token in enumerate(self):
            path = []  # List of indices from the leaf to the root
            code = []  # Binary Huffman encoding from the leaf to the root

            node_idx = i
            while node_idx < root_idx:
                if node_idx >= vocab_size: path.append(node_idx)
                code.append(binary[node_idx])
                node_idx = parent[node_idx]
            path.append(root_idx)

            # These are path and code from the root to the leaf
            token.path = [j - vocab_size for j in path[::-1]]
            token.code = code[::-1]





def readfile(filename):
    vector_file = open(filename, 'r')
    embedding = []
    label = []
    while True:
        line = vector_file.readline()
        if line == '':
            break
        label_token = line.split(' ', 1)[0]
        label.append(label_token)
        embedding_token = line.split(' ', 1)[1]
        embedding_token = embedding_token.split(' ')
        embedding_token = [float(x) for x in embedding_token]
        embedding.append(embedding_token)
    del embedding[0]
    del label[0]
    embedding = np.array(embedding)

    return embedding,label






def test_dataset(filename,y2):
    testfile = open(filename, 'r')
    data = []
    analogy_voca = []
    print('Processing analogy dataset')
    while True:
        line = testfile.readline()
        if line == '':
            break
        testline = line.split()
        for idx, factor in enumerate(testline):
            if factor not in y2:
                break
            elif idx + 1 == len(testline):
                data.append(testline)
            analogy_voca.append(factor)
    analogy_voca = list(set(analogy_voca))
    return data, analogy_voca




def sum_of_squares(n):
    return sum([i**2 for i in n])

def cosine(a,b):
    # deno_1 = math.sqrt(sum_of_squares(a))
    # p1 = [i/deno_1 for i in a]
    # deno_2 = math.sqrt(sum_of_squares(b))
    # p2 = [i/deno_2 for i in b]
    # result = np.dot(p1,p2)
    result = np.dot(a,b)
    return result



def make_norm_embedding(embedding):
    norm_embedding = []
    for embed in embedding:
        deno_1 = math.sqrt(sum_of_squares(embed))
        norm_embed = [i/deno_1 for i in embed]
        norm_embedding.append(norm_embed)
    norm_embedding = np.array(norm_embedding)
    return norm_embedding



index = 0
def test_analogy(norm_y1, y2, analogy_data, analogy_voca):
    global index
    analogy_voca_vector = []
    analogy_voca_index = []
    for voca in analogy_voca:
        voca_index = y2.index(voca)
        voca_vector = norm_y1[voca_index]
        analogy_voca_vector.append(voca_vector)
        analogy_voca_index.append(voca_index)

    target_list = []
    prediction = 0
    if index+300 > len(analogy_data):
        index = 0
    batch = analogy_data[index : index + 300]
    for idx,factor in enumerate(batch):
        target = norm_y1[y2.index(batch[idx][1])] - norm_y1[y2.index(batch[idx][0])] + norm_y1[y2.index(batch[idx][2])]
        target_list.append(target)
    print('analogy test Implementation')
    for num, target_token in enumerate(target_list):
        pred_list = {}
        for i, vector in enumerate(analogy_voca_vector):
            cosine_value = cosine(target_token, vector)
            pred_list[analogy_voca_index[i]] = math.sqrt(cosine_value**2)
        sorted_x = sorted(pred_list.items(), key=lambda kv: kv[1])
        if sorted_x[-1][0] == y2.index(batch[num][3]) or sorted_x[-2][0] == y2.index(batch[num][3]) or sorted_x[-3][0] == y2.index(batch[num][3]) or \
                sorted_x[-4][0] == y2.index(batch[num][3]):
            prediction += 1
        if num == 0:
            print(vocab.re_vocab_hash[sorted_x[-1][0]],':',sorted_x[-1][1],vocab.re_vocab_hash[sorted_x[-2][0]],':',sorted_x[-2][1],
                  vocab.re_vocab_hash[sorted_x[-3][0]],':',sorted_x[-3][1],vocab.re_vocab_hash[sorted_x[-4][0]],':',sorted_x[-4][1],
                  vocab.re_vocab_hash[sorted_x[-5][0]],':',sorted_x[-5][1],vocab.re_vocab_hash[sorted_x[-6][0]],':',sorted_x[-6][1])
    target_vector = norm_y1[y2.index(batch[0][1])] - norm_y1[y2.index(batch[0][0])] + norm_y1[y2.index(batch[0][2])]
    score = (prediction / len(batch))*100
    index += 300
    print(batch[0][0], batch[0][1], batch[0][2], batch[0][3], 'target :', cosine(target_vector,norm_y1[y2.index(batch[0][3])]))
    return score




def make_tsne(filename):
    tsne_file = open(filename, 'r')
    embedding = []
    label = []
    while True:
        line = tsne_file.readline()
        if line == '':
            break
        label_token = line.split(' ', 1)[0]
        label.append(label_token)
        embedding_token = line.split(' ', 1)[1]
        embedding_token = embedding_token.split(' ')
        embedding_token = [float(x) for x in embedding_token]
        embedding.append(embedding_token)
    del embedding[0]
    del label[0]
    embedding = np.array(embedding)

    for index, vector in enumerate(embedding):
        deno_1 = math.sqrt(sum_of_squares(embedding[index]))
        embedding[index] = [k / deno_1 for k in embedding[index]]
    return embedding,label


def print_tsne(embedding,label):
    num_points = 400

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    two_d_embeddings = tsne.fit_transform(embedding[1:num_points + 1, :])
    words = [label[i] for i in range(1, num_points + 1)]
    return two_d_embeddings, words


def plot(embedding, labels):
    assert embedding.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15,15))  # in inches
    for i, label in enumerate(labels):
        x, y = embedding[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
    pylab.show()


class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """

    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab])  # Normalizing constant # 해당 token(단어)의 count 개수를 3/4 승해서 다 더해줌.

        table_size = np.uint32(1e3)  # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        print('Filling unigram table')
        p = 0  # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(unigram.count, power)) / norm # 각 단어들의 count의 3/4승해서 다 더해준 값(norm)을 각 값으로 나눠주어 p값을 구함.
            while i < table_size and float(i) / table_size < p: # 사이즈가 10의 8승인 테이블 사이즈에 p 분포만큼 해당 단어의 인덱스 개수만큼 맵핑시킴
                table[i] = j
                i += 1
        self.table = table # 즉 테이블은 단어 각각의 p 확률분포만큼 사이즈가 10의 8승인 테이블에 인덱스를 맵핑시키는 것임.
        print(p)

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count) # 0에서 10의 8승 사이의 숫자들 중 원하는 사이즈만큼 랜덤하게 추출
        return [self.table[i] for i in indices] # 추출된 0~10의 8승 사이의 숫자들을 테이블에서 단어 인덱스로 반환


def sigmoid(z): # 시그모이드
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


def init_net(dim, vocab_size):
    # Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
    # syn0 = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(vocab_size, dim)) # 사이즈가 V X N이고 각 값이 -0.5/N 에서 +0.5/N사이의 값들을 random uniform하게 테이블로 생성


    # syn0 = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size, dim))  # 사이즈가 V X N이고 각 값이 -0.5/N 에서 +0.5/N사이의 값들을 random uniform하게 테이블로 생성
    tmp = np.random.uniform(low=-0.5, high=0.5, size=(vocab_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    # Init syn1 with zeros
    # syn1 = np.zeros(shape=(vocab_size, dim)) # syn0과 같으나 첨엔 zero를 입힘
    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)

    return (syn0, syn1)


def subsampling(object_data):
    threshold = 1e-5
    global voca_freq # voca_freq == law_word_counts
    voca_total_count = len(object_data)
    voca_prob = {word: vocab.vocab_items[word].count/voca_total_count for word in vocab.vocab_hash.values()}
    prob = {word: 1-np.sqrt(threshold/voca_prob[word]) for word in vocab.vocab_hash.values()}
    voca_sampled = [word for word in object_data if random.random() < (1-prob[word])]
    return voca_sampled



def train_process(pid): #pid는 range(num_processes)
    # Set fi to point to the right chunk of training file
    vocab_list = vocab.vocab_items
    vocab_list = [i.word for i in vocab_list]
    analogy_data, analogy_voca = test_dataset('analogy task.txt', vocab_list)
    print('analogy done')
    epoch_count = 0
    num_epoch = 25

    while epoch_count < num_epoch :
        start = vocab.bytes / num_processes * pid  # byte는 현재 위치를 가르키는 정수 , 그거에 연산 횟수를 나눠주고 다시 pid를 곱해줌, 그 값을 start로
        end = vocab.bytes if pid == num_processes - 1 else vocab.bytes / num_processes * (
                    pid + 1)  # end는 pid가 연산횟수 -1과 같아질때 end값 아니라면 else 뒤의 값
        fi.seek(start)
        # print 'Worker %d beginning training at %d, ending at %d' % (pid, start, end)

        # Recalculate alpha
        alpha = starting_alpha * (1 - epoch_count/num_epoch)
        if alpha == 0 :
            alpha = starting_alpha * (1 - (epoch_count-0.5)/num_epoch)

        word_count = 0
        last_word_count = 0
        loss = 0
        global_word_count.value = 0
        print(start, end, fi.seek(start), fi.tell())

        while fi.tell() < end:

            line = fi.readline().strip()
            # Skip blank lines
            if not line:
                continue

            # Init sent, a list of indices of words in line
            if subsamp == 0 :
                sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])  # sent는 line 한줄을 의미
            elif subsamp == 1 :
                print('Subsampling implentation')
                sent = subsampling(vocab.indices(['<bol>'] + line.split() + ['<eol>']))  # sent는 line 한줄을 의미

            for sent_pos, token in enumerate(sent):  # sent의 단어와 인덱스
                if word_count % 10000 == 0:  # word count가 10000개가 되면
                    global_word_count.value += (word_count - last_word_count)  # global(전체) word count 업데이트
                    last_word_count = word_count  # word count를 last word count로 씌워줌

                    # Recalculate alpha
                    # alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count) # word count 비율만큼 alpha값을 하향조정
                    # if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001

                    # Print progress info
                    if subsamp == 0 :
                        sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                                         (alpha, global_word_count.value, vocab.word_count,
                                          float(global_word_count.value) / vocab.word_count * 100))
                        sys.stdout.flush()
                    elif subsamp == 1 :
                        sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                                         (alpha, global_word_count.value, len(sent),
                                          float(global_word_count.value) / len(sent) * 100))
                        sys.stdout.flush()


                # Randomize window size, where win is the max window size
                current_win = np.random.randint(low=1, high=win + 1)
                context_start = max(sent_pos - current_win, 0)  # sent_pos는 중심단어 , current_window는 현재 window size
                context_end = min(sent_pos + current_win + 1, len(sent))
                context = sent[context_start:sent_pos] + sent[sent_pos + 1:context_end]
                # Turn into an iterator? context는 중심단어를 제외한 앞뒤 context word list

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
                    for context_word in context:  # context_word가 label_data, label data를 하나 꺼냄
                        # Init neu1e with zeros
                        neu1e = np.zeros(dim)

                        # Compute neu1e and update syn1
                        if neg > 0:
                            classifiers = [(token, 1)] + [(target, 0) for target in table.sample(
                                neg)]  # token은 중심단어 , 즉 batch_data, target은 neg sample , 중심단어와 neg_sample를 같이 묶어서 classifier로 만듬
                        else:
                            classifiers = zip(vocab[token].path, vocab[token].code)
                        for target, label in classifiers:
                            z = np.dot(syn0[context_word],
                                       syn1[target])  # 주변단어(label data)의 syn0 벡터값과 중심단어 및 neg sample의 syn1 벡터값을 내적
                            p = sigmoid(z)  # 그 값을 시그모이드값으로
                            g = alpha * (label - p)  # 교차엔트로피 미분값 , 여기서 alpha는 학습률
                            loss += abs(g)
                            neu1e += g * syn1[target]  # Error to backpropagate to syn0 # 손실함수 미분값에 neg sample의 syn1 매개변수를 곱한 값이
                            syn1[target] += g * syn0[context_word]  # Update syn1 #

                        # Update syn0
                        syn0[context_word] += neu1e

                word_count += 1
                if word_count % 10000 == 0:
                    print('    loss 값 :', loss, '    epoch :',epoch_count+1)
                    loss = 0
                if word_count % 1000000 == 0 :
                    norm_y = make_norm_embedding(syn0)
                    print(test_analogy(norm_y, vocab_list, analogy_data, analogy_voca),'%')
                    # embedding, label = make_tsne('file')

                    two_d_embeddings, words = print_tsne(norm_y, vocab_list)
                    plot(two_d_embeddings, words)

        epoch_count += 1



    # Print progress info
    global_word_count.value += (word_count - last_word_count)
    sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                     (alpha, global_word_count.value, len(sent),
                      float(global_word_count.value) / len(sent) * 100))
    sys.stdout.flush()



def save(vocab, syn0, fo, binary):
    print('Saving model to', fo)
    dim = len(syn0[0])
    if binary:
        fo = open(fo, 'wb')
        fo.write('%d %d\n'.encode() % (len(syn0), dim))
        fo.write('\n'.encode())
        for token, vector in zip(vocab, syn0):
            fo.write('%s'.encode() % token.word)
            for s in vector:
                fo.write(struct.pack('f', s))
            fo.write('\n'.encode())
    else:
        fo = open(fo, 'w')
        fo.write('%d %d\n' % (len(syn0), dim))
        for token, vector in zip(vocab, syn0):
            word = token.word
            vector_str = ' '.join([str(s) for s in vector])
            fo.write('%s %s\n' % (word, vector_str))

    fo.close()


def __init_process(*args):
    global vocab, syn0, syn1, table, cbow, neg, dim, starting_alpha
    global win, num_processes, global_word_count, subsamp, fi

    vocab, syn0_tmp, syn1_tmp, table, cbow, neg, dim, starting_alpha, win, num_processes, global_word_count, subsamp = args[:-1]
    fi = open(args[-1], 'r')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)


def train(fi, fo, cbow, neg, dim, alpha, win, min_count, num_processes, binary, subsamp):
    # Read train file to init vocab
    vocab = Vocab(fi, min_count)
    # Init net
    syn0, syn1 = init_net(dim, len(vocab))


    global_word_count = Value('i', 0)
    table = None
    if neg > 0:
        print('Initializing unigram table')
        table = UnigramTable(vocab)
    else:
        print('Initializing Huffman tree')
        vocab.encode_huffman()

    # Begin training using num_processes workers
    t0 = time.time()

    pool = Pool(processes=num_processes, initializer=__init_process,
                initargs=(vocab, syn0, syn1, table, cbow, neg, dim, alpha,
                          win, num_processes, global_word_count, subsamp, fi))
    pool.map(train_process, range(num_processes)) # train_process함수에 range(num_processes) 인자를 넣어준다

    t1 = time.time()
    print('Completed training. Training took', (t1 - t0) / 60, 'minutes')

    # Save model to file
    save(vocab, syn0, fo, binary)







if __name__ == '__main__':

    train(fi='text8', fo='file', cbow = False, neg=5, dim=300, alpha=0.001, win=5, min_count=5,num_processes=1, binary=False, subsamp =1)

    embedding, label = make_tsne('file')
    two_d_embeddings, words = print_tsne(embedding,label)
    plot(two_d_embeddings, words)

