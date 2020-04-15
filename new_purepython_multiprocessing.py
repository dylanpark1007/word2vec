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
import os
import glob

os.environ["CUDA_VISIBLE_DEVICES"]= '1'

heldout_folder = "./data/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/"
training_folder = "./data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/"
allfiles1 = glob.glob(heldout_folder + "*")
allfiles2 = glob.glob(training_folder + "*")
allfiles = allfiles1 + allfiles2

def __allfiles(allfiles, num_processes):
    files = []
    start = 0
    index = num_processes
    for i in range(len(allfiles)//num_processes):
        files_token = allfiles[start:index]
        files.append(files_token)
        start += num_processes
        index += num_processes
    return files
num_processes = 4
files = __allfiles(allfiles1,num_processes)


with open("1billion-preprocessed_data.bin", "rb") as f:
    total_data = pickle.load(f)

voca_dict = total_data[0]
for token in ['<bol>', '<eol>']:
    voca_dict[token] = len(voca_dict)
voca_dict['<unk>'] = len(voca_dict)

voca_freq = total_data[1]
for token in ['<bol>', '<eol>']:
    voca_freq[token] = 100000
voca_freq['<unk>'] = 100000

reversed_voca_dict = dict((v, k) for k, v in voca_dict.items())
# reversed_voca_dict = total_data[2]
# for token in ['<bol>', '<eol>']:
#     voca_dict[len(voca_dict)] = token

voca_mostcommon = [(voca_dict[k] , v) for k,v in voca_freq.items()]

pwi = total_data[3]
for token in ['<bol>', '<eol>']:
    pwi[token] = -2
pwi['<unk>'] = 1
index_pwi = dict((voca_dict[k],v) for k,v in pwi.items())

# 각 확률값의 합 1
unigram_values = total_data[4]
unigram_words = total_data[5]
index_unigram_words = [voca_dict[k] for k in unigram_words]

total_length = sum(list(voca_freq.values()))
print('total_length : ',total_length)






class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None  # Path (list of indices) from the root to the word (leaf)
        self.code = None  # Huffman encoding


class Vocab:
    def __init__(self, fi, min_count):


        # VocabItem으로 넣어주기 (eol bol 포함)

        fi_vocab_items = []
        fi_vocab_hash = {}
        word_count = 0
        fi = open(fi, 'r') ## fi는 파일이름, 파일이름을 담은 변수

        # Add special tokens <bol> (beginning of line) and <eol> (end of line)
        for token in ['<bol>', '<eol>']:
            fi_vocab_hash[token] = len(fi_vocab_items)
            fi_vocab_items.append(VocabItem(token)) ## class '<bol>', '<eol>'을 vocab_items에 넣어놨음

        for line in fi:
            tokens = line.split()
            for token in tokens:
                if token not in fi_vocab_hash:
                    fi_vocab_hash[token] = len(fi_vocab_items) # vocab_hash는 {단어 : index} 형태의 단어사전
                    fi_vocab_items.append(VocabItem(token))  # vocab_items는 VocabItem class에 가입된 단어들을 입력순서대로 쭉 나열한 리스트(중복x) -> 단어사전을 만들기 위한것

                # assert vocab_items[vocab_hash[token]].word == token, 'Wrong vocab_hash index'
                # vocab_items[vocab_hash[token]].count += 1 # vocab_hash에 있는 단어들 각각의 count를 세어줌(해당 단어가 corpus에 몇개나오는지)
                word_count += 1 # 총 단어의 개수를 세어줌

                if word_count % 10000 == 0:
                    sys.stdout.write("\rReading word %d" % word_count) # sys.stdout.write는 프린트를 하고, str개수를 반환함 / flush는 버퍼를 비워서 불필요한 메모리를 확보
                    sys.stdout.flush()

            # Add special tokens <bol> (beginning of line) and <eol> (end of line)
            # vocab_items[vocab_hash['<bol>']].count += 1 # 각 tokens(=line)이 끝나면 bol과 eol을 각각 카운트해줌, 즉 line 개수를 의미
            # vocab_items[vocab_hash['<eol>']].count += 1
            word_count += 2


        self.bytes = fi.tell() # fi.tell은 파일의 현재 위치를 정수로 가리키는 함수
        # self.vocab_items = vocab_items  # List of VocabItem objects ## 앞에서 정리한 데이터들을 self로 저장
        self.vocab_hash = voca_dict  # Mapping from each token to its index in vocab
        self.re_vocab_hash = reversed_voca_dict
        self.voca_count = voca_freq
        self.vocab_items = [VocabItem(item) for item in list(voca_freq.keys())]

        self.word_count = word_count  # Total number of words in train file

        # Add special token <unk> (unknown),
        # merge words occurring less than min_count into <unk>, and
        # sort vocab in descending order by frequency in train file
        # self.__sort(min_count)

        # assert self.word_count == sum([t.count for t in self.vocab_items]), 'word_count and sum of t.count do not agree'
        print('')
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


    # def __sort(self, min_count):
    #     tmp = []
    #     tmp.append(VocabItem('<unk>')) # tmp에 unk 하나 넣어주고
    #     unk_hash = 0
    #
    #     count_unk = 0
    #     for token in self.vocab_items: #voab_items 리스트에서 단어 하나씩 꺼내서 그 단어의 count가 내가 정한 min_count보다 작으면 unk count(unk가 몇 종류인지)를 하나씩 올려줌, unk_hash는 unk가 총 몇개 들어있는지
    #         if token.count < min_count:
    #             count_unk += 1
    #             tmp[unk_hash].count += token.count
    #         else:
    #             tmp.append(token) # unk가 아닌 애들은 tmp에 넣어줌
    #
    #     tmp.sort(key=lambda token: token.count, reverse=True) # token count 순서로 내림차순 정렬 ,  tmp는 unk이 없는 리스트
    #
    #     # Update vocab_hash
    #     vocab_hash = {}
    #     for i, token in enumerate(tmp):
    #         vocab_hash[token.word] = i
    #
    #     self.vocab_items = tmp
    #     self.vocab_hash = vocab_hash
    #     self.re_vocab_hash = dict((v, k) for k, v in vocab_hash.items())
    #
    #     print('Unknown vocab size:', count_unk)

    def indices(self, tokens): ## vocab_hash를 이용해서 해당 voca의 index반환
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]

    def return_vocab(self, tokens):
        return [self.re_vocab_hash[token] for token in tokens]


    def encode_huffman(self):
        # Build a Huffman tree
        vocab_size = len(self.vocab_hash)
        count = [self.voca_count[t] for t in self.voca_count] + [1e15] * (vocab_size - 1)
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




def test_analogy(norm_y1, y2, analogy_data, analogy_voca):
    global index
    analogy_voca_vector = []
    analogy_voca_index = []
    batch = analogy_data[::]
    for voca in analogy_voca:
        voca_index = y2.index(voca)
        voca_vector = norm_y1[voca_index]
        analogy_voca_vector.append(voca_vector)
        analogy_voca_index.append(voca_index)

    target_list = []
    prediction = 0
    for idx,factor in enumerate(batch):
        target = norm_y1[y2.index(batch[idx][1])] - norm_y1[y2.index(batch[idx][0])] + norm_y1[y2.index(batch[idx][2])]
        target_list.append(target)
    print('')
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

        # norm = sum([math.pow(t.count, power) for t in vocab])  # Normalizing constant # 해당 token(단어)의 count 개수를 3/4 승해서 다 더해줌.

        table_size = np.uint32(1e3)  # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        print('Filling unigram table')
        p = 0  # Cumulative probability
        i = 0
        for j, unigram in enumerate(index_unigram_words):
            p +=  unigram_values[j] # 각 단어들의 count의 3/4승해서 다 더해준 값(norm)을 각 값으로 나눠주어 p값을 구함.
            while i < table_size and float(i) / table_size < p: # 사이즈가 10의 8승인 테이블 사이즈에 p 분포만큼 해당 단어의 인덱스 개수만큼 맵핑시킴
                table[i] = unigram
                i += 1
        self.table = table # 즉 테이블은 단어 각각의 p 확률분포만큼 사이즈가 10의 8승인 테이블에 인덱스를 맵핑시키는 것임.


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


    syn0 = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size, dim))  # 사이즈가 V X N이고 각 값이 -0.5/N 에서 +0.5/N사이의 값들을 random uniform하게 테이블로 생성
    # tmp = np.random.uniform(low=-0.5, high=0.5, size=(vocab_size, dim))
    # syn0 = np.ctypeslib.as_ctypes(tmp)
    # syn0 = Array(syn0._type_, syn0, lock=False)

    # Init syn1 with zeros
    syn1 = np.zeros(shape=(vocab_size, dim)) # syn0과 같으나 첨엔 zero를 입힘
    # tmp = np.zeros(shape=(vocab_size, dim))
    # syn1 = np.ctypeslib.as_ctypes(tmp)
    # syn1 = Array(syn1._type_, syn1, lock=False)

    return (syn0, syn1)

dim = 300
syn0, syn1 = init_net(dim, len(voca_dict))

def subsampling(object_data):
    voca_sampled = [word for word in object_data if random.random() < (1-index_pwi[word])]
    return voca_sampled



# print('analogy done')
# epoch_count = 0
# num_epoch = 25
#
# while epoch_count < num_epoch:

#15,21,16,17  0.69 1.35 0.81 1.06

def train_process(pid): #pid는 range(num_processes)
    # Set fi to point to the right chunk of training file
    global syn0
    global syn1
    print(syn0[0][0:4])
    start = 0  # byte는 현재 위치를 가르키는 정수 , 그거에 연산 횟수를 나눠주고 다시 pid를 곱해줌, 그 값을 start로
    end = vocab.bytes   # end는 pid가 연산횟수 -1과 같아질때 end값 아니라면 else 뒤의 값
    fi.seek(start)
    # print 'Worker %d beginning training at %d, ending at %d' % (pid, start, end)

    # Recalculate alpha
    alpha = starting_alpha
    # alpha = starting_alpha * (1 - epoch_count/num_epoch)
    # if alpha == 0 :
    #     alpha = starting_alpha * (1 - (epoch_count-0.5)/num_epoch)

    word_count = 0
    last_word_count = 0

    vocab_list = vocab.vocab_items
    vocab_list = [i.word for i in vocab_list]
    # analogy_data_semantic, analogy_voca_semantic = test_dataset('./analogy task_semantic.txt', vocab_list)
    # analogy_data_syntactic, analogy_voca_syntactic = test_dataset('./analogy task_syntactic.txt', vocab_list)
    while fi.tell() < end:
        line = fi.readline().strip()
        # Skip blank lines
        if not line:
            continue

        # Init sent, a list of indices of words in line
        if subsamp == 0 :
            sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])  # sent는 line 한줄을 의미
        elif subsamp == 1 :
            sent = subsampling(vocab.indices(['<bol>'] + line.split() + ['<eol>']))  # sent는 line 한줄을 의미
        # sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])
        for sent_pos, token in enumerate(sent):  # sent의 단어와 인덱스
            if word_count % 10000 == 0:  # word count가 10000개가 되면
                global_word_count.value += (word_count - last_word_count)  # global(전체) word count 업데이트
                last_word_count = word_count  # word count를 last word count로 씌워줌

                # Recalculate alpha
                alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count) # word count 비율만큼 alpha값을 하향조정
                if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001

                # Print progress info
                # sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                #                  (alpha, global_word_count.value, vocab.word_count,
                #                   float(global_word_count.value) / vocab.word_count * 100))
                # sys.stdout.flush()
                print("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                                 (alpha, global_word_count.value, vocab.word_count,
                                  float(global_word_count.value) / vocab.word_count * 100))



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
                        neu1e += g * syn1[target]  # Error to backpropagate to syn0 # 손실함수 미분값에 neg sample의 syn1 매개변수를 곱한 값이
                        syn1[target] += g * syn0[context_word]  # Update syn1 #

                    # Update syn0
                    syn0[context_word] += neu1e

            word_count += 1

            # if word_count % 10000 == 0:
            #     print('    loss 값 :', loss, '    epoch :',epoch_count+1)
            #     loss = 0
            if word_count % 200000 == 0 :
                norm_y = make_norm_embedding(syn0)
                print('semantic : ',test_analogy(norm_y, vocab_list, analogy_data_semantic, analogy_voca_semantic),'%')
                print('syntactic : ',test_analogy(norm_y, vocab_list, analogy_data_syntactic, analogy_voca_syntactic), '%')
            #     # embedding, label = make_tsne('file')
            #
            #     two_d_embeddings, words = print_tsne(norm_y, vocab_list)
            #     plot(two_d_embeddings, words)
    # Print progress info
    global_word_count.value += (word_count - last_word_count)
    sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                     (alpha, global_word_count.value, len(sent),
                      float(global_word_count.value) / len(sent) * 100))
    sys.stdout.flush()
    fi.close()



def save(syn0, syn1):
    with open('./model/first_syn0.bin','wb') as f1:
        pickle.dump(syn0,f1)
    with open('./model/first_syn1.bin','wb') as f2:
        pickle.dump(syn1,f2)
    print('complete file save')



def load():
    with open('./model/first_syn0.bin','rb') as f1:
        syn0 = pickle.load(f1)
    with open('./model/first_syn1.bin','rb') as f2:
        syn1 = pickle.load(f2)
    print('complete file load')
    return syn0, syn1


def __init_process(*args):
    global vocab,  table, cbow, neg, dim, starting_alpha
    global win, num_processes, global_word_count, subsamp, fi

    vocab,  table, cbow, neg, dim, starting_alpha, win, num_processes, global_word_count, subsamp = args[:-1]
    fi = open(args[-1], 'r')
    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore', RuntimeWarning)
    #     syn0 = np.ctypeslib.as_array(syn0_tmp)
    #     syn1 = np.ctypeslib.as_array(syn1_tmp)



def train_first(fi, fo, cbow, neg, dim, alpha, win, min_count, num_processes, binary, subsamp):
    # Read train file to init vocab
    vocab = Vocab(fi, min_count)
    # Init net
    syn0, syn1 = init_net(dim, len(voca_dict))

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
                initargs=(vocab, syn0, syn1,  table, cbow, neg, dim, alpha,
                          win, num_processes, global_word_count, subsamp, fi))
    pool.map(train_processor, range(num_processes)) # train_process함수에 range(num_processes) 인자를 넣어준다

    t1 = time.time()
    print('')
    print('Completed training. Training took', (t1 - t0) / 60, 'minutes')
    pool.close()
    pool.join()
    # Save model to file
    # with open('./model/first_syn0.bin','wb') as f1:
    #     pickle.dump(syn0,f1)
    # with open('./model/first_syn1.bin','wb') as f2:
    #     pickle.dump(syn1,f2)




def train(pid):
    print(pid)
    global syn0
    global syn1
    fi = part_files
    fo = 'file'
    cbow = False
    neg = 15
    dim = 300
    alpha = 0.025
    win = 5
    min_count = 5
    num_processes = 4
    binary = False
    subsamp = 0
    # Read train file to init vocab
    vocab = Vocab(fi[pid], min_count)
    # Init net
    # with open('./model/first_syn0.bin','rb') as f1:
    #     syn0 = pickle.load(f1)
    # with open('./model/first_syn1.bin','rb') as f2:
    #     syn1 = pickle.load(f2)


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
    __init_process(vocab, table, cbow, neg, dim, alpha,
                          win, num_processes, global_word_count,subsamp, fi[pid])
    train_process(pid) # train_process함수에 range(num_processes) 인자를 넣어준다

    t1 = time.time()
    print('Completed training. Training took', (t1 - t0) / 60, 'minutes')


    # Save model to file
    # with open('./model/first_syn0.bin','wb') as f3:
    #     pickle.dump(syn0,f3)
    # with open('./model/first_syn1.bin','wb') as f4:
    #     pickle.dump(syn1,f4)
    # save(vocab, syn0, fo, binary)



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # # parser.add_argument('-train', help='Training file', dest='fi')
    # # parser.add_argument('-model', help='Output model file', dest='fo')
    # parser.add_argument('-train', help='Training file', dest='fi', required=True)
    # parser.add_argument('-model', help='Output model file', dest='fo', required=True)
    # parser.add_argument('-cbow', help='1 for CBOW, 0 for skip-gram', dest='cbow', default=1, type=int)
    # parser.add_argument('-negative',
    #                     help='Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax',
    #                     dest='neg', default=5, type=int)
    # parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=100, type=int)
    # parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.025, type=float)
    # parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int)
    # parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=5,
    #                     type=int)
    # parser.add_argument('-processes', help='Number of processes', dest='num_processes', default=1, type=int)
    # parser.add_argument('-binary', help='1 for output model in binary format, 0 otherwise', dest='binary', default=0,
    #                     type=int)
    # # TO DO: parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)
    # args = parser.parse_args()
    #
    # train(args.fi, args.fo, bool(args.cbow), args.neg, args.dim, args.alpha, args.win, args.min_count, args.num_processes, bool(args.binary))

    # train(fi='text8', fo='file', cbow = False, neg=0, dim=300, alpha=0.025, win=5, min_count=5,num_processes=1, binary=False)


    for part_files in files:
        pool = Pool(processes=num_processes)
        pool.map(train, range(num_processes))
        pool.close()
        pool.join()







## tsne가 잘못됨
## epoch 주기 - 에폭시넘어갈때 파일이 닫히면서 fi.seek 에러남
## epoch 되면 learning rate 감소되는거 적용

## one billion 돌려보기
## subsampling 적용
## num_process 4랑 1이랑 결과값 비교해보기
## 차이 안나면 4 or 6 or 8 로
## num_process가 2 이상일때 epoch 주면 엉키므로 돌리고 나면 pickle로 저장, 그 다음에 불러들여서 다시 돌리기

## negative sampling은 각 파일별로? huffman tree는 code, path만 전체 파일로 돌리는거는 개별 파일로
## 그대로 가되, for 문으로 fi 이름만 바꿔서 돌아가게


## multiprocessing을 하면 데이터가 엉켜서 학습이 안되진 않는데, 학습은 되는데 똑같은 파일을 4번 읽음
## num_process들을 파일별로 할당할 수 있나?
## subsampling 미리 만들어놓은 pwi로 쓰기
## unigram table도 미리 만들어놓은 pwi로 쓰기

