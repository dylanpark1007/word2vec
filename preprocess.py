### making dictionary & other required data

import pickle
import re
import glob
import math


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", "  ", string)
    string = re.sub(r"'t", " ", string)
    string = re.sub(r"!", "  ", string)
    string = re.sub(r"\(", "  ", string)
    string = re.sub(r"\)", "  ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"[\.,-]", "", string)
    return string.strip().lower()


if __name__ == "__main__":
    heldout_folder = "./data/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/"
    training_folder = "./data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/"

    dictionary = {}
    count = {}
    reverse_dictionary = {}
    # dictionary 생성
    dic_cnt = 0
    print("heldout_folder reading...")
    allfiles = glob.glob(heldout_folder + "*")
    for cnt, filename in enumerate(allfiles):
        print(int((cnt / len(allfiles) * 100)), "% complete...")
        with open(filename, 'r', encoding='UTF8') as f:
            while True:
                line = f.readline()
                if not line: break
                cleanline = clean_str(line)
                cleanline = cleanline.split()
                for i in cleanline:
                    if i not in dictionary.keys():
                        dictionary[i] = dic_cnt
                        dic_cnt += 1
                    if i in count.keys():
                        count[i] += 1
                    else:
                        count[i] = 1

    print("heldout folder complete.")
    print("training folder reading...")

    allfiles = glob.glob(training_folder + "*")
    for cnt, filename in enumerate(allfiles):
        print(int((cnt / len(allfiles) * 100)), "% complete...")
        with open(filename, 'r', encoding='UTF8') as f:
            while True:
                line = f.readline()
                if not line: break
                cleanline = clean_str(line)
                cleanline = cleanline.split()
                for i in cleanline:
                    if i not in dictionary.keys():
                        dictionary[i] = dic_cnt
                        dic_cnt += 1
                    if i in count.keys():
                        count[i] += 1
                    else:
                        count[i] = 1

    print("dictionary 길이 :", len(dictionary))

    for i in dictionary.keys():
        reverse_dictionary[dictionary[i]] = i

    max_common = max(count.values())
    for i in count.keys():
        if count[i] == max_common:
            max_common_word = i

    print("Most common word : ", max_common_word)

    print("filtering words under min_freq...")
    # 단어 빈도 5번 미만인 것들은 unk로 바꿈
    min_freq = 5
    count["UNK"] = 0
    dictionary["UNK"] = -1
    unkcnt = 0
    cnt = 0
    for i in list(dictionary.keys()):
        cnt += 1
        if count[i] < min_freq:
            unkcnt += 1
            count["UNK"] += 1
            del dictionary[i]
            del count[i]
    dic_cnt = 0
    for i in list(dictionary.keys()):
        dictionary[i] = dic_cnt
        dic_cnt += 1
    print("unkcnt=", unkcnt)
    print("dictionary length:", len(dictionary),"=",dic_cnt)
    print("count length:", len(count))
    for i in dictionary.keys():
        reverse_dictionary[dictionary[i]] = i

    print("reverse_dictionary created.")

    all_num = sum(count.values())
    t = 1e-5
    fwi = {}
    pwi = {}
    for i in count.keys():
        fwi[i] = count[i] / all_num
    for i in count.keys():
        pwi[i] = 1 - math.sqrt(t / fwi[i])

    # pwi test
    dic_words = ["mother", "father", "gentleman", "watch"]
    for i in dic_words:
        print(i, "의  pwi :", pwi[i])

    unigram_words = []
    unigram_values = []
    sum_fwi = 0
    for i in fwi.keys():
        sum_fwi += math.pow(fwi[i], 0.75)
    for i in fwi.keys():
        unigram_words.append(i)
        unigram_values.append(math.pow(fwi[i], 0.75) / sum_fwi)
    print("unigram_words/values created.")
    with open("./1billion-preprocessed_data.bin", "wb+") as f:
        total_data = [dictionary, count, reverse_dictionary, pwi, unigram_values, unigram_words]
        pickle.dump(total_data, f)
    print("preproessed_data.bin saved.")
