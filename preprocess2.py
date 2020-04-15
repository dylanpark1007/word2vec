### making vocabulary for each datafile in 1-billion dataset

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
    heldout_folder = "/hdd/data/text/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/"
    training_folder = "/hdd/data/text/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/"

    with open("1billion-preprocessed_data.bin", "rb") as f:
        total_data = pickle.load(f)

    dictionary = total_data[0]
    count = total_data[1]
    reverse_dictionary = total_data[2]

    filecnt = 0

    print("heldout_folder reading...")
    allfiles = glob.glob(heldout_folder + "*")
    for cnt, filename in enumerate(allfiles):
        vocabulary = []
        print(int((cnt / len(allfiles) * 100)), "% complete...")
        with open(filename, 'r', encoding='UTF8') as f:
            while True:
                line = f.readline()
                if not line: break
                cleanline = clean_str(line)
                cleanline = cleanline.split()
                for i in cleanline:
                    if i in dictionary.keys():
                        vocabulary.append(i)

        with open("./1billion-voca/voca" + str(filecnt), "wb+") as f:
            pickle.dump(vocabulary, f)
        filecnt += 1

    print("heldout folder complete.")
    print("training folder reading...")

    allfiles = glob.glob(training_folder + "*")
    for cnt, filename in enumerate(allfiles):
        vocabulary = []

        print(int((cnt / len(allfiles) * 100)), "% complete...")
        with open(filename, 'r', encoding='UTF8') as f:
            while True:
                line = f.readline()
                if not line: break
                cleanline = clean_str(line)
                cleanline = cleanline.split()
                for i in cleanline:
                    if i in dictionary.keys():
                        vocabulary.append(i)

        with open("./1billion-voca/voca" + str(filecnt), "wb+") as f:
            pickle.dump(vocabulary, f)
        filecnt += 1

    print("all completed.")