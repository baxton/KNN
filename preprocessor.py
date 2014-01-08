
import os
import csv
import math
import trie
from array import array
from pathes import PATHES
from collections import defaultdict
from datetime import datetime as dt




def fetch_keywords(fname, foutname):
    kwords = set([])

    with open(fname, 'r') as fin:
        reader = csv.DictReader(fin)
        first = True
        for l in reader:
            if not first:
                tags = l['Tags'].split(' ')
                kwords.update(tags)
            else:
                first = False

    result = list(kwords)
    result.sort()
    i = 1
    with open(foutname, 'w+') as fout:
        for kw in result:
            fout.write('%s %d\n' % (kw, i))
            i += 1



def normalize(arr):
    sum = 0.
    i = 2
    while i < len(arr):
        sum += arr[i] ** 2
        i += 2
    sum = math.sqrt(sum)
    i = 2
    while i < len(arr):
        arr[i] /= sum
        i += 2

    return arr


def get_words_array(orig):
    result = array('d')
    words = defaultdict(float)
    # counting
    for w in orig:
        words[w] += 1.
    # sorting
    tmp = words.keys()
    tmp.sort()

    result.append(len(tmp))

    # filling in
    for w in tmp:
        result.append(w)
        result.append(words[w])

    return normalize(result)



def save_to_file(fout, id, title, body, tags):
    data = array('d', [0., id])
    data.extend(get_words_array(title))
    data.extend(get_words_array(body))

    tags.sort()
    data.append(len(tags))
    data.extend(tags)

    data[0] = len(data) - 1

    data.tofile(fout)





def process(fname):
    os.chdir(PATHES['BASE'])

    words = trie.Trie()
    words.load_file(PATHES['DICT'] + '\\Eng.bin', word_col=0, index_col=1)

    kwords = trie.Trie()
    kwords.load_file(PATHES['DICT'] + '\\Keywords.txt', word_col=0, index_col=1)

    lines = 50000

    with open(fname, 'r') as fin:
        with open(PATHES['PROCESSED'] + '\\Model.bin', 'wb+') as fout:
            reader = csv.DictReader(fin)
            first = True
            for l in reader:
                if not first:
                    id = int(l['Id'])
                    title = words.tokenize(l['Title'])
                    body = words.tokenize(l['Body'])
                    tags = kwords.tokenize(l['Tags'])

                    save_to_file(fout, id, title, body, tags)
                    if 0 == lines % 50000:
                        print lines, 'last id: ', id
                else:
                    first = False

                lines += 1



def main():
    os.chdir(PATHES['BASE'])

    # keywords fetching
##    fetch_keywords(PATHES['BASE'] + '\\Train.csv', PATHES['DICT'] + '\\Keywords.txt')

    start = dt.now()
    process('Train.csv')
    print "Finished: %d sec" % (dt.now() - start).seconds



if __name__ == '__main__':
    main()
