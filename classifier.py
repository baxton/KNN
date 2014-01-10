
import os
import csv
import trie
import ctypes
from pathes import PATHES
from datetime import datetime as dt
from processor import get_words_array

dll = ctypes.cdll.LoadLibrary(PATHES['BASE'] + r'\cpp\classifier.dll')

##void set_fully_classified_threshold(double val)
##struct MSG* allocate_messages(int size)
##void free_messages(struct MSG* p, int size)
##void add_message(void* messages, long idx, long id, long t_num_words, double* t_words, long num_words, double* words, int cross_testing)
##void get_message_key_words(struct MSG* messages, int idx, int** key_words, int* size)
##void free_message_key_words(int* key_words)
##void classify(struct MSG* messages, int messages_size, const char* model, int* error, int* num_of_packs)




class Classifier():
    def __init__(self, messages_size, model_file_name):
        self.__allocate_messages = dll.allocate_messages
        self.__free_messages = dll.free_messages
        self.__set_fully_classified_threshold = dll.set_fully_classified_threshold
        self.__add_message = dll.add_message
        self.__get_message_key_words = dll.get_message_key_words
        self.__free_message_key_words = dll.free_message_key_words
        self.__get_message_id = dll.get_message_id
        self.__classify = dll.classify

        self.messages_size = messages_size
        self.model_file_name = model_file_name

        self.__prepare()



    def __prepare(self):
        self.messages = self.__allocate_messages(self.messages_size)

    # title & bode: <len><w1><num1>...<wN><numN>
    def add_message(self, index, id, title, body, tags, cross_testing=False):
        t_words, t_num_words = title.buffer_info()
        words, num_words = body.buffer_info()
        self.__add_message(self.messages, index, id,
                           t_num_words - 1, t_words + 8,
                           num_words - 1, words + 8,
                           1 if cross_testing else 0)


    def free_resources(self):
        self.__free_messages(self.messages, self.messages_size)

    def classify(self):
        number_of_packs = ctypes.c_int(0)
        error = ctypes.c_int(0)
        self.__classify(self.messages, self.messages_size, self.model_file_name, ctypes.addressof(error), ctypes.addressof(number_of_packs))

        print "Classification finished, number of packs %d, error code %d" % (number_of_packs.value, error.value)


    def get_messages(self):
        result = []
        for i in range(self.messages_size):
            p_void = ctypes.c_void_p(0)
            size = ctypes.c_int(0)
            self.__get_message_key_words(self.messages, i, ctypes.addressof(p_void), ctypes.addressof(size))
            id = self.__get_message_id(self.messages, i)
            if 0 < size.value:
                tags = ctypes.cast(p_void, ctypes.POINTER(ctypes.c_int*size.value))
                tmp = [id]
                for j in range(size.value):
                    tmp.append(tags.contents[j])
                result.append(tmp)
            self.__free_message_key_words(p_void)
        return result

    def save_to_file(self, fout, key_words):
        for i in range(self.messages_size):
            p_void = ctypes.c_void_p(0)
            size = ctypes.c_int(0)
            self.__get_message_key_words(self.messages, i, ctypes.addressof(p_void), ctypes.addressof(size))
            id = self.__get_message_id(self.messages, i)
            if 0 < size.value:
                tags = ctypes.cast(p_void, ctypes.POINTER(ctypes.c_int*size.value))
                tmp = []
                for j in range(size.value / 2):
                    tmp.append( '%s' % key_words[int(tags.contents[j*2+0])] )
                tmp[0] = '"' + tmp[0]
                tmp[-1] = tmp[-1] + '"'

                fout.write('"%d",' % id)
                fout.write(' '.join(tmp))
                fout.write('\n')

            self.__free_message_key_words(p_void)


#
# Testing
#

def get_keywords():
    pass

def next_message(fin, use_tags=False):
    words = trie.Trie()
    words.load_file(PATHES['DICT'] + '\\Word_stem.txt', word_col=1, index_col=0)

    kwords = trie.Trie()
    kwords.load_file(PATHES['DICT'] + '\\Keywords.txt', word_col=0, index_col=1)

    reader = csv.DictReader(fin)
    for l in reader:
        id = int(l['Id'])
        title = get_words_array(words.tokenize(l['Title']))
        body = get_words_array(words.tokenize(l['Body']))
        tags =[] if not use_tags else kwords.tokenize(l['Tags'])
        yield id, title, body, tags


def main():
    os.chdir(PATHES['BASE'])

    num_skip = 1000
    messages_to_classify = 50

    start = dt.now()

    # initialization
    c = Classifier(messages_to_classify, PATHES['PROCESSED'] + '\\Model.bin')

    # read messages
    messages_store = []
    messages_read = 0
    with open(PATHES['BASE'] + '\\Train.csv', 'r') as fin:
        for id, title, body, tags in next_message(fin, use_tags=True):
            if num_skip:
                num_skip -= 1
                continue
            print id, title, body, tags
            messages_store.append(title)
            messages_store.append(body)

            c.add_message(messages_read, id, title, body, tags, True)

            messages_read += 1
            if messages_read == messages_to_classify:
                break

    print "Messages loaded %d sec" % (dt.now() - start).seconds
    start = dt.now()

    # classification
    c.classify()

    # free memory
    while len(messages_store):
        messages_store.pop()

    print "Classification finished %d sec" % (dt.now() - start).seconds

    key_words = []
    with open(PATHES['DICT'] + '\\Keywords.txt', 'r') as fin:
        for l in fin:
            w, id = l.split(' ')
            key_words.append(w)


    with open(PATHES['PROCESSED'] + '\\result.txt','w+') as fout:
        c.save_to_file(fout, key_words)

    # release resources
    c.free_resources()




if __name__ == '__main__':
    main()
