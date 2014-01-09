
import ctypes
from pathes import PATHES

dll = ctypes.cdll.LoadLibrary(PATHES['BASE'] + r'\cpp\classifier.dll')


class Classifier():
    def __init__(self):
        self.__allocate_messages = dll.allocate_messages



def main():
    pass




if __name__ == '__main__':
    main()
