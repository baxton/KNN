#!/usr/bin/python


import math
from array import array


def tofile(fout, a):
    len_a = array('d', [len(a)])
    len_a.tofile(fout)
    a.tofile(fout)


def normalize(a):
    l = 0.0
    for i in range(len(a)):
        l += float(a[i]) ** 2
    l = math.sqrt(l)
    for i in range(len(a)):
        a[i] = float(a[i]) / l
    return a

def main():

    words = [
                [1, 2, 56, 77, 155],
                [12, 20, 21],
                [1, 3, 21]
            ]

    title_words = [
                    [1, 2],
                    [12, 56],
                    [1, 21]
                ]

    weights = [
                [1.5, 3., 0.5, 1.02, 2.1],
                [3.4, 2.2, 1.0],
                [2.1, 1.05, 3.01]
            ]

    title_weights = [
                    normalize([2.3, 2.5]),
                    normalize([2.1, 1.02]),
                    normalize([3., 3.5])
                ]

    for i in range(len(weights)):
        weights[i] = normalize(weights[i])


    a = []

    for i in range(len(words)):
        ar = array('d', [i+1, len(title_words[i])])

        for j in range(len(title_words[i])):
            ar.append(title_words[i][j])
            ar.append(title_weights[i][j])

        ar.extend([len(words[i])])

        for j in range(len(words[i])):
            ar.append(words[i][j])
            ar.append(weights[i][j])
        ar.extend([1., 22.])
        a.append(ar)


    fout = open('python_arr.txt', 'w+')

    for ar in a:
        print len(ar)
        print ar
        tofile(fout, ar)

    fout.flush()
    fout.close()


if __name__ == "__main__":
    main()
