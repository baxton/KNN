

import numpy as np
import scipy as sp
from sklearn.ensemble import RandomForestRegressor
import datetime






class Counter(object):
    def __init__(self):
        self.node_id = 0

    def next(self):
        n = self.node_id
        self.node_id += 1
        return n


def sigmoid(X):
##    e = sp.exp(-X)
##    e = 0.0000001 if e ==
    v = 1.  / (1. + sp.exp(-X))
    if sp.isnan(v).sum() or sp.isinf(v).sum():
        i=0
    return v

def cost(theta, X, Y):
    M = X.shape[0]
    h = sigmoid(np.dot(X, theta))
    h[h == 1.] = .99999999
    h[h == 0.] = .00000001
    E = np.sum(-Y * sp.log(h) - (1 - Y) * sp.log(1 - h)) / M
    grad = np.dot(h - Y, X) ##/ M

    return E, grad


def cost_lin(theta, X, Y):
    M = X.shape[0]
    tmp = np.dot(X, theta) - Y

    E = (np.sum(tmp**2) / 2.) / M
    grad = np.dot(tmp, X) / M

    return E, grad



def minimize_gc(theta, X, Y, max_iterations=100, func=cost):
    E, grad = func(theta, X, Y)

    e = 0.0001
    cur_iter = 0
    a = .4

    while E > e and cur_iter < max_iterations:
        cur_iter += 1

        theta = theta - grad * a
        new_e, grad = func(theta, X, Y)

        if E < new_e:
            a /= 2.

        E = new_e

    return theta


def h(theta, v):
    return sigmoid(np.dot(v, theta))


def k_out_of_n(a, k):
    n = a.shape[0]
    result = sp.zeros((k,))
    result = a[:k]
    for i in range(k,n):
        r = sp.random.randint(0, i)
        if r < k:
            result[r] = a[i]
    return result





def split(X, Y, xidx, lnum):

    N = X.shape[0]
    x_min = X[:,xidx].min()
    x_max = X[:,xidx].max()

    min_x = x_min
    min_rss = -1
    left_ii = None
    left_num = 0

    #print "N: ", N

    for x in sp.linspace(x_min, x_max, 1000):
    #for x in X[:,xidx]:
        # estimate MSE for this x
        ii = X[:,xidx] <= x
        num = ii.sum()

        if lnum > num or lnum > N-num:
            continue

        mean = Y.mean()
        left_mean = Y[ii].mean()
        right_mean = Y[~ii].mean()
        #print x, ": ", left_mean, right_mean

        M = Y.shape[0]
        left_M = num
        right_M = N - num

        rss = np.sum((Y - mean)**2) - (np.sum((Y[ii] - left_mean)**2) + np.sum((Y[~ii] - right_mean)**2))
        ##rss = np.sum((Y - mean)**2)/M - (np.sum((Y[ii] - left_mean)**2)/left_M + np.sum((Y[~ii] - right_mean)**2)/right_M)
        ##rss = np.abs(np.sum((Y[ii] - left_mean)**2) - np.sum((Y[~ii] - right_mean)**2))

        #print x, ": ", rss

        if (not np.isnan(rss) and not np.isinf(rss)) and (min_rss == -1 or min_rss < rss):
            min_x = x
            min_rss = rss
            left_ii = ii
            left_num = num
    #print "Min x:", min_x, ": ", min_rss
    return min_x, min_rss, left_ii, left_num



class node(object):
    def __init__(self, lnum, k, counter):
        self.xidx = None
        self.d = None
        self.theta = None
        self.left = None
        self.right = None
        self.val = None
        self.lnum = lnum
        self.k = k

        self.counter = counter
        self.id = self.counter.next()

    def print_tree(self):
        if self.d != None:
            print "node " + str(self.id) + ": [", self.d, ", ", self.xidx, "] left " + str(self.left.id) + " right " + str(self.right.id)
        else:
            print "node " + str(self.id) + ": (leaf) %f" % self.val

        if self.left != None:
            self.left.print_tree()
        if self.right != None:
            self.right.print_tree()

    def print_tree_ex(self, tree_id):
        arr = np.array(self.asarray())
        arr = arr[ np.argsort(arr[:,0]) ]

        print "double tree_%d [][VEC_LEN] = {" % tree_id
        first_row = True
        for a in arr[:,1:]:
            print ("" if first_row else ",\n") + "  {",
            first = True
            for n in a:
                if not first:
                    print ",",
                print "%10.16f" % n,
                first = False
            print "}",
            first_row = False
        print "\n};"

    def asarray(self):
        theta_len = 26
        row_len = theta_len + 1 + 1 # id, leaf/non-leaf, theta/data
        result = []

        if self.d != None:
            # [leaf/non-leaf, idx, val, left, right, ...]  5 + 22
            # [leaf/non-leaf, theta0, ..., theta26]        27
            row = [0.] * row_len
            row[0] = self.id
            row[1] = 0
            row[2] = self.xidx
            row[3] = self.d
            row[4] = self.left.id
            row[5] = self.right.id
            result.append(row)
            result.extend( self.left.asarray() )
            result.extend( self.right.asarray() )
        else:
            row = [0.] * row_len
            row[0] = self.id
            row[1] = 1
            row[2:] = self.theta
            result.append(row)

        return result


    def split(self, X, Y):
        # number of items and features
        N = X.shape[0]
        FN = X.shape[1]

        # get feature randomly
        #idx = sp.random.randint(0, FN, 20)
        idx = k_out_of_n(np.array(range(FN)), self.k)

        # split
        max_idx = -1
        max_rss = -1
        max_left_ii = None
        max_left_num = -1
        max_x = -1

        for i in idx:
            x, rss, left_ii, left_num = split(X, Y, i, self.lnum)
            if max_rss == -1 or max_rss < rss:
                max_x = x
                max_idx = i
                max_rss = rss
                max_left_ii = left_ii
                max_left_num = left_num

        x = max_x
        idx = max_idx
        rss = max_rss
        left_ii = max_left_ii
        left_num = max_left_num

        ##print "selected :" , "0" if rss1 > rss2 else "1", rss1, rss2

        #x, rss, left_ii, left_num = split(X, Y, idx, self.lnum)

        # if we have split with enough items
        if -1 != rss:
            self.d = x
            self.xidx = idx

            self.left = node(self.lnum, self.k, self.counter)
            self.right = node(self.lnum, self.k, self.counter)

            self.left.split(X[left_ii], Y[left_ii])
            self.right.split(X[~left_ii], Y[~left_ii])

            ##print self.id, " idx: ", self.xidx, "; x: ", x, "[", left_ii.sum(), ",", (~left_ii).sum(), "] l->", self.left.id, " r->", self.right.id

        else:
            tmpX = np.zeros((N,FN + 1))
            tmpX[:,0] = 1.
            tmpX[:,1:] = X
            theta = sp.rand(FN+1)
            #self.theta = minimize_gc(theta, tmpX, Y, max_iterations=1000)
            #self.theta = minimize_gc(theta, tmpX, Y, func=cost_lin, max_iterations=2000)
            self.val = Y.mean()

    def predict(self, v):
        if None != self.d:
            if v[self.xidx] <= self.d:
                return self.left.predict(v)
            else:
                return self.right.predict(v)

        #return h(self.theta, sp.concatenate(([1.], v)))
        #return np.dot(sp.concatenate(([1.], v)), self.theta)
        return self.val




class RF:
    def __init__(self, trees, lnum, k):
        self.TREES = trees
        self.lnum = lnum
        self.k = k
        self.forest = []

    def fit(self, X, Y):
        N = Y.shape[0]

        for i in range(self.TREES):
            d=datetime.datetime.now()
            sp.random.seed(d.hour * 60 * 60 * 1000 + d.minute * 60 * 1000 + d.second * 1000 + d.microsecond)
            ii = sp.random.randint(0, N, int(N * .7))

            tmpX = X[ii]
            tmpY = Y[ii]
            counter = Counter()
            t = node(self.lnum, self.k, counter)
            t.split(tmpX, tmpY)
            self.forest.append(t)


    def predict(self, X):
        res = sp.zeros((X.shape[0],))
        i = 0
        for x in X:
            total_p = 0.
            for t in self.forest:
                p = t.predict(x)
                total_p += p
            res[i] = total_p / self.TREES
            i += 1

        return res




class LR:
    def __init__(self):
        self.theta = None

    def fit(self, X, Y):
        FN = 1+X.shape[1]

        tmpX = sp.ones((X.shape[0], FN))
        tmpX[:,1:] = X

        self.theta = sp.rand(FN)
        self.theta = minimize_gc(self.theta, tmpX, Y, func=cost_lin, max_iterations=2000)


    def predict(self, X):
        res = sp.zeros((X.shape[0],))
        i = 0
        for x in X:
            p = np.dot(sp.concatenate(([1.], x)), self.theta)
            res[i] = p
            i += 1

        return res

####################################################################

def get_k_of_n(k, low, high):
    numbers = np.array(range(low, low + k))
    for i in range(low + k, high):
        r = sp.random.randint(low, i) - low
        if r < k:
            numbers[r] = i
    return numbers




def predict(train, test):

    #rf_w = RandomForestRegressor(n_estimators=40)
    #rf_l = RandomForestRegressor(n_estimators=40)
    #rf_h = RandomForestRegressor(n_estimators=40)
    #rf_w = RF(10, 5, 5)
    #rf_l = RF(10, 5, 5)
    #rf_h = RF(10, 5, 5)

    rf_w = LR()
    rf_l = LR()
    rf_h = LR()


    rf_w.fit(train[:,:-3], train[:,-3])
    rf_l.fit(train[:,:-3], train[:,-2])
    rf_h.fit(train[:,:-3], train[:,-1])

    preds_w = rf_w.predict(test[:, :-3])
    preds_l = rf_l.predict(test[:, :-3])
    preds_h = rf_l.predict(test[:, :-3])

    sse_w = sp.sqrt(((test[:,-3] - preds_w) ** 2).sum())
    print sse_w

    sse_l = sp.sqrt(((test[:,-2] - preds_l) ** 2).sum())
    print sse_l

    sse_h = sp.sqrt(((test[:,-1] - preds_l) ** 2).sum())
    print sse_h

    return preds_w, preds_l, preds_h





def main():

    ID_IDX         = 0

    AGEDAYS_IDX    = 1
    GAGEDAYS_IDX   = 2
    SEX_IDX        = 3
    MUACCM_IDX     = 4
    SFTMM_IDX      = 5
    BFED_IDX       = 6
    WEAN_IDX       = 7
    GAGEBRTH_IDX   = 8
    MAGE_IDX       = 9
    MHTCM_IDX      = 10
    MPARITY_IDX    = 11
    FHTCM_IDX      = 12

    WTKG_IDX       = 13
    LENCM_IDX      = 14
    HCIRCM_IDX     = 15



    fname = "C:\\Temp\\ch2\\data\\exampleData.csv"

    data_txt = sp.loadtxt(fname, dtype="S20", delimiter=',')

    means = [-0.443631, -0.443048, 1.51245, -0.162933, -0.0821803, 0.995769, 0.619883, 0.00493325, -0.0546572, -0.0356498, 4.73872, -0.0217742]

    for i in range(data_txt.shape[0]):
        for j in range(1, data_txt.shape[1]-3):
            if data_txt[i, j] == '.':
                data_txt[i, j] = means[j-1]

    data = data_txt.astype(float)
    data_txt = None

    N = data.shape[0]

    FN = data.shape[1] - 1 - 3
    data_ex = sp.zeros((N, FN * 2 + 3))

    #data_ex[:,:FN] = data[:,1:FN+1]
    data_ex[:,-3:] = data[:,-3:]

    for i in range(N):
        data_ex[i,0] = data[i,AGEDAYS_IDX]
        data_ex[i,1] = data[i,GAGEDAYS_IDX]
        data_ex[i,2] = data[i,SEX_IDX]
        data_ex[i,3] = data[i,MUACCM_IDX]
        data_ex[i,4] = data[i,SFTMM_IDX]
        data_ex[i,5] = data[i,BFED_IDX]
        data_ex[i,6] = data[i,WEAN_IDX]
        data_ex[i,7] = data[i,GAGEBRTH_IDX]
        data_ex[i,8] = data[i,MAGE_IDX]
        data_ex[i,9] = data[i,MHTCM_IDX]
        data_ex[i,10] = data[i,MPARITY_IDX]
        data_ex[i,11] = data[i,FHTCM_IDX]

    F1 = 1
    F2 = 10
    F3 = 3
    for i in range(N):
        break
        data_ex[i,FN+0] = data[i,F1] * data[i,F2]


    N = data.shape[0]
    k = int(N * .6)
    train_ii = get_k_of_n(k, 0, N)
    test_ii = [i for i in range(N) if i not in train_ii]

    train = data_ex[train_ii]
    test = data_ex[test_ii]


    preds_w, preds_l, preds_h = predict(train, test)

    for i in range(10):
        print data[i, 0], preds_w[i], test[i,-3], preds_l[i], test[i,-2], preds_h[i], test[i,-1]


if __name__ == '__main__':
    main()
