

import numpy as np
import scipy as sp
import  scipy.spatial.distance as spd
import scipy.optimize as spo


PROMOTE=False


FLD_ID=0
FLD_TIME=1
FLD_SEX=2
FLD_STATUS=3
FLD_F1=4
FLD_F2=5
FLD_F3=6
FLD_F4=7
FLD_F5=8
FLD_F6=9
FLD_F7=10
FLD_F8=11
FLD_W=12
FLD_B=13



# id, w, b, "1", male, female, status, features: time, F1-F8
# for 24 I can have 3 variants of each feature
NUM_BUCKETS = 10
CELLS_IN_BUCKET = 4
NUM_FEATURES=3 + 1 + 2 + 1 + CELLS_IN_BUCKET * NUM_BUCKETS
THETA_LEN=NUM_FEATURES-3


class Params(object):
    def __init__(self, X, Y, TMP=None):
        self.X = X
        self.Y = Y
        self.TMP = TMP


class ChildStuntedness(object):

    @staticmethod
    def cost(t, *args):
        params = args[0]

        X = params.X
        Y = params.Y

        M = X.shape[0]

        tmp = np.dot(X, t) - Y

        E = (np.sum(tmp ** 2) / 2.) / M

        return E

    @staticmethod
    def grad(t, *args):
        params = args[0]
        X = params.X
        Y = params.Y

        M = X.shape[0]

        tmp = np.dot(X, t) - Y

        grad = np.dot(tmp, X) / M

        return grad

    #def gd(self, theta)

    def train(self, data, for_weight=True):
        # data: id, w, b, 1, m, f, s1, s2, [time, f1-f8] * 10
        Y = data[:,2] if not for_weight else data[:,1]
        X = data[:,3:]
        params = Params(X, Y)

        local_theta = sp.rand(THETA_LEN)

        sol = spo.fmin_cg(ChildStuntedness.cost, local_theta, args=(params, ), fprime=ChildStuntedness.grad, maxiter=1000, full_output=False, disp=False, retall=False)

        return sol


    def get_data(self, training_str_arr, for_test=False):
        l = len(training_str_arr)
        if PROMOTE:
            columns = 14 if not for_test else 12
        else:
            columns = 14
        data = np.zeros((l, columns))

        for i in range(l):
            data[i,:] = np.fromstring(training_str_arr[i], dtype=float, sep=',')

        return data

    def get_features(self, data, for_test=False, feature_creator=None):
        u=np.unique(data[:,0])
        print "Unique IDs: %s" % str(u.shape)

        #
        features=sp.zeros((len(u), NUM_FEATURES))

        cur_id = -1
        cur_idx = -1
        cur_pos = 0

        for d in data:
            if d[FLD_ID] != cur_id:
                # switch to the new series
                cur_idx += 1
                cur_pos = 0
                cur_id = d[FLD_ID]
                # save id
                features[cur_idx, cur_pos]=cur_id
                cur_pos += 1
                # save weight
                features[cur_idx, cur_pos]=d[FLD_W] if not for_test else 0.
                cur_pos += 1
                # save birth time
                features[cur_idx, cur_pos]=d[FLD_B] if not for_test else 0.
                cur_pos += 1
                # X0 always 1
                features[cur_idx, cur_pos]=1.
                cur_pos += 1
                # save sex: 2 fields: 1st for male, 2nd for female
                features[cur_idx, cur_pos + d[FLD_SEX] ] = 1.
                cur_pos += 2
                # status
                features[cur_idx, cur_pos] = d[FLD_STATUS]
                cur_pos += 1

            bucket_idx = int(d[FLD_TIME] * 10)
            # just in case
            #bucket_idx = bucket_idx % NUM_BUCKETS

            # buckets start with idx=7 and each has CELLS_IN_BUCKET cells
            cur_pos = 7

            if feature_creator:
                feature_creator(features, cur_idx, cur_pos, bucket_idx*CELLS_IN_BUCKET, d, None)

        return features

    @staticmethod
    def get_sample(a, k):
        sample = a[0:k].copy()
        for i in range(k, a.shape[0]):
            r = np.random.randint(0, i)
            if r < k:
                sample[r] = a[i]
        return sample

    def calc(self, d, local_theta):
        result = np.dot(local_theta, d[3:])
        return result


    def MC(self, data_features, N, for_weight):

        min_err = 1000000
        min_theta = None

        for z in range(N):
            k = int(data_features.shape[0] *.8)
            train_data = ChildStuntedness.get_sample(data_features, k)
            indices = [i for i, id in enumerate(data_features[:,0]) if id not in train_data[0]]
            test_data = data_features[indices]

            local_theta = self.train(train_data, for_weight=for_weight)

            result = np.dot(test_data[:,3:], local_theta)

            M = test_data.shape[0]
            idx = 1 if for_weight else 2

            err = np.sum((result - test_data[:,idx])**2) / M

            #print err
            #print local_theta

            if not np.isnan(err) and not np.isinf(err) and err < min_err:
                min_err = err
                min_theta = local_theta

        print "Min err: ", min_err
        return min_theta, min_err


    def predict(self, training, testing):
        features_t = {9: {'DEG': [1]}, 4: {'DEG': [1]}, 6: {'DEG': [1]}, 7: {'DEG': [1]}}
##        {
##            8: {'DEG': [2.5]},
##            9: {'DIV': [10]},
##            10: {'MUL': [9]},
##            4: {'DIV_SUB': [4, 6], 'DEG': [2.5]},
##            7: {'DEG': [0.5]}}

        print "======================="
        print training
        print "======================="
        print testing

        features_w = {8: {'DIV': [6]}, 5: {'DEG': [-3]}, 6: {'MUL': [8]}}


        fc_t = ChildStuntedness.FeatureCreatorBase(features_t)
        fc_w = ChildStuntedness.FeatureCreatorBase(features_w)

        train_data = self.get_data(training, for_test=False)
        train_data = train_data[train_data[:,0].argsort()]

        train_features_t = self.get_features(train_data, feature_creator=fc_t)
        train_features_w = self.get_features(train_data, feature_creator=fc_w)
        train_data = None


        theta_t, err_t = self.MC(train_features_t, 1, False)
        theta_w, err_w = self.MC(train_features_w, 1, True)

        if np.isnan(err_t):
            print "ERROR err_t is nan", theta_t
        if np.isnan(err_w):
            print "ERROR err_w is nan", theta_w

        test_data = self.get_data(testing, for_test=True)
        test_data = test_data[test_data[:,0].argsort()]
        print "Test data: %s" % str(test_data.shape)

        test_features_t = self.get_features(test_data, for_test=True, feature_creator=fc_t)
        test_features_w = self.get_features(test_data, for_test=True, feature_creator=fc_w)
        test_data = None

        print "Features T: %s" % str(test_features_t.shape)
        print "Features W: %s" % str(test_features_w.shape)

        # predict
        result_len = test_features_t.shape[0] * 2
        result = [0.] * result_len

        u = np.unique(test_features_t[:,0])
        u.sort()

        result_idx = 0
        for id in u:
            d_t = test_features_t[test_features_t[:,0] == id]
            d_w = test_features_w[test_features_w[:,0] == id]

            t = self.calc(d_t[0], theta_t)
            w = self.calc(d_w[0], theta_w)

            result[result_idx*2 + 0] = t
            result[result_idx*2 + 1] = w
            result_idx += 1

        print result
        return result


        ss = np.argsort(test_features_t[:,0])

        result_idx = 0
        for i in ss:
            d_t = test_features_t[i]
            d_w = test_features_w[i]

            t = self.calc(d_t, theta_t)
            w = self.calc(d_w, theta_w)

            result[result_idx*2 + 0] = t
            result[result_idx*2 + 1] = w
            result_idx += 1

        print "Result: %s" % str(len(result))
        return result


    #### feature creators
    class FeatureCreatorBase(object):
        def __init__(self, features={FLD_F1 : {'D' : [1,2]}}):
            self.features = features


        def __call__(self, features, row_idx, cur_pos, bucket_pos, data, prev_data):
            for f, ops in self.features.items():
                if 777 == f:
                    continue
                for o, values in ops.items():
                    if o == 'NOP':
                        features[row_idx, cur_pos + bucket_pos] = data[f]
                        cur_pos += 1
                    elif o == 'AVR':
                        if None != prev_data:
                            features[row_idx, cur_pos + bucket_pos] += data[f] - prev_data[f]
                        else:
                            features[row_idx, cur_pos + bucket_pos] += data[f]
                        cur_pos += 1

                    elif o == 'DEG':
                        for v in values:
                            if v < 0 and data[f] == 0.:
                                features[row_idx, cur_pos + bucket_pos] = 0.
                            else:
                                features[row_idx, cur_pos + bucket_pos] = data[f]**v
                            cur_pos += 1
                    elif o == 'DIV':
                        for v in values:
                            features[row_idx, cur_pos + bucket_pos] = (data[f] / data[v] if data[v] != 0. else 0.)
                            cur_pos += 1
                    elif o == 'ADD':
                        for v in values:
                            features[row_idx, cur_pos + bucket_pos] = data[f] + data[v]
                            cur_pos += 1
                    elif o == 'SUB':
                        for v in values:
                            features[row_idx, cur_pos + bucket_pos] = data[f] - data[v]
                            cur_pos += 1
                    elif o == 'MUL':
                        for v in values:
                            features[row_idx, cur_pos + bucket_pos] = data[f] * data[v]
                            cur_pos += 1
                    elif o == 'DIV_SUB':
                            sub = values[0] - values[1];
                            features[row_idx, cur_pos + bucket_pos] = data[f] / sub if sub != 0. else 0.
                            cur_pos += 1
                    elif o == 'DIV_MUL':
                            sub = values[0] * values[1];
                            features[row_idx, cur_pos + bucket_pos] = data[f] / sub if sub != 0. else 0.
                            cur_pos += 1
                    elif o == 'LN':
                            features[row_idx, cur_pos + bucket_pos] = sp.log(data[f]) if data[f] > 0. else 0.
                            cur_pos += 1
                    else:
                        print "unknown operation: ", o










def get_errors(result, test_data):
    c = ChildStuntedness()
    data = c.get_data(test_data, False)

    u = np.unique(data[:,0])
    u.sort()

    tmp = np.zeros((u.shape[0], 3))
    idx = 0
    for id in u:
        i = np.argwhere(data[:,0] == id)[0]
        tmp[idx,0] = data[i,FLD_ID]
        tmp[idx,1] = data[i,FLD_B]
        tmp[idx,2] = data[i,FLD_W]
        idx += 1

    M = tmp.shape[0]

    ss = np.argsort(tmp[:,0])
    test_tmp = tmp[ss]

    err_w = 0.
    err_t = 0.

    for i in range(test_tmp.shape[0]):
        dt = result[i*2+0] - test_tmp[i,1]
        dw = result[i*2+1] - test_tmp[i,2]
        err_t += dt**2
        err_w += dw**2

    err_t /= M
    err_w /= M

    return err_t, err_w


def main():

    BASE_PATH='C:\\Temp\\tc01\\'
    DATA_PATH=BASE_PATH + 'data\\'

    train=[]
    with open(DATA_PATH+'train2.csv', "r") as fin:
        for line in fin:
            line = line.strip()
            train.append(line)

    test=[]
    with open(DATA_PATH+'test2.csv', "r") as fin:
        for line in fin:
            line = line.strip()
            test.append(line)

    print "Train: ", len(train)
    print "Test: ", len(test)

    c = ChildStuntedness()
    res = c.predict(train, test)

    print get_errors(res, test)


if __name__ == '__main__':
    main()
