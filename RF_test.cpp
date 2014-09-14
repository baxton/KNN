

// To test utils:
// g++ --std=c++0x -W -Wall -Wno-sign-compare -O2 -s -pipe -mmmx -msse -msse2 -msse3 pairs1.cpp -o pairs1.exe
//
// g++ -g -m32 --std=c++0x pairs1.cpp -o pairs1.exe
//


#include <cstdlib>
#include <time.h>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include<sstream>
#include <fstream>
#include <map>
#include <set>



using namespace std;


#define LOG(...) {cout << __VA_ARGS__ << endl;}



#define ALIGNMENT  __attribute__((aligned(16))


//
// Utils
//

//double abs(double d) {return d > 0. ? d : -d;}


namespace utils {

bool equal(double v1, double v2, double e = 0.0001) {
    if (::abs(v1 - v2) < e)
        return true;
    return false;
}

}   // utils






//
// Memory management
//
namespace memory {

template<class T>
struct DESTRUCTOR {
    static void destroy(T* p) {delete p;}
};

template<class T>
struct DESTRUCTOR_ARRAY {
    static void destroy(T* p) {delete [] p;}
};



template<class T, class D=DESTRUCTOR_ARRAY<T> >
struct ptr {
    T* p_;
    int* ref_;

    ptr() : p_(NULL), ref_(NULL) {}
    ptr(T* p) : p_(p), ref_(new int(1)) {}
    ptr(const ptr& other) {
        p_ = other.p_;
        ref_ = other.ref_;
        if (ref_)
            ++(*ref_);
    }

    ~ptr() {free();}

    ptr& operator=(const ptr& other) {
        ptr tmp = other;
        swap(tmp);
        return *this;
    }

    T* get() {return p_;}

    T& operator[] (int i) {
        return get()[i];
    }

    void free() {
        if (ref_) {
            if (0 == --(*ref_)) {
                D::destroy(p_);
                delete ref_;

                p_ = NULL;
                ref_ = NULL;
            }
        }
    }

    void swap(ptr& other) {
        T* tmp = p_;
        p_ = other.p_;
        other.p_ = tmp;

        int* tmp_i = ref_;
        ref_ = other.ref_;
        other.ref_ = tmp_i;
    }
};




}   // memory

//
// random numbers
//
struct random {
    static void seed(int s=-1) {
        if (s == -1)
            srand(time(NULL));
        else
            srand(s);
    }

    static int randint() {
        return rand();
    }

    static int randint(int low, int high) {
        int r = rand();
        r = r % (high - low) + low;
        return r;
    }

    static void randint(int low, int high, int* numbers, int size) {
        for (int i = 0; i < size; ++i)
            numbers[i] = rand() % (high - low) + low;
    }

    /*
     * Retuns:
     * k indices out of [0-n) range
     * with no repetitions
     */
    static void get_k_of_n(int k, int n, int* numbers) {
        for (int i = 0; i < k; ++i) {
            numbers[i] = i;
        }

        for (int i = k; i < n; ++i) {
            int r = randint(0, i+1);
            if (r < k) {
                numbers[r] = i;
            }
        }
    }

    template<class T>
    static void shuffle(T* buffer, int size) {
        for (int i = 0; i < size; ++i) {
            int r = randint(0, size);

            T tmp = buffer[i];
            buffer[i] = buffer[r];
            buffer[r] = tmp;
        }
    }

};


//
// Lin algebra
//
struct linalg {

    static void bootstrap(const double* __restrict__ X,
                          const double* __restrict__ Y,
                          int rows,
                          int columns,
                          double* __restrict__ bs_X,
                          double* __restrict__ bs_Y,
                          int bs_rows) {
        random::seed();
        int indices[bs_rows];
        random::randint(0, rows, indices, bs_rows);

        for (int x = 0; x < bs_rows; ++x) {
            copy(&X[ indices[x] * columns ], &bs_X[ x * columns ], columns);
            bs_Y[x] = Y[ indices[x] ];
        }
    }

    static void range(int low, int high, int* buffer) {
        for (int i = low; i < high; ++i) {
            buffer[i] = i;
        }
    }


    static void linspace(double min, double max, int num, double* buffer) {
        double delta = (max - min) / (num - 1.);

        buffer[0] = min;
        for (int i = 1; i < num-1; ++i) {
            buffer[i] = buffer[i-1]+delta;
        }
        buffer[num-1] = max;
    }

    // op with scalar
    static void mul(double scalar, const double* __restrict__ v, double* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] = v[i] * scalar;
        }
    }

    static void div(double scalar, const double* __restrict__ v, double* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] = v[i] / scalar;
        }
    }

    static void sub(double scalar, const double* __restrict__ v, double* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] = v[i] - scalar;
        }
    }

    static void add(double scalar, const double* __restrict__ v, double* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] = v[i] + scalar;
        }
    }

    static void pow(double scalar, const double* __restrict__ v, double* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] = ::pow(v[i], scalar);
        }
    }

    static void pow_inplace(double scalar, double* __restrict__ v, int size) {
        for (int i = 0; i < size; ++i) {
            v[i] = ::pow(v[i], scalar);
        }
    }

    static void mul_and_add(double scalar, const double* __restrict__ v, double* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] += v[i] * scalar;
        }
    }

    // vec to vec
    static void mul(const double* __restrict__ v1, const double* __restrict__ v2, double* r, int rows, int columns) {
        int N = rows * columns;
        for (int i = 0; i < N; ++i) {
            r[i] = v1[i] * v2[i];
        }
    }

    static void mul_and_add(const double* __restrict__ v1, const double* __restrict__ v2, double* r, int rows, int columns) {
        int N = rows * columns;
        for (int i = 0; i < N; ++i) {
            r[i] += v1[i] * v2[i];
        }
    }

    static void div(const double* __restrict__ v1, const double* __restrict__ v2, double* r, int rows, int columns) {
        int N = rows * columns;
        for (int i = 0; i < N; ++i) {
            r[i] = v1[i] / v2[i];
        }
    }

    static void sub(const double* __restrict__ v1, const double* __restrict__ v2, double* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] = v1[i] - v2[i];
        }
    }

    static void add(const double* __restrict__ v1, const double* __restrict__ v2, double* r, int rows, int columns) {
        int N = rows * columns;
        for (int i = 0; i < N; ++i) {
            r[i] = v1[i] + v2[i];
        }
    }

    static double sum(const double* __restrict__ v, int size) {
        double sum = 0.;
        for (int i = 0; i < size; ++i) {
            sum += v[i];
        }
        return sum;
    }

    // dot product
    static double dot(const double* __restrict__ v1, const double* __restrict__ v2, int size) {
        double r = 0.;
        for (int i = 0; i < size; ++i) {
            r += v1[i] * v2[i];
        }
        return r;
    }

    /* rows - number of rows in the matrix 'm' and elements in the vector 'r'
     * columns - number of columns in the matrix 'm' and elements in the vector 'v'
     */
    static void dot_m_to_v(const double* __restrict__ m, const double* __restrict__ v, double* r, int rows, int columns) {
        for (int row = 0; row < rows; ++row) {
            int begin = row * columns;
            double sum = 0.;
            for (int col = 0; col < columns; ++col) {
                sum += m[begin + col] * v[col];
            }
            r[row] = sum;
        }
    }

    static void dot_v_to_m(const double* __restrict__ v, const double* __restrict__ m, double* r, int rows, int columns) {
        set(r, 0., columns);
        for (int row = 0; row < rows; ++row) {
            mul_and_add(v[row], &m[row*columns], r, columns);
        }
    }


    static void min_max(const double* __restrict__ m, int rows, int columns, int idx, double& min, double& max) {
        min = 1000000000.;
        max = -1000000000.;
        for (int i = 0; i < rows; ++i) {
            if (min > m[i*columns+idx])
                min = m[i*columns+idx];
            if (max < m[i*columns+idx])
                max = m[i*columns+idx];
        }
    }


    static double mean(const double* __restrict__ v, int size) {
        double sum = 0.;
        for (int i = 0; i < size; ++i) {
            sum += v[i];
        }
        return sum / size;
    }

    static double* clone(const double* __restrict__ v, int size) {
        double * new_v = new double[size];
        for (int i = 0; i < size; ++i) {
            new_v[i] = v[i];
        }
        return new_v;
    }

    static void copy(const double* __restrict__ s, double* __restrict__ d, int size) {
        for (int i = 0; i < size; ++i) {
            d[i] = s[i];
        }
    }

    static double* zeros(int size) {
        return alloc_and_set(size, 0.);
    }

    static double* alloc_and_set(int size, double val) {
        double* v = new double[size];
        for (int i = 0; i < size; ++i) {
            v[i] = val;
        }
        return v;
    }

    static void set(double* __restrict__ v, double scalar, int size) {
        for (int i = 0; i < size; ++i)
            v[i] = scalar;
    }


    static void split_array(const double* origX, const double* origY, int rows, int columns,
                            int feature_idx, double feature_val,
                            double** leftX, double** leftY, int& left_rows,
                            double** rightX, double** rightY, int& right_rows) {
        left_rows = 0;
        right_rows = 0;
        for (int i = 0; i < rows; ++i) {
            if (origX[i * columns + feature_idx] <= feature_val) {
                ++left_rows;
            }
            else {
                ++right_rows;
            }
        }

        // allocate
        *leftX = new double[left_rows * columns];
        *leftY = new double[left_rows];
        *rightX = new double[right_rows * columns];
        *rightY = new double[right_rows];

        //
        int left_idx = 0;
        int right_idx = 0;
        for (int i = 0; i < rows; ++i) {
            if (origX[i * columns + feature_idx] <= feature_val) {
                copy(&origX[i*columns], &(*leftX)[left_idx*columns], columns);
                (*leftY)[left_idx] = origY[i];
                ++left_idx;
            }
            else {
                copy(&origX[i*columns], &(*rightX)[right_idx*columns], columns);
                (*rightY)[right_idx] = origY[i];
                ++right_idx;
            }
        }
    }

};


//
// Optimization
//
struct optimize {
    typedef void (*FUNC)(const double* theta, const double* X, const double* Y, double* cost, double* grad, int rows, int columns);

    static void minimize_gc(double* __restrict__ theta, const double* X, int rows, int columns, const double* Y, FUNC func, int max_iterations) {
        double cost = 0.;
        double grad[columns];

        double e = 0.0001;
        double a = .4;

        func(theta, X, Y, &cost, grad, rows, columns);

        int cur_iter = 0;

        while (cost > e && cur_iter < max_iterations) {
            ++cur_iter;

            for (int i = 0; i < columns; ++i) {
                theta[i] = theta[i] - a * grad[i];
            }

            double new_cost;
            func(theta, X, Y, &new_cost, grad, rows, columns);

            if (cost < new_cost)
                a /= 2.;

            cost = new_cost;
        }
    }

    // for simple func: 1/M * 1/2 * SUM( (H - Y)**2 )
    static void quadratic_cost(const double* theta, const double* X, const double* Y, double* cost, double* grad, int rows, int columns) {
        double M = rows;

        memory::ptr<double> tmp( linalg::zeros(rows) );
        linalg::dot_m_to_v(X, theta, tmp.get(), rows, columns);

        memory::ptr<double> deltas( linalg::zeros(rows) );
        linalg::sub(tmp.get(), Y, deltas.get(), rows);

        linalg::pow(2., deltas.get(), tmp.get(), rows);
        *cost = (linalg::sum(tmp.get(), rows) / 2.) / M;

        linalg::dot_v_to_m(deltas.get(), X, grad, rows, columns);
        linalg::div(M, grad, grad, columns);
    }

    //
    // Logistic regression
    //
    static void sigmoid(const double* __restrict__ x, double* r, int size, bool correct_borders=true) {
        for (int i = 0; i < size; ++i) {
            r[i] = 1. / (1. + ::exp(-x[i]));
            if (correct_borders) {
                if (r[i] == 1.)
                    r[i] = .99999999;
                else if (r[i] == 0.)
                    r[i] = .00000001;
            }
        }
    }

    static void logistic_h(const double* __restrict__ theta, const double* __restrict__ X, double* r, int rows, int columns) {
        linalg::dot_m_to_v(X, theta, r, rows, columns);
        sigmoid(r, r, rows);
    }

    // for logistic cos func: sigmoid( h(X) )
    static void logistic_cost(const double* theta, const double* X, const double* Y, double* cost, double* grad, int rows, int columns) {
        double M = rows;

        memory::ptr<double> h = linalg::zeros(rows);
        logistic_h(theta, X, h.get(), rows, columns);

        double E = 0.;
        for (int i = 0; i < rows; ++i) {
            E += (-Y[i]) * ::log(h.get()[i]) - (1. - Y[i]) * ::log(1. - h.get()[i]);
        }
        E /= M;

        *cost = E;

        memory::ptr<double> deltas = linalg::zeros(rows);
        linalg::sub(h.get(), Y, deltas.get(), rows);
        linalg::dot_v_to_m(deltas.get(), X, grad, rows, columns);
    }
};      // optimize



//
// RF node val
//

struct node_val_base {
    virtual ~node_val_base(){};
    virtual double get_val(const double* v, int size) = 0;
};

struct node_val_mean : public node_val_base {
    double val_;

    node_val_mean(const double* X,
                  const double* Y,
                  int rows,
                  int columns) {
        val_ = linalg::mean(Y, rows);
    }
    virtual ~node_val_mean() {}

    virtual double get_val(const double*, int) {
        return val_;
    }
};

struct node_val_linear_regression : public node_val_base {

    memory::ptr<double> theta_;

    node_val_linear_regression(const double* X,
                               const double* Y,
                               int rows,
                               int columns) {
        memory::ptr<double> tmp = linalg::zeros(rows*(columns+1));
        for (int i = 0; i < rows; ++i) {
            tmp.get()[i*(columns+1) + 0] = 1.;
            linalg::copy(&X[i*columns], &(tmp.get()[i*(columns+1) + 1]), columns);
        }

        theta_ = memory::ptr<double>(new double[columns+1]);
        for (int i = 0; i < columns+1; ++i)
            theta_.get()[i] = ((double)random::randint(0, 1000) / 1000.) - .5;
        optimize::minimize_gc(theta_.get(), X, rows, columns, Y, optimize::quadratic_cost, 100);
    }

    virtual ~node_val_linear_regression() {}

    virtual double get_val(const double* v, int size) {
        if (!v)
            return -1.;

        double tmp[size+1];
        tmp[0] = 1.;
        linalg::copy(v, &tmp[1], size);
        double p = linalg::dot(tmp, theta_.get(), size);
        return p;
    }
};

struct node_val_logistic_regression : public node_val_base {

    memory::ptr<double> theta_;

    node_val_logistic_regression(const double* X,
                               const double* Y,
                               int rows,
                               int columns) {
        memory::ptr<double> tmp = linalg::zeros(rows*(columns+1));
        for (int i = 0; i < rows; ++i) {
            tmp.get()[i*(columns+1) + 0] = 1.;
            linalg::copy(&X[i*columns], &(tmp.get()[i*(columns+1) + 1]), columns);
        }

        theta_ = memory::ptr<double>(new double[columns+1]);
        for (int i = 0; i < columns+1; ++i)
            theta_.get()[i] = ((double)random::randint(0, 1000) / 1000.) - .5;
        optimize::minimize_gc(theta_.get(), X, rows, columns, Y, optimize::logistic_cost, 1000);
    }

    virtual ~node_val_logistic_regression() {}

    virtual double get_val(const double* v, int size) {
        if (!v)
            return -1.;

        double tmp[size+1];
        tmp[0] = 1.;
        linalg::copy(v, &tmp[1], size);
        double r;
        optimize::logistic_h(theta_.get(), tmp, &r, 1, size+1);

        return r;
    }
};


//
//
//

class Counter {
    int count;
public:
    Counter() : count(0) {}

    ~Counter() {}

    int next() {
        return count++;
    }
};

//
// RF split criteria
//

struct estimator_regressor {
    double operator() (const double* __restrict__ X,
                       const double* __restrict__ Y,
                       int rows,
                       int columns,
                       int feature_idx,
                       int lnum,
                       double& feature_val,
                       int& left_rows,
                       int& right_rows)
    {
        double N = rows;     // number of elements in the original Y array
        double x_min, x_max;
        linalg::min_max(X, rows, columns, feature_idx, x_min, x_max);

        double max_gain = -1.;
        double max_val  = -1.;

        int num = 500;
        double buffer[num];
        linalg::linspace(x_min, x_max, num, buffer);

        for (int i = 0; i < num; ++i) {
            double left_M = 0.;  // number of elements in the left part
            double right_M = 0.; // number of elements in the right part

            double mean = 0.;
            double left_mean = 0.;
            double right_mean = 0.;

            double x = buffer[i];

            for (int r = 0; r < rows; ++r) {
                int idx = r * columns + feature_idx;
                if (X[idx] <= x) {
                    ++left_M;
                    left_mean += Y[r];
                }
                else {
                    ++right_M;
                    right_mean += Y[r];
                }
                mean += Y[r];
            }

            if (lnum > left_M || lnum > right_M)
                continue;

            mean /= N;
            left_mean /= left_M;
            right_mean /= right_M;


            double sum = 0.;
            double left_sum = 0.;
            double right_sum = 0.;

            for (int r = 0; r < rows; ++r) {
                int idx = r * columns + feature_idx;
                if (X[idx] <= x) {
                    left_sum += (Y[r] - left_mean)*(Y[r] - left_mean);
                }
                else {
                    right_sum += (Y[r] - right_mean)*(Y[r] - right_mean);
                }
                sum += (Y[r] - mean)*(Y[r] - mean);
            }

            double gain = sum - (left_sum + right_sum);
            if (!isnan(gain) && !isinf(gain) && (max_gain == -1 || max_gain < gain)) {
                max_gain = gain;
                max_val  = x;
                left_rows = left_M;
                right_rows = right_M;
            }
        }

        feature_val = max_val;
        return max_gain;
    }
};

/*
 * Decision tree class
 */
template<class ESTIMATOR=estimator_regressor, class NODE_VAL=node_val_mean>
class dtree {
    memory::ptr<Counter> counter_;

    int k_;
    int lnum_;
    dtree* left_;
    dtree* right_;

    int feature_idx_;
    double feature_val_;

    node_val_base* pVal_;

    int id_;

public:
    dtree() :
        counter_(new Counter()),
        k_(1),
        lnum_(2),
        left_(NULL),
        right_(NULL),
        feature_idx_(-1),
        feature_val_(-1.),
        pVal_(NULL),
        id_(counter_.get()->next())
    {}

    dtree(int k, int lnum, memory::ptr<Counter>& counter) :
        counter_(counter),
        k_(k),
        lnum_(lnum),
        left_(NULL),
        right_(NULL),
        feature_idx_(-1),
        feature_val_(-1.),
        pVal_(NULL),
        id_(counter_.get()->next())
    {}

    ~dtree() {free();}

    void asarray(map<int, vector<double> >& node_set) {
        vector<double> res;
        if (-1 != feature_idx_) {
            res.push_back(0);
            res.push_back(id_);
            res.push_back(feature_idx_);
            res.push_back(feature_val_);
            res.push_back(left_->id_);
            res.push_back(right_->id_);

            node_set[id_] = res;

            left_->asarray(node_set);
            right_->asarray(node_set);
        }
        else{
            res.push_back(1);
            res.push_back(id_);
            res.push_back(pVal_->get_val(NULL, 0));
            res.push_back(0);
            res.push_back(0);
            res.push_back(0);

            node_set[id_] = res;
        }
    }

    ostream& print(ostream& os) {
        map<int, vector<double> > node_set;
        asarray(node_set);

        for (map<int, vector<double> >::iterator it = node_set.begin(); it != node_set.end(); ++it) {
            vector<double>& vec = it->second;

            os << "  {";
            for (int i = 0; i < 6; ++i) {
                os << vec[i] << ", ";
            }
            os << "}," << endl;
        }

        return os;
    }


    void free() {
        if (pVal_) {
            delete pVal_;
            pVal_ = NULL;
        }

        if (left_) {
            left_->free();
            delete left_;
            left_ = NULL;
        }

        if (right_) {
            right_->free();
            delete right_;
            right_ = NULL;
        }
    }

    void set_K(int k) { k_ = k; }
    void set_LNUM(int lnum) { lnum_ = lnum; }


    void get_best_split(const double* __restrict__ X,
                        const double* __restrict__ Y,
                        int rows,
                        int columns,
                        int& best_feature_idx, double& best_val, double& best_score, int* feature_indices) {

        ESTIMATOR split;

        for (int f = 0; f < k_; ++f) {
            int feature_idx = feature_indices[f];

            double fature_val;
            int left_rows = 0, right_rows = 0;

            double score = split(X, Y, rows, columns, feature_idx, lnum_, fature_val, left_rows, right_rows);

            if (score != -1 && (best_score == -1 || score > best_score)) {
                best_feature_idx = feature_idx;
                best_val = fature_val;
                best_score = score;
            }
        }
    }


    void fit(const double* __restrict__ X,
             const double* __restrict__ Y,
             int rows,
             int columns) {

        // select features
        int feature_indices[k_];
        random::get_k_of_n(k_, columns, feature_indices);


        // select the best split
        int    best_feature_idx = -1;
        double best_val         = -1;
        double best_score       = -1;
        get_best_split(X, Y, rows, columns, best_feature_idx, best_val, best_score, feature_indices);


        // split the node
        if (-1 != best_feature_idx) {
            feature_idx_ = best_feature_idx;
            feature_val_ = best_val;

            left_ = new dtree<ESTIMATOR, NODE_VAL>(k_, lnum_, counter_);
            right_ = new dtree<ESTIMATOR, NODE_VAL>(k_, lnum_, counter_);

            double *leftX, *leftY;
            double *rightX, *rightY;
            int left_rows, right_rows;
            linalg::split_array(X, Y, rows, columns, feature_idx_, feature_val_, &leftX, &leftY, left_rows, &rightX, &rightY, right_rows);


            left_->fit(leftX, leftY, left_rows, columns);
            right_->fit(rightX, rightY, right_rows, columns);

            delete [] leftX;
            delete [] leftY;
            delete [] rightX;
            delete [] rightY;
        }
        else {
            // for now I use mean for regression
            pVal_ = new NODE_VAL(X, Y, rows, columns);
        }
    }

    double predict(const double* v, int size) {
        if (-1 != feature_idx_) {
            if (v[feature_idx_] <= feature_val_)
                return left_->predict(v, size);
            else
                return right_->predict(v, size);
        }

        return pVal_->get_val(v, size);
    }

private:

};


//
// RandomForest
//

template<class ESTIMATOR, class NODE_VAL>
class RF {
    int TREES;
    int K;
    int LNUM;

    dtree<ESTIMATOR, NODE_VAL>* forest;

public:
    RF(int trees, int k, int lnum) :
        TREES(trees),
        K(k),
        LNUM(lnum)
    {
        forest = new dtree<ESTIMATOR, NODE_VAL>[TREES];
        for (int t = 0; t < TREES; ++t) {
            forest[t].set_K(K);
            forest[t].set_LNUM(LNUM);
        }
    }

    ~RF() {
        delete [] forest;
    }

    void fit(const double* __restrict__ X,
             const double* __restrict__ Y,
             int rows,
             int columns) {

        int bs_rows = (int)(rows * .8);     // bootstrap rows
        memory::ptr<double> new_X = linalg::zeros(bs_rows * columns);
        memory::ptr<double> new_Y = linalg::zeros(bs_rows);

        for (int t = 0; t < TREES; ++t) {
            //LOG("Fit tree #" << t)
            linalg::bootstrap(X, Y, rows, columns, new_X.get(), new_Y.get(), bs_rows);

            dtree<ESTIMATOR, NODE_VAL>& tree = forest[t];
            tree.fit(new_X.get(), new_Y.get(), bs_rows, columns);
        }
    }

    double predict(const double* __restrict__ v, int size) {
        double p = 0.;

        for (int t = 0; t < TREES; ++t) {
            //LOG("Predict tree #" << t)
            p += forest[t].predict(v, size);
        }

        return p / TREES;
    }

    ostream& print(ostream& os) {
        os << "namespace RF {" << endl;
        os << "const int VEC_SIZE = 6;" << endl;
        os << "const int TREES = " << TREES << ";" << endl;

        for (int t = 0; t < TREES; ++t) {
            os << "double tree_" << t << "[][VEC_SIZE] = {" << endl;
            forest[t].print(os);
            os << "};" << endl;
        }

        os << "double* forest[] = {";
        for (int t = 0; t < TREES; ++t) {
            os << "&tree_" << t << "[0][0], ";
        }
        os << "};" << endl;

        //os << "};   // RF" << endl;

        return os;
    }


private:
    RF(const RF&);
    RF& operator= (const RF&);
};




/////////////////////////////////////////////////////////////////////////
// Testing
/////////////////////////////////////////////////////////////////////////

//Total positives: 99; Total negatives: 63
//Sens: 0.987654, prec: 0.808081, F1: 0.888889, Total ones: 81, Total: 162
//500 5 3 0.3

//Total positives: 95; Total negatives: 67
//Sens: 0.975309, prec: 0.831579, F1: 0.897727, Total ones: 81, Total: 162
//500 5 3 0.4

// with Logistic reg
//Total positives: 80; Total negatives: 82
//Sens: 0.851852, prec: 0.8625, F1: 0.857143, Total ones: 81, Total: 162
//40 20 50 0.48
//
//Total positives: 80; Total negatives: 82
//Sens: 0.851852, prec: 0.8625, F1: 0.857143, Total ones: 81, Total: 162
//40 20 50 0.48

const string fname_train = "C:\\Temp\\asteroid\\vectors\\pairs_train.txt";
const string fname_test = "C:\\Temp\\asteroid\\vectors\\pairs_test.txt";

const double ROWS_FOR_TRAIN = .6;

const int X_COLUMNS = 27 * 2;


const int TREES = 1024*2;
const int K = 20;
const int LNUM = 3;
const double THRESHOLD = .5;

const int RA_IDX_1 = 25;
const int RA_IDX_2 = X_COLUMNS - 2;
const int DEC_IDX_1 = 26;
const int DEC_IDX_2 = X_COLUMNS - 1;

void load_data(vector<double>& X, vector<double>& Y, const string& fname) {

    ifstream fin;
    fin.open(fname, ifstream::in);

    string line;
    while (std::getline(fin, line)) {
        if (line.empty())
            continue;

        // X_COLUMNS * 4 + 1
        // 1,382,1485,2340,2460,2540,2370,2270,2580,2980,2940,2590,2410,2660,3220,3120,2740,2520,2380,2660,2640,2560,2620,2295,2365,2345,2415,2615,
        double tmp;
        char c;

        istringstream is(line);
        is >> tmp >> c;
        Y.push_back(tmp);

        for (int i = 0; i < X_COLUMNS; ++i) {
            is >> tmp >> c;
            X.push_back(tmp);
        }
    }

    LOG("X: " << X.size())
    LOG("Y: " << Y.size())
}

template<class ESTIMATOR, class NODE_VAL>
void test(RF<ESTIMATOR, NODE_VAL>& rf, const double* X, const double* Y, int rows, int columns) {

    double TP = 0.;
    double FP = 0.;
    double TN = 0.;
    double FN = 0.;
    double YSUM = 0.;

    for (int v = 0; v < rows; ++v) {
        double p = rf.predict(&X[v * columns], columns);

        p = p > THRESHOLD ? 1. : 0.;

        if (p == 1.      and Y[v] == 1.)
            TP += 1.;
        else if (p == 1. and Y[v] == 0.)
            FP += 1.;
        else if (p == 0. and Y[v] == 1.)
            FN += 1.;
        else if (p == 0. and Y[v] == 0.)
            TN += 1.;

        YSUM += Y[v];
    }

    LOG("Total positives: " << (TP + FP) << "; Total negatives: " << (TN + FN) )

    double sensitivity = TP / (TP + FN);
    double precision    = TP / (TP + FP);
    double F1 = 2. * (sensitivity * precision) / (sensitivity + precision);

    LOG("Sens: " << sensitivity << ", prec: " << precision << ", F1: " << F1 << ", Total ones: " << YSUM << ", Total: " << rows)

}

int main() {

    vector<double> train_X;
    vector<double> train_Y;
    load_data(train_X, train_Y, fname_train);

    int rows = train_Y.size();
    int columns = X_COLUMNS;

    LOG("Train X: " << train_X.size())
    LOG("Train Y: " << train_Y.size())

    //  TREES, K, LNUM, THRESHOLD
    //RF<estimator_regressor, node_val_logistic_regression> rf(TREES, K, LNUM);
    //RF<estimator_regressor, node_val_linear_regression> rf(TREES, K, LNUM);
    RF<estimator_regressor, node_val_mean> rf(TREES, K, LNUM);

    // teach
    rf.fit(&train_X[0], &train_Y[0], rows, columns);


    //rf.print(cout);

    //
    // Test
    //
    vector<double> test_X;
    vector<double> test_Y;
    load_data(test_X, test_Y, fname_test);

    rows = test_Y.size();
    columns = X_COLUMNS;

    test(rf, &test_X[0], &test_Y[0], rows, columns);
    LOG(TREES << " " << K << " " << LNUM << " " << THRESHOLD)

    return 0;
}
