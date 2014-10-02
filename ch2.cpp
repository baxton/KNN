

// To test utils:
// g++ --std=c++0x -W -Wall -Wno-sign-compare -O2 -s -pipe -mmmx -msse -msse2 -msse3 process.cpp -o process.exe
//
// g++ -g process.cpp -o process.exe
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
#include <algorithm>



using namespace std;


#define LOG(...) {cout << __VA_ARGS__ << endl;}



#define NOVAL -99.


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


struct counters {
    double min;
    double max;
    double mean;
    int    cnt;

    counters() :
        min(NOVAL),
        max(0.),
        mean(0.),
        cnt(0)
    {}

};


struct age_val_map {
    typedef map<double, vector<counters> > MAP;
    typedef map<double, vector<counters> >::iterator ITER;

    MAP storage;

    MAP& get() { return storage; }

    void add(const vector<double>& vec) {
        // age
        double k = roundf(vec[1] * 10000) / 10000;

        ITER it = storage.find(k);
        if (it != storage.end()) {
            for (int i = 0; i < 16; ++i) {
                if (vec[i] != NOVAL) {
                    it->second[i].mean += vec[i];
                    it->second[i].cnt += 1;
                }
            }
        }
        else  {
            vector<counters> cnts;
            for (int i = 0; i < 16; ++i) {
                cnts.push_back(counters());
            }
            for (int i = 0; i < 16; ++i) {
                if (vec[i] != NOVAL) {
                    cnts[i].mean = vec[i];
                    cnts[i].cnt = 1;
                }
            }

            storage[k] = cnts;
        }
    }

    map<double, vector<double> > as_vectors() {
        static double means[] = {839.551, -0.443631, -0.443048, 1.51245, -0.162933, -0.0821803, 0.995769, 0.619883, 0.00493325, -0.0546572, -0.0356498, 4.73872, -0.0217742, -0.354779, -0.406458, -0.223038};

        map<double, vector<double> > age_val;
        vector<double> tmp;

        enum {
            TOKENS_NUM = 16,

            ID_IDX         = 0,     // ID
            AGEDAYS_IDX    = 1,
            GAGEDAYS_IDX   = 2,
            SEX_IDX        = 3,
            MUACCM_IDX     = 4,
            SFTMM_IDX      = 5,
            BFED_IDX       = 6,
            WEAN_IDX       = 7,
            GAGEBRTH_IDX   = 8,
            MAGE_IDX       = 9,
            MHTCM_IDX      = 10,
            MPARITY_IDX    = 11,
            FHTCM_IDX      = 12,
            WTKG_IDX       = 13,    // DV
            LENCM_IDX      = 14,    // DV
            HCIRCM_IDX     = 15,    // DV

            X_COLUMNS      = 12,

        };


        for (ITER it = storage.begin(); it != storage.end(); ++it) {
            for (int i = 0; i < 16; ++i) {
                counters& cnts = it->second[i];
                if (i == SEX_IDX) {
                    double m = means[SEX_IDX];
                    if (cnts.cnt) {
                        m = cnts.mean / cnts.cnt;
                    }
                    tmp.push_back(m < 1.5 ? 1 : 2);
                }
                else if (i == BFED_IDX) {
                    double m = means[BFED_IDX];
                    if (cnts.cnt) {
                        m = cnts.mean / cnts.cnt;
                    }
                    tmp.push_back(m < .5 ? 0 : 1);
                }
                else if (i == WEAN_IDX) {
                    double m = means[WEAN_IDX];
                    if (cnts.cnt) {
                        m = cnts.mean / cnts.cnt;
                    }
                    tmp.push_back(m < .5 ? 0 : 1);
                }
                else if (i == FHTCM_IDX) {
                    double m = means[FHTCM_IDX];
                    if (cnts.cnt) {
                        m = cnts.mean / cnts.cnt;
                    }
                    tmp.push_back(m);
                }
                else {
                    if (cnts.cnt) {
                        tmp.push_back(cnts.mean / cnts.cnt);
                    }
                    else {
                        tmp.push_back(means[i]);
                    }
                }
            }

            age_val[ it->first ] = tmp;
            tmp.clear();
        }
        return age_val;
    }
};



map <double, vector<double> > means(age_val_map& avm) {
    return avm.as_vectors();
}

double get_val(double k, int idx, age_val_map& m) {
    static double means[] = {839.551, -0.443631, -0.443048, 1.51245, -0.162933, -0.0821803, 0.995769, 0.619883, 0.00493325, -0.0546572, -0.0356498, 4.73872, -0.0217742, -0.354779, -0.406458, -0.223038};
    static map <double, vector<double> > avm = utils::means(m);

    double r = means[idx];

    map <double, vector<double> >::iterator it = avm.lower_bound(k);
    if (it != avm.end()) {
        int cnt = 1;
        r = it->second[idx];
        map <double, vector<double> >::iterator it2 = it; ++it2;
        map <double, vector<double> >::iterator it3 = it; ++it3;

        if (it2 != avm.end()) {
            r += it2->second[idx];
            ++cnt;
        }

        if (it3 != avm.end()) {
            r += it3->second[idx];
            ++cnt;
        }

        r /= cnt;
    }

    return r;
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
        optimize::minimize_gc(theta_.get(), X, rows, columns, Y, optimize::quadratic_cost, 1000);
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
            if (/*!isnan(gain) &&  !isinf(gain) &&*/ (max_gain == -1 || max_gain < gain)) {
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
    memory::ptr<Counter, memory::DESTRUCTOR<Counter> > counter_;

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

    dtree(int k, int lnum, memory::ptr<Counter, memory::DESTRUCTOR<Counter> >& counter) :
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

    ~dtree() {
        free();
    }

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
//
/////////////////////////////////////////////////////////////////////////

class ChildStuntedness2 {

    enum {
        TOKENS_NUM = 16,

        ID_IDX         = 0,     // ID
        AGEDAYS_IDX    = 1,
        GAGEDAYS_IDX   = 2,
        SEX_IDX        = 3,
        MUACCM_IDX     = 4,
        SFTMM_IDX      = 5,
        BFED_IDX       = 6,
        WEAN_IDX       = 7,
        GAGEBRTH_IDX   = 8,
        MAGE_IDX       = 9,
        MHTCM_IDX      = 10,
        MPARITY_IDX    = 11,
        FHTCM_IDX      = 12,
        WTKG_IDX       = 13,    // DV
        LENCM_IDX      = 14,    // DV
        HCIRCM_IDX     = 15,    // DV

        X_COLUMNS      = 12,

    };

/*
    RF<estimator_regressor, node_val_linear_regression> rf_w;
    RF<estimator_regressor, node_val_linear_regression> rf_l;
    RF<estimator_regressor, node_val_linear_regression> rf_h;
*/

    RF<estimator_regressor, node_val_mean> rf_w;
    RF<estimator_regressor, node_val_mean> rf_l;
    RF<estimator_regressor, node_val_mean> rf_h;

    vector<double> X;
    vector<double> Y1;
    vector<double> Y2;
    vector<double> Y3;

    utils::age_val_map avm;


public:
    ChildStuntedness2() :
        rf_w(10, 5, 5),
        rf_l(10, 5, 5),
        rf_h(10, 5, 5),
        X(),
        Y1(),
        Y2(),
        Y3()
    {
        X.reserve(X_COLUMNS*1024*sizeof(double));
        Y1.reserve(1024*sizeof(double));
        Y2.reserve(1024*sizeof(double));
        Y3.reserve(1024*sizeof(double));
    }
    ~ChildStuntedness2() {}

    void add_to_data(const vector<double>& vec) {

        X.push_back( vec[AGEDAYS_IDX] );     // 1
        X.push_back( vec[GAGEDAYS_IDX] );    // 2
        X.push_back( vec[SEX_IDX] );
        X.push_back( vec[MUACCM_IDX] );
        X.push_back( vec[SFTMM_IDX] );
        X.push_back( vec[BFED_IDX] );
        X.push_back( vec[WEAN_IDX] );
        X.push_back( vec[GAGEBRTH_IDX] );
        X.push_back( vec[MAGE_IDX] );
        X.push_back( vec[MHTCM_IDX] );
        X.push_back( vec[MPARITY_IDX] );
        X.push_back( vec[FHTCM_IDX] );       // 12

        Y1.push_back( vec[WTKG_IDX] );       // 13
        Y2.push_back( vec[LENCM_IDX] );      // 14
        Y3.push_back( vec[HCIRCM_IDX] );     // 15
    }

    std::vector<std::string> predict(const std::vector<std::string>& train_str, const std::vector<std::string>& test_str) {
        //
        // train
        //
        vector<double> tmp;

        int size = train_str.size();
        for (int i = 0; i < size; ++i) {
            line2vec(train_str[i], tmp);

            avm.add(tmp);

            add_to_data(tmp);
        }
        tmp.clear();

        int rows = Y1.size();

        preproc();

        rf_w.fit(&X[0], &Y1[0], rows, X_COLUMNS);
        rf_l.fit(&X[0], &Y2[0], rows, X_COLUMNS);
        rf_h.fit(&X[0], &Y3[0], rows, X_COLUMNS);

        X.clear();
        Y1.clear();
        Y2.clear();
        Y3.clear();

        //
        // now do prediction
        //
        std::vector<std::string> result;

        stringstream ss;

        LOG("predicting")

        size = test_str.size();
        for (int i = 0; i < size; ++i) {
            line2vec(test_str[i], tmp);
            add_to_data(tmp);
            preproc();

            ss << rf_w.predict( &X[0], X_COLUMNS );
            ss << "," << rf_l.predict( &X[0], X_COLUMNS );
            ss << "," << rf_h.predict( &X[0], X_COLUMNS );

            result.push_back(ss.str());
            ss.str("");

            X.clear();
        }

        return result;
    }


private:
    ChildStuntedness2(const ChildStuntedness2&);
    ChildStuntedness2& operator= (const ChildStuntedness2&);


    void preproc() {
/*
        ID_IDX         = 0,     // ID

        IN X:
        AGEDAYS_IDX    = 1,
        GAGEDAYS_IDX   = 2,
        SEX_IDX        = 3,
        MUACCM_IDX     = 4,
        SFTMM_IDX      = 5,
        BFED_IDX       = 6,
        WEAN_IDX       = 7,
        GAGEBRTH_IDX   = 8,
        MAGE_IDX       = 9,
        MHTCM_IDX      = 10,
        MPARITY_IDX    = 11,
        FHTCM_IDX      = 12,

        IN Ys:
        WTKG_IDX       = 13,    // DV
        LENCM_IDX      = 14,    // DV
        HCIRCM_IDX     = 15,    // DV
*/

        int size = X.size() / X_COLUMNS;

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < 12; ++j) {
                if (NOVAL == X[i*X_COLUMNS + j]) {
                    X[i*X_COLUMNS + j] = utils::get_val(X[i*X_COLUMNS + 0], j+1, avm);
                }
            }

            //

            X[i*X_COLUMNS + AGEDAYS_IDX - 1] = 0.;
        }


    }


    void line2vec(const string& line, vector<double>& vec) {
        char sep = ',';

        vec.clear();

        size_t old_pos = 0;
        size_t pos = line.find(sep, old_pos);

        int idx = 0;

        while (std::string::npos != pos) {
            string ss = line.substr(old_pos, pos - old_pos);

            if ("." == ss) {
                vec.push_back(NOVAL);
            }
            else {
                vec.push_back( atof(ss.c_str()) );
            }

            old_pos = pos + 1;
            pos = line.find(sep, old_pos);

            ++idx;
        }

        if (old_pos != std::string::npos && old_pos < line.length()) {
            string ss = line.substr(old_pos);

            if ("." == ss) {
                vec.push_back(NOVAL);
            }
            else {
                vec.push_back( atof(ss.c_str()) );
            }
        }
    }

};




/////////////////////////////////////////////////////////////////////////
// Testing
/////////////////////////////////////////////////////////////////////////



void line2vec(const string& line, vector<double>& vec) {
    char sep = ',';

    vec.clear();

    size_t old_pos = 0;
    size_t pos = line.find(sep, old_pos);

    int idx = 0;

    while (std::string::npos != pos) {
        string ss = line.substr(old_pos, pos - old_pos);

        if ("." == ss) {
            vec.push_back(0);
        }
        else {
            vec.push_back( atof(ss.c_str()) );
        }

        old_pos = pos + 1;
        pos = line.find(sep, old_pos);

        ++idx;
    }

    if (old_pos != std::string::npos && old_pos < line.length()) {
        string ss = line.substr(old_pos);

        if ("." == ss) {
            vec.push_back(0);
        }
        else {
            vec.push_back( atof(ss.c_str()) );
        }
    }
}


int process_result(const vector<string>& preds, const vector<string>& y) {
    int size = y.size();

    int WTKG_IDX       = 13;    // DV
    int LENCM_IDX      = 14;    // DV
    int HCIRCM_IDX     = 15;    // DV

    double sse_w = 0.;
    double sse_l = 0.;
    double sse_h = 0.;

    for (int i = 0; i < size; ++i) {
        vector<double> pp;
        line2vec(preds[i], pp);

        double pw = pp[0];
        double pl = pp[1];
        double ph = pp[2];

        vector<double> tmp;
        line2vec(y[i], tmp);

        sse_w += (pw - tmp[WTKG_IDX]) * (pw - tmp[WTKG_IDX]);
        sse_l += (pl - tmp[LENCM_IDX]) * (pl - tmp[LENCM_IDX]);
        sse_h += (ph - tmp[HCIRCM_IDX]) * (ph - tmp[HCIRCM_IDX]);
    }

    sse_w /= size;
    sse_l /= size;
    sse_h /= size;

    sse_w = sqrt(sse_w);
    sse_l = sqrt(sse_w);
    sse_h = sqrt(sse_w);

    LOG("SSE W: " << sse_w)
    LOG("SSE L: " << sse_l)
    LOG("SSE H: " << sse_h)
}


int main(int, const char**) {
    //string fname = "/home/maxim/ch2/data/exampleData.csv";
    string fname = "C:\\Temp\\ch2\\data\\exampleData.csv";

    ifstream fin;
    fin.open(fname.c_str(), ifstream::in);

    vector<string> tmp;
    string line;
    while (std::getline(fin, line)) {
        tmp.push_back(line);
    }
    LOG("file reading done")

    int N = tmp.size();
    int k = (int)(N * .6);

    random::seed();

    int indices[k];
    random::get_k_of_n(k, N, indices);
    std::sort(&indices[0], &indices[k]);

    vector<string> train_str;
    vector<string> test_str;

    int idx = 0;
    for (int i = 0; i < N; ++i) {
        if (i == indices[idx]) {
            ++idx;
            train_str.push_back(tmp[i]);
        }
        else {
            test_str.push_back(tmp[i]);
        }
    }
    tmp.clear();
    LOG("data splitting done")

    //
    // test
    //

    ChildStuntedness2 o;
    LOG("init done")
    vector<string> result = o.predict(train_str, test_str);

    for (int i = 0; i < test_str.size(); ++i) {
        size_t pos = test_str[i].find(',');
        LOG("[" << test_str[i].substr(0,pos) << "] " << result[i])
    }

    process_result(result, test_str);

    LOG("done")

    return 0;
}

