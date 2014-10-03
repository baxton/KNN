

// To test utils:
// g++ --std=c++0x -W -Wall -Wno-sign-compare -O2 -s -pipe -mmmx -msse -msse2 -msse3 process.cpp -o process2.exe
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


#define SUBMISSION


using namespace std;


#define LOG(...) {cout << __VA_ARGS__ << endl;}




#define        TOKENS_NUM      16

#define        ID_IDX          0
#define        AGEDAYS_IDX     1
#define        GAGEDAYS_IDX    2
#define        SEX_IDX         3
#define        MUACCM_IDX      4
#define        SFTMM_IDX       5
#define        BFED_IDX        6
#define        WEAN_IDX        7
#define        GAGEBRTH_IDX    8
#define        MAGE_IDX        9
#define        MHTCM_IDX       10
#define        MPARITY_IDX     11
#define        FHTCM_IDX       12
#define        WTKG_IDX        13
#define        LENCM_IDX       14
#define        HCIRCM_IDX      15


#define KEY_FIELD_W GAGEDAYS_IDX
#define KEY_FIELD_L GAGEDAYS_IDX
#define KEY_FIELD_H FHTCM_IDX




vector<int> features;
int K;




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




template<int KEY_FIELD>
struct age_val_map {
    typedef map<double, vector<counters> > MAP;
    typedef map<double, vector<counters> >::iterator ITER;

    MAP storage;

    MAP& get() { return storage; }

    void add(const vector<double>& vec) {
        double k = roundf(vec[KEY_FIELD] * 10000) / 10000;;

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

/*        enum {
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
*/

        for (ITER it = storage.begin(); it != storage.end(); ++it) {
            for (int i = 0; i < 16; ++i) {
                counters& cnts = it->second[i];
                if (i == SEX_IDX) {
                    double m = means[SEX_IDX];
                    if (cnts.cnt) {
                        m = cnts.cnt ? cnts.mean / cnts.cnt : 0.;
                    }
                    tmp.push_back(m < 1.5 ? 1 : 2);
                }
                else if (i == BFED_IDX) {
                    double m = means[BFED_IDX];
                    if (cnts.cnt) {
                        m = cnts.cnt ? cnts.mean / cnts.cnt : 0.;
                    }
                    tmp.push_back(m < .5 ? 0 : 1);
                }
                else if (i == WEAN_IDX) {
                    double m = means[WEAN_IDX];
                    if (cnts.cnt) {
                        m = cnts.cnt ? cnts.mean / cnts.cnt : 0.;
                    }
                    tmp.push_back(m < .5 ? 0 : 1);
                }
                else if (i == FHTCM_IDX) {
                    double m = means[FHTCM_IDX];
                    if (cnts.cnt) {
                        m = cnts.cnt ? cnts.mean / cnts.cnt : 0.;
                    }
                    tmp.push_back(m);
                }
                else {
                    if (cnts.cnt) {
                        tmp.push_back(cnts.cnt ? cnts.mean / cnts.cnt : 0.);
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


template<int KEY_FIELD>
map <double, vector<double> > means(age_val_map<KEY_FIELD>& avm) {
    return avm.as_vectors();
}

template<int KEY_FIELD>
double get_val(double k, int idx, age_val_map<KEY_FIELD>& m) {
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

int allocated = 0;
int shared = 0;
int deshared = 0;
int deallocated_arrays = 0;
int deallocated_singles = 0;

void print_stat() {
    LOG("Memory stat")
    LOG("   allocated: " << allocated)
    LOG("   shared: " << shared)
    LOG("   deshared: " << deshared)
    LOG("   deallocated arrays: " << deallocated_arrays)
    LOG("   deallocated singles: " << deallocated_singles)
}

template<class T>
struct DESTRUCTOR {
    static void destroy(T* p) {
        ++deallocated_singles;
        delete p;
    }
};

template<class T>
struct DESTRUCTOR_ARRAY {
    static void destroy(T* p) {
        ++deallocated_arrays;
        delete [] p;
    }
};



template<class T, class D>  // DESTRUCTOR_ARRAY<T>
struct ptr {
    T* p_;
    int* ref_;

    ptr() : p_(NULL), ref_(NULL) {}
    ptr(T* p) : p_(p), ref_(new int(1)) {
        ++allocated;
    }
    ptr(const ptr& other) {
        p_ = other.p_;
        ref_ = other.ref_;
        if (ref_)
            ++(*ref_);

        ++shared;
    }

    ~ptr() {free();}

    ptr& operator=(const ptr& other) {
        ++shared;

        ptr tmp = other;
        swap(tmp);
        return *this;
    }

    T* get() {return p_;}

    T* operator->() {return get();}

    T& operator[] (int i) {
        return get()[i];
    }

    void free() {
        if (ref_) {
            if (0 == --(*ref_)) {
                D::destroy(p_);
                delete ref_;
            }
            else{
                ++deshared;
            }
        }
        else {
            ++deshared;
        }
        p_ = NULL;
        ref_ = NULL;
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

    // op with scalar
    static void mul(double scalar, const double* __restrict__ v, double* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] = v[i] * scalar;
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
        return size ? sum / size : 0.;
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
// RandomForest
//





//
// Fast dtree
//

class dtree_node {
    int COLUMNS;
    int K;
    int LNUM;

public:
    dtree_node(int columns, int k, int lnum) :
        COLUMNS(columns),
        K(k),
        LNUM(lnum)
    {}

private:
    struct data {
        data() :
            count(0),
            idx(-1),
            x_val(-1.),
            y_sum(0.),
            y_sum_squared(0.)
        {}

        int count;
        int idx;
        double x_val;
        double y_sum;
        double y_sum_squared;
    };

    struct split_data {
        split_data(int K) {
            indices = memory::ptr<int, memory::DESTRUCTOR_ARRAY<int> >(new int[K]);
            accums = memory::ptr<std::map<double, data>, memory::DESTRUCTOR_ARRAY<std::map<double, data> > >(new std::map<double, data>[K]);
        }

        memory::ptr<int, memory::DESTRUCTOR_ARRAY<int> > indices;
        memory::ptr<std::map<double, data>, memory::DESTRUCTOR_ARRAY<std::map<double, data> > > accums;
    };


    // temporary data for splitting a node
    memory::ptr<split_data, memory::DESTRUCTOR<split_data> > sd_;
    double total_y_sum;
    double total_y_sum_squared;
    double total_count;

    int node_vector_idx;

public:

    void set_node_vector_idx(int idx) { node_vector_idx = idx; }
    int get_node_vector_idx() const { return node_vector_idx; }

    double get_mean() const { return total_count ? total_y_sum / total_count : 0.; }

    void start_splitting(int* indices) {
        //sd_.free();
        sd_ = memory::ptr<split_data, memory::DESTRUCTOR<split_data> >(new split_data(K));

        for (int i = 0; i < K; ++i)
            sd_->indices[i] = indices[i];

        total_y_sum = 0.;
        total_y_sum_squared = 0.;
        total_count = 0;
    }

    void process_splitting(const double* __restrict__ x, double y) {
        for (int i = 0; i < K; ++i) {
            double val = x[ sd_->indices[i] ];
            //LOG("Idx: " << sd_->indices[i] << "; Val: " << val)

            typename std::map<double, data>::iterator result = sd_->accums[i].find(val);
            if (sd_->accums[i].end() == result) {
                data d;
                d.idx = sd_->indices[i];
                d.x_val = val;
                d.count = 1;
                d.y_sum += y;
                d.y_sum_squared += y * y;
                sd_->accums[i].insert(std::pair<double, data>(val, d));
            }
            else {
                data& d = result->second;
                d.count += 1;
                d.y_sum += y;
                d.y_sum_squared += y * y;
            }
        }

        total_y_sum += y;
        total_y_sum_squared += y * y;
        total_count += 1;
    }

    void stop_splitting(int* idx, double* val, double* gain) {

        const double BEST_GAIN_THRESHOLD = 0.5;

        int best_idx = -1;
        double best_val = -1.;
        double best_gain = 0.;

        double mean = total_count ? total_y_sum / total_count : 0.;
        double mean_squared = mean * mean;

        for (int i = 0; i < K; ++i) {
            // latest element contains sums for total set

            //
            double left_sum_accum = 0.;
            double right_sum_accum = 0.;
            double left_sum_squared_accum = 0.;
            double right_sum_squared_accum = 0.;
            int left_count_accum = 0;
            int right_count_accum = 0;
            for (typename std::map<double, data>::iterator it = sd_->accums[i].begin(); it != sd_->accums[i].end(); ++it) {
                data& d = it->second;

                left_sum_accum += d.y_sum;
                right_sum_accum = (total_y_sum - left_sum_accum);
                left_sum_squared_accum += d.y_sum_squared;
                right_sum_squared_accum = (total_y_sum_squared - left_sum_squared_accum);

                left_count_accum += d.count;
                right_count_accum = ((int)total_count - left_count_accum);

                double left_mean = left_count_accum ? left_sum_accum / left_count_accum : 0.;
                double right_mean = right_count_accum ? right_sum_accum / right_count_accum : 0.;

                double left_mean_squared = left_mean * left_mean;
                double right_mean_squared = right_mean * right_mean;

                double g = (total_y_sum_squared + mean_squared * total_count - 2. * mean * total_y_sum) -
                           (left_sum_squared_accum + left_mean_squared * left_count_accum - 2. * left_mean * left_sum_accum) -
                           (right_sum_squared_accum + right_mean_squared * right_count_accum - 2. * right_mean * right_sum_accum);

                //LOG("Idx: " << d.idx << "; Val: " <<  d.x_val << "; Gain: " << g)

                if (g > BEST_GAIN_THRESHOLD && g > best_gain) {
                //if (g > best_gain) {
                    if (LNUM <= left_count_accum && LNUM <= right_count_accum) {
                        best_idx = d.idx;
                        best_val = d.x_val;
                        best_gain = g;
                    }
                }
            }
        }

        *idx = best_idx;
        *val = best_val;
        *gain = best_gain;

        //sd_.free();
    }

};



class RF {
    int TREES;
    int COLUMN_NUMBER;
    int KF;
    int LN;

public:
    typedef dtree_node NODE;

    RF(int trees, int columns, int kf, int ln) :
        TREES(trees),
        COLUMN_NUMBER(columns),
        KF(kf),
        LN(ln)
    {
        forest = memory::ptr<vector<double>, memory::DESTRUCTOR_ARRAY<vector<double> > >(new vector<double>[TREES]);
        nodes = memory::ptr<vector<NODE>, memory::DESTRUCTOR_ARRAY<vector<NODE> > >(new vector<NODE>[TREES]);
    }

private:
    memory::ptr<vector<double>, memory::DESTRUCTOR_ARRAY<vector<double> > > forest;
    memory::ptr<vector<NODE>, memory::DESTRUCTOR_ARRAY<vector<NODE> > > nodes;

public:

    enum {
        LEAF = 0,
        NON_LEAF = 1,
        DATA_LEN = 0,
        VEC_LEN = 6,

        //ID_IDX = 0,
        CLS_IDX = 1,
        DATA_IDX = 2,
        IDX_IDX = 2,
        VAL_IDX = 3,
        LEFT_IDX = 4,
        RIGHT_IDX = 5,
    };


    void dump(ofstream& fout) {
        int vec_len = VEC_LEN;

        fout.write((char*)&TREES, sizeof(int));         // Num of trees
        fout.write((char*)&vec_len, sizeof(int));       // Vector's length

        for (int t = 0; t < TREES; ++t) {
            vector<double>& tree = forest[t];
            int rows = tree.size() / vec_len;

            fout.write((char*)&rows, sizeof(int));      // Num of vectors in the tree

            for (int r = 0; r < rows; ++r) {
                int idx = r * vec_len;
                for (int c = 0; c < vec_len; ++c) {
                    fout.write((char*)&tree[idx + c], sizeof(double));
                }
            }
        }
    }

    ostream& print(ostream& os) {
        for (int t = 0; t < TREES; ++t) {
            print_tree(os, t);
        }
        return os;
    }

    ostream& print_tree(ostream& os, int tree_idx) {
        os << "// Tree #" << tree_idx << endl;
        os << "double tree_" << tree_idx << "[][" << VEC_LEN << "] = {" << endl;

        vector<double>& tree = forest[tree_idx];
        int rows = tree.size() / VEC_LEN;

        for (int r = 0; r < rows; ++r) {
            os << "  {";
            for (int c = 0; c < VEC_LEN; ++c) {
                int idx = r * VEC_LEN + c;

                os << tree[idx] << ",";
            }
            os << "}," << endl;
        }

        os << "};" << endl;

        return os;
    }



    void reset_forest() {
        for (int t = 0; t < TREES; ++t) {
            forest[t].clear();
            nodes[t].clear();
        }
    }

    bool get_node(int tree_idx, const double* __restrict__ x, NODE** node_result) {
        vector<double>& tree = forest[tree_idx];
        int nodes_num = tree.size() / VEC_LEN;        // currently available nodes

        bool result = true;
        int found_idx = -1;

        if (nodes_num) {
            int cur_node = 0;

            while (true) {
                double* node = &tree[cur_node];

                if (LEAF == node[CLS_IDX]) {
                    result = false;
                    break;  // nothing to return
                }

                if (-1. == node[LEFT_IDX] && 0 <= node[RIGHT_IDX]) {
                    found_idx = (int)node[RIGHT_IDX];
                    break;
                }
                else {
                    int idx = (int)node[IDX_IDX];

                    if (x[idx] <= node[VAL_IDX]) {
                        cur_node = (int)node[LEFT_IDX] * VEC_LEN;
                    }
                    else {
                        cur_node = (int)node[RIGHT_IDX] * VEC_LEN;
                    }
                }
            }
        }
        else {
            int indices[KF];
            random::get_k_of_n(KF, COLUMN_NUMBER, &indices[0]);

            NODE root(COLUMN_NUMBER, KF, LN);
            root.set_node_vector_idx(0);
            root.start_splitting(indices);

            nodes[tree_idx].push_back(root);
            found_idx = nodes[tree_idx].size()- 1;

            tree.push_back(0);
            tree.push_back(NON_LEAF);
            tree.push_back(0.);         // idx
            tree.push_back(0.);         // val
            tree.push_back(-1.);        // left
            tree.push_back(found_idx);  // conv: left = -1 && right = node_id
            for (int i = 0; i < DATA_LEN; ++i)
                tree.push_back(0.);     // padding
        }


        if (-1 != found_idx)
            *node_result = &nodes[tree_idx][found_idx];
        return result;
    }

    void start_fit() {
        random::seed();
    }

    void process_fit(const double* __restrict__ x, double y) {
        for (int t = 0; t < TREES; ++t) {

            NODE* node = NULL;
            if (get_node(t, x, &node)) {
                if (node) {
                    node->process_splitting(x, y);
                }
                else
                    LOG("ERROR: node not found")
            }
        }
    }

    /* Node format:
     *
     * id, class, data..., 0...
     * id, class, idx, val, left, right, 0...
     *
     * Where: class = leaf/non-leaf
     */

    bool stop_fit() {
        bool stop_fitting = true;

        for (int t = 0; t < TREES; ++t) {
            vector<NODE> tmp;

            for (int n = 0; n < nodes[t].size(); ++n) {
                NODE& node = nodes[t][n];

                int idx;
                double val, gain;

                node.stop_splitting(&idx, &val, &gain);

                if (-1 == idx) {
                    // leaf
                    double mean = node.get_mean();
                    vector<double>& tree = forest[t];
                    tree[ node.get_node_vector_idx() * VEC_LEN + CLS_IDX] = LEAF;
                    tree[ node.get_node_vector_idx() * VEC_LEN + DATA_IDX] = mean;
                }
                else {
                    stop_fitting = false;

                    // split
                    int indices[KF];

                    random::get_k_of_n(KF, COLUMN_NUMBER, indices);
                    NODE new_node1(COLUMN_NUMBER, KF, LN);
                    new_node1.start_splitting(indices);

                    random::get_k_of_n(KF, COLUMN_NUMBER, indices);
                    NODE new_node2(COLUMN_NUMBER, KF, LN);
                    new_node2.start_splitting(indices);

                    int idx1 = tmp.size();
                    int idx2 = idx1 + 1;

                    //
                    vector<double>& tree = forest[t];

                    int node_vector_idx1 = tree.size() / VEC_LEN;
                    tree.push_back(node_vector_idx1);
                    tree.push_back(NON_LEAF);
                    tree.push_back(0.);         // idx
                    tree.push_back(0.);         // val
                    tree.push_back(-1.);        // left
                    tree.push_back(idx1);       // conv: left = -1 && right = node_id
                    for (int i = 0; i < DATA_LEN; ++i)
                        tree.push_back(0.);     // padding

                    int node_vector_idx2 = tree.size() / VEC_LEN;
                    tree.push_back(node_vector_idx2);
                    tree.push_back(NON_LEAF);
                    tree.push_back(0.);         // idx
                    tree.push_back(0.);         // val
                    tree.push_back(-1.);        // left
                    tree.push_back(idx2);       // conv: left = -1 && right = node_id
                    for (int i = 0; i < DATA_LEN; ++i)
                        tree.push_back(0.);     // padding


                    //
                    new_node1.set_node_vector_idx(node_vector_idx1);
                    new_node2.set_node_vector_idx(node_vector_idx2);

                    tmp.push_back(new_node1);
                    tmp.push_back(new_node2);

                    //
                    tree[ node.get_node_vector_idx() * VEC_LEN + IDX_IDX] = idx;
                    tree[ node.get_node_vector_idx() * VEC_LEN + VAL_IDX] = val;
                    tree[ node.get_node_vector_idx() * VEC_LEN + LEFT_IDX ] = node_vector_idx1;
                    tree[ node.get_node_vector_idx() * VEC_LEN + RIGHT_IDX ] = node_vector_idx2;
                }
            }
            tmp.swap(nodes[t]);
        }

        return stop_fitting;
    }


    double predict(const double* __restrict__ x) {
        double p = 0.;

        for (int t = 0; t < TREES; ++t) {
            vector<double>& tree = forest[t];

            int cur_node = 0;

            while (true) {
                double* node = &tree[cur_node];

                if (LEAF == node[CLS_IDX]) {
                    p += node[DATA_IDX];
                    break;  // nothing to return
                }
                else {
                    int idx = (int)node[IDX_IDX];


                    if (x[idx] <= node[VAL_IDX]) {
                        cur_node = (int)node[LEFT_IDX] * VEC_LEN;
                    }
                    else {
                        cur_node = (int)node[RIGHT_IDX] * VEC_LEN;
                    }
                }
            }   // while (1)
        }   // for TREES

        return p / TREES;
    }



};



/////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////

class ChildStuntedness2 {
/*
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

        //X_COLUMNS      = 12,

    };
*/
    int X_COLUMNS_W;
    int X_COLUMNS_L;
    int X_COLUMNS_H;

    memory::ptr<RF, memory::DESTRUCTOR<RF> > rf_w;
    memory::ptr<RF, memory::DESTRUCTOR<RF> > rf_l;
    memory::ptr<RF, memory::DESTRUCTOR<RF> > rf_h;

    vector<double> X_W;
    vector<double> X_L;
    vector<double> X_H;

    vector<double> Y1;
    vector<double> Y2;
    vector<double> Y3;

    utils::age_val_map<KEY_FIELD_W> avm_w;
    utils::age_val_map<KEY_FIELD_L> avm_l;
    utils::age_val_map<KEY_FIELD_H> avm_h;


public:
    ChildStuntedness2() :
        X_COLUMNS_W(12),
        X_COLUMNS_L(12),
        X_COLUMNS_H(12)
    {
        X_W.reserve(X_COLUMNS_W*1024*sizeof(double));
        X_L.reserve(X_COLUMNS_L*1024*sizeof(double));
        X_H.reserve(X_COLUMNS_H*1024*sizeof(double));

        Y1.reserve(1024*sizeof(double));
        Y2.reserve(1024*sizeof(double));
        Y3.reserve(1024*sizeof(double));

        int tmp[] = {0,1,2,3,4,5,6,7,8,9,10,11};
        //int tmp[] = {1,3,5,7,9,11,0,1,3,7};
#if defined SUBMISSION
        features.clear();
        features.assign(&tmp[0], &tmp[sizeof(tmp) / sizeof(int)]);
        K = sqrt(features.size());
        //K = 5;
#endif

        LOG("Features: ")
        stringstream ss;
        for (int i = 0; i < features.size(); ++i)
            ss << features[i] << ",";
        LOG(ss.str())

    }
    ~ChildStuntedness2() {}

    void add_to_data(/*const*/ vector<double>& vec) {

        // W
        {
            vector<double> tmp(vec.begin(), vec.end());

            X_COLUMNS_W = features.size();
            for (int i = 0; i < X_COLUMNS_W; ++i) {
                int idx = features[i] + 1;
                if (NOVAL == vec[idx]) {
                    vec[idx] = utils::get_val(vec[KEY_FIELD_W], idx, avm_w);
                }

                //X.push_back( vec[MAGE_IDX] ? vec[idx] / vec[MAGE_IDX] : 0.);
                //X.push_back( vec[idx] * vec[idx] );
                X_W.push_back( exp(vec[idx] / 6.));
            }
            X_COLUMNS_W *= 1;
        }

        // L
        {
            vector<double> tmp(vec.begin(), vec.end());

            X_COLUMNS_L = features.size();
            for (int i = 0; i < X_COLUMNS_L; ++i) {
                int idx = features[i] + 1;
                if (NOVAL == vec[idx]) {
                    vec[idx] = utils::get_val(vec[KEY_FIELD_L], idx, avm_l);
                }

                //X.push_back( vec[MAGE_IDX] ? vec[idx] / vec[MAGE_IDX] : 0.);
                //X.push_back( vec[idx] * vec[idx] );
                X_L.push_back( exp(vec[idx] / 6.));
            }
            X_COLUMNS_L *= 1;
        }

        // H
        {
            vector<double> tmp(vec.begin(), vec.end());

            X_COLUMNS_H = features.size();
            for (int i = 0; i < X_COLUMNS_H; ++i) {
                int idx = features[i] + 1;
                if (NOVAL == vec[idx]) {
                    vec[idx] = utils::get_val(vec[KEY_FIELD_H], idx, avm_h);
                }

                //X.push_back( vec[MAGE_IDX] ? vec[idx] / vec[MAGE_IDX] : 0.);
                //X.push_back( vec[idx] * vec[idx] );
                X_H.push_back( exp(vec[idx] / 6.));
            }
            X_COLUMNS_H *= 1;
        }

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
            //avm_w.add(tmp);
            //avm_l.add(tmp);
            //avm_h.add(tmp);
        }
        LOG("means are done")
        for (int i = 0; i < size; ++i) {
            line2vec(train_str[i], tmp);
            add_to_data(tmp);
        }
        tmp.clear();

        int rows = Y1.size();

        preproc();

        LOG("parsing done")

        // fitting
        int TREES = 50;
        int LNUM  = 7;
        rf_w = memory::ptr<RF, memory::DESTRUCTOR<RF> >(new RF(TREES, X_COLUMNS_W, K, LNUM));
        rf_l = memory::ptr<RF, memory::DESTRUCTOR<RF> >(new RF(TREES, X_COLUMNS_L, K, LNUM));
        rf_h = memory::ptr<RF, memory::DESTRUCTOR<RF> >(new RF(TREES, X_COLUMNS_H, K, LNUM));

        rf_w->start_fit();
        rf_l->start_fit();
        rf_h->start_fit();

        bool stop_w = false;
        bool stop_l = false;
        bool stop_h = false;

        while (!stop_w || !stop_l || !stop_h) {
            for (int r = 0; r < rows; ++r) {
                if (!stop_w) {
                    int idx = r * X_COLUMNS_W;
                    rf_w->process_fit(&X_W[idx], Y1[r]);
                }
                if (!stop_l) {
                    int idx = r * X_COLUMNS_L;
                    rf_l->process_fit(&X_L[idx], Y2[r]);
                }
                if (!stop_h) {
                    int idx = r * X_COLUMNS_H;
                    rf_h->process_fit(&X_H[idx], Y3[r]);
                }
            }

            stop_w = rf_w->stop_fit();
            stop_l = rf_l->stop_fit();
            stop_h = rf_h->stop_fit();

        }

        //
        X_W.clear();
        X_L.clear();
        X_H.clear();
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

            ss << rf_w->predict( &X_W[0] );
            ss << "," << rf_l->predict( &X_L[0] );
            ss << "," << rf_h->predict( &X_H[0] );

            result.push_back(ss.str());
            ss.str("");

            X_W.clear();
            X_L.clear();
            X_H.clear();
        }

        LOG("done")

        return result;
    }


private:
    ChildStuntedness2(const ChildStuntedness2&);
    ChildStuntedness2& operator= (const ChildStuntedness2&);


    void preproc() {
        return;
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


template<class T>
void line2vec(const string& line, vector<T>& vec) {
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


void process_result(const vector<string>& preds, const vector<string>& y) {
    int size = y.size();

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
    sse_l = sqrt(sse_l);
    sse_h = sqrt(sse_h);

    LOG("SSE W: " << sse_w)
    LOG("SSE L: " << sse_l)
    LOG("SSE H: " << sse_h)
}


int main(int argc, const char** argv) {

    if (2 == argc) {
        string features_str = argv[1];
        line2vec(features_str, features);

        LOG("features " << features_str)

    }
    else {
        for (int  i = 0; i < 12; ++i) {
            features.push_back(i);
        }
    }
    K = sqrt(features.size());

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

    clock_t start = clock();

    ChildStuntedness2 o;
    LOG("init done")
    vector<string> result = o.predict(train_str, test_str);

    float finish = (float)clock() - start;

//    for (int i = 0; i < test_str.size(); ++i) {
//        size_t pos = test_str[i].find(',');
//        LOG("[" << test_str[i].substr(0,pos) << "] " << result[i])
//    }

    process_result(result, test_str);

    LOG("Time: " << (finish / CLOCKS_PER_SEC))
    LOG("done")

    memory::print_stat();

    return 0;
}


