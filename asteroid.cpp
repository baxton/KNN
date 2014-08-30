

// To test utils:
// g++ -DTEST_UTILS --std=c++0x -W -Wall -Wno-sign-compare -O2 -s -pipe -mmmx -msse -msse2 -msse3 random_forest.cpp -o random_forest.exe
//
// To test asteroid detector
// g++ -DTEST_DETECTOR --std=c++0x -W -Wall -Wno-sign-compare -O2 -s -pipe -mmmx -msse -msse2 -msse3 random_forest.cpp -o random_forest.exe
//
// Fast test
// g++ -DTEST_FAST --std=c++0x -W -Wall -Wno-sign-compare -O2 -s -pipe -mmmx -msse -msse2 -msse3 random_forest.cpp -o random_forest.exe
//


#include <cstdlib>
#include <time.h>
#include <cmath>
#include <vector>
#include <string>

using namespace std;

#define FOR_SUBMISSION


#define LOG


#if defined TEST_DETECTOR
#   include <sstream>
#   include <iostream>
#   include <fstream>

#   undef FOR_SUBMISSION

struct app {
    ofstream log_;

    app() {
        log_.open("C:\\Temp\\asteroid\\logs\\detector.txt", ofstream::out);
    }
    ~app() {
        log_.close();
    }

    ostream& log() { return log_; }
} app;

#   undef LOG
#   define LOG app.log()
#endif // TEST_DETECTOR


#if defined TEST_FAST
#   include <sstream>
#   include <iostream>
#   include <fstream>

#   undef FOR_SUBMISSION

struct app {
    ofstream log_;

    app() {
        log_.open("C:\\Temp\\asteroid\\logs\\detector.txt", ofstream::out);
    }
    ~app() {
        log_.close();
    }

    ostream& log() { return log_; }
} app;

#   undef LOG
#   define LOG app.log()
#endif // TEST_FAST


#if defined TEST_UTILS
#   include <iostream>

#   undef FOR_SUBMISSION

#   undef LOG
#   define LOG cout
#endif  // TEST


//
// Utils
//

//double abs(double d) {return d > 0. ? d : -d;}

bool equal(double v1, double v2) {
    if (::abs(v1 - v2) < 0.0001)
        return true;
    return false;
}

void detection_parser(const std::string& detection,
                      int& id,
                      int& img_id,
                      double& ra,
                      double& dec,
                      double& x,
                      double& y,
                      double& mag,
                      int& neo) {
}

//
// Memory management
//
template<class T>
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
    }

    T* get() {return p_;}

    void free() {
        if (ref_) {
            if (0 == --(*ref_)) {
                delete [] p_;
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

//
// random numbers
//
struct random {
    void seed(int s=-1) {
        if (s == -1)
            srand(time(NULL));
        else
            srand(s);
    }

    int randint() {
        return rand();
    }

    int randint(int low, int high) {
        int r = rand();
        r = r % (high - low) + low;
        return r;
    }

    void randint(int low, int high, int* numbers, int size) {
        for (int i = 0; i < size; ++i)
            numbers[i] = rand() % (high - low) + low;
    }

    /*
     * Retuns:
     * k indices out of [0-n) range
     * with no repetitions
     */
    void get_k_of_n(int k, int n, int* numbers) {
        for (int i = 0; i < k; ++i) {
            numbers[i] = i;
        }

        for (int i = k; i < n; ++i) {
            int r = randint(0, i);
            if (r < k) {
                numbers[r] = i;
            }
        }
    }
};


//
// Lin algebra
//
struct linalg {
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

    static void pow(double scalar, double* __restrict__ v, int size) {
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

        ptr<double> tmp( linalg::zeros(rows) );
        linalg::dot_m_to_v(X, theta, tmp.get(), rows, columns);

        ptr<double> deltas( linalg::zeros(rows) );
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

        ptr<double> h = linalg::zeros(rows);
        logistic_h(theta, X, h.get(), rows, columns);

        double E = 0.;
        for (int i = 0; i < rows; ++i) {
            E += (-Y[i]) * ::log(h.get()[i]) - (1. - Y[i]) * ::log(1. - h.get()[i]);
        }
        E /= M;

        *cost = E;

        ptr<double> deltas = linalg::zeros(rows);
        linalg::sub(h.get(), Y, deltas.get(), rows);
        linalg::dot_v_to_m(deltas.get(), X, grad, rows, columns);
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
                       double feature_val)
    {
        double M = rows;     // number of elements in the original Y array
        double left_M = 0.;  // number of elements in the left part
        double right_M = 0.; // number of elements in the right part

        double mean = 0.;
        double left_mean = 0.;
        double right_mean = 0.;

        double sum = 0.;
        double left_sum = 0.;
        double right_sum = 0.;

        // calc means and number of elements
        for (int i = 0; i < rows; ++i) {
            int idx = i * columns + feature_idx;

            if (X[idx] <= feature_val) {
                left_M += 1.;
                left_mean += Y[i];
            }
            else {
                right_M += 1.;
                right_mean += Y[i];
            }

            mean += Y[i];
        }

        if (M == 0. || left_M == 0. || right_M == 0.) {
            return -1.;
        }

        mean /= M;
        left_mean /= left_M;
        right_mean /= right_M;

        // calculate sums of squared errors and MSE gain of information
        for (int i = 0; i < rows; ++i) {
            int idx = i * columns + feature_idx;

            if (X[idx] <= feature_val) {
                left_sum += (Y[i] - left_mean) * (Y[i] - left_mean);
            }
            else {
                right_sum += (Y[i] - right_mean) * (Y[i] - right_mean);
            }

            sum += (Y[i] - mean) * (Y[i] - mean);
        }

        //double err = sum / M - (left_sum / left_M + right_sum / right_M);
        double err = sum - (left_sum + right_sum);
        return err;
    }
};

/*
 * Decision tree class
 */
template<class ESTIMATOR=estimator_regressor>
class dtree {
    int k_;
    int lnum_;
    dtree* left_;
    dtree* right_;

public:
    dtree(int k, int lnum) :
        k_(k),
        lnum_(lnum),
        left_(NULL),
        right_(NULL)
    {}

    void fit(const double* __restrict__ X,
             const double* __restrict__ Y,
             int rows,
             int columns) {
        //
        ;
    }

private:
    /*
     * Return: information gain estimation
     */
    double get_best_split(const double* __restrict__ X,
                          const double* __restrict__ Y,
                          int rows,
                          int columns,
                          int feature_idx,
                          /*OUT*/ double* feature_val) {
        ESTIMATOR estimator;
        double max_x = 0.;
        double max_gain = -1;

        for (int i = 0; i < rows; ++i) {
            int idx = i * columns + feature_idx;
            double x = X[idx];

            double gain = estimator(X, Y, rows, columns, feature_idx, x);
            if (gain != -1. && (max_gain == -1 || max_gain < gain)) {
                max_x = x;
                max_gain = gain;
            }
        }
        *feature_val = max_x;
        return max_gain;
    }
};

//
// RandomForest
//

class RF {

public:
    RF() {}
};


///////////////////////////////////////////////////////////////////////////////
//
// Star Object
//
///////////////////////////////////////////////////////////////////////////////

// I investigate a square 10x10
#define AREA_WIDTH 10

struct star_object {
    enum {
        RA_IDX = 0,
        DEC_IDX,
        RA_ADJ_IDX,
        DEC_ADJ_IDX,
        X_IDX,
        Y_IDX,
        X_ADJ_IDX,
        Y_ADJ_IDX,
        MAG_IDX,
    };

    // image data
    double img[AREA_WIDTH * AREA_WIDTH];

    // RA, DEC, RA_adjusted, DEC_adjusted, x, y, x_adjusted, y_adjusted, magnitude
    double attributes[8];
};

///////////////////////////////////////////////////////////////////////////////
//
// Asteroid Detector
//
///////////////////////////////////////////////////////////////////////////////

class AsteroidDetector {





public:

    int trainingData(int width,
                     int height,
                     vector<int>& imageData_1,
                     vector<string>& header_1,
                     vector<double>& wcs_1,
                     vector<int>& imageData_2,
                     vector<string>& header_2,
                     vector<double>& wcs_2,
                     vector<int>& imageData_3,
                     vector<string>& header_3,
                     vector<double>& wcs_3,
                     vector<int>& imageData_4,
                     vector<string>& header_4,
                     vector<double>& wcs_4,
                     vector<string>& detections) {
        //
        return 0;
    }

    int testingData(string imageID,
                    int width,
                    int height,
                    vector<int>& imageData_1,
                    vector<string>& header_1,
                    vector<double>& wcs_1,
                    vector<int>& imageData_2,
                    vector<string>& header_2,
                    vector<double>& wcs_2,
                    vector<int>& imageData_3,
                    vector<string>& header_3,
                    vector<double>& wcs_3,
                    vector<int>& imageData_4,
                    vector<string>& header_4,
                    vector<double>& wcs_4) {
        //
        return 0;
    }

    vector<string> getAnswer() {
        vector<string> result;

        return result;
    }
};


///////////////////////////////////////////////////////////////////////////////
// NOTE: I do not need everything below this line for submission
///////////////////////////////////////////////////////////////////////////////

//
// Mains
//

#if defined TEST_DETECTOR

void read_data();

int main(int argc, char* argv[]) {

    read_data();

    return 0;
}

template<class T>
void vectors_reserve(T v[4], int size) {
    v[0].reserve(size);
    v[1].reserve(size);
    v[2].reserve(size);
    v[3].reserve(size);
}

template<class T>
void vectors_clear(T v[4]) {
    v[0].clear();
    v[1].clear();
    v[2].clear();
    v[3].clear();
}

void read_img(int size, vector<int>& data, vector<string>& headers, vector<double>& wcs) {
    string line;
    int v_int;
    double v_double;

    for (int i = 0; i < size; ++i) {
        std::getline(cin, line);
        v_int = std::atoi(line.c_str());
        data.push_back(v_int);
    }

    std::getline(cin, line);
    v_int = std::atoi(line.c_str());   // N headers
    for (int i = 0; i < v_int; ++i) {
        std::getline(cin, line);
        headers.push_back(line);
    }

    for (int i = 0; i < 8; ++i) {
        std::getline(cin, line);
        v_double = atof(line.c_str());
        wcs.push_back(v_double);
    }
}

void save_train_set(int set_id, int width, int height, vector<int> imageData[4], vector<string> header[4], vector<double> wcs[4], vector<string> detections) {
    std::stringstream ss;
    ss << "C:\\Temp\\asteroid\\data2\\train_" << set_id << ".txt";
    ofstream ofs;
    ofs.open(ss.str(), std::ofstream::out);

    ofs << width << endl << height << endl;

    for (int idx = 0; idx < 4; ++idx)
    {
        int size = imageData[idx].size();
        for (int i = 0; i < size; ++i) {
            ofs << imageData[idx][i] << endl;
        }
        size = header[idx].size();
        ofs << size << endl;
        for (int i = 0; i < size; ++i) {
            ofs << header[idx][i] << endl;
        }
        for (int i = 0; i < 8; ++i) {
            ofs << wcs[idx][i] << endl;
        }
    }

    int size = detections.size();
    ofs << size << endl;
    for (int i = 0; i < size; ++i) {
        ofs << detections[i];
        if (i < size - 1)
            ofs << endl;
    }

    ofs.flush();
    ofs.close();
}

void save_test_set(int set_id, int width, int height, vector<int> imageData[4], vector<string> header[4], vector<double> wcs[4]) {
    std::stringstream ss;
    ss << "C:\\Temp\\asteroid\\data2\\test_" << set_id << ".txt";
    ofstream ofs;
    ofs.open(ss.str(), std::ofstream::out);

    ofs << width << endl << height << endl;

    for (int idx = 0; idx < 4; ++idx)
    {
        int size = imageData[idx].size();
        for (int i = 0; i < size; ++i) {
            ofs << imageData[idx][i] << endl;
        }
        size = header[idx].size();
        ofs << size << endl;
        for (int i = 0; i < size; ++i) {
            ofs << header[idx][i] << endl;
        }
        for (int i = 0; i < 8; ++i) {
            ofs << wcs[idx][i];
            if (idx == 3 && i < 7)
                ofs << endl;
        }
    }
    ofs.flush();
    ofs.close();
}


void read_data() {
    AsteroidDetector detector;

    vector<int> imageData[4];
    vector<string> header[4];
    vector<double> wcs[4];
    vector<string> detections;

    string line;

    // train
    int cur_train_set = 0;
    int max_train_set = 5;

    for (int i = 0; i < 100; ++i) {
        vectors_clear(imageData);
        vectors_clear(header);
        vectors_clear(wcs);
        detections.clear();

        // width & hight
        std::getline(cin, line);
        int width = std::atoi(line.c_str());
        std::getline(cin, line);
        int height = std::atoi(line.c_str());
        int size = width * height;

        vectors_reserve(imageData, size);
        read_img(size, imageData[0], header[0], wcs[0]);
        read_img(size, imageData[1], header[1], wcs[1]);
        read_img(size, imageData[2], header[2], wcs[2]);
        read_img(size, imageData[3], header[3], wcs[3]);

        // detections
        std::getline(cin, line);
        int v_int = std::atoi(line.c_str());
        for (int n = 0; n < v_int; ++n) {
            std::getline(cin, line);
            detections.push_back(line);
        }

        int res = detector.trainingData(width, height, imageData[0], header[0], wcs[0], imageData[1], header[1], wcs[1], imageData[2], header[2], wcs[2], imageData[3], header[3], wcs[3], detections);

        // for debugging
        ++cur_train_set;
        if (cur_train_set >= max_train_set) {
            res = 1;
        }
        save_train_set(cur_train_set, width, height, imageData, header, wcs, detections);
        //

        cout << res << endl;

        if (res == 1)
            break;
    }   // end of train


    // test
    int cur_test_set = 0;

    for (int i = 0; i < 20; ++i) {
        vectors_clear(imageData);
        vectors_clear(header);
        vectors_clear(wcs);

        std::getline(cin, line);
        string imageID = line;

        // width & hight
        std::getline(cin, line);
        int width = std::atoi(line.c_str());
        std::getline(cin, line);
        int height = std::atoi(line.c_str());
        int size = width * height;

        vectors_reserve(imageData, size);
        read_img(size, imageData[0], header[0], wcs[0]);
        read_img(size, imageData[1], header[1], wcs[1]);
        read_img(size, imageData[2], header[2], wcs[2]);
        read_img(size, imageData[3], header[3], wcs[3]);

        int res = detector.testingData(imageID, width, height, imageData[0], header[0], wcs[0], imageData[1], header[1], wcs[1], imageData[2], header[2], wcs[2], imageData[3], header[3], wcs[3]);
        cout << res << endl;

        ++cur_test_set;
        save_test_set(cur_test_set, width, height, imageData, header, wcs);
    }

    // result

    vector<string> result = detector.getAnswer();

    // output result
    int size = result.size();
    cout << size << endl;
    for (int i = 0; i < size; ++i) {
        cout << result[i] << endl;
    }
}


#endif // TEST_DETECTOR





#if defined TEST_UTILS

void test_linalg();
void test_optimize();

int main(int argc, char* argv[]) {

    //LOG("Testing utils:")
    test_linalg();
    test_optimize();

    return 0;
}

void test_linalg() {
    int total_tests = 0;
    int success_tests = 0;
    // 1
    {
        ptr<double> v1 = linalg::zeros(3);
        ptr<double> v2 = linalg::zeros(3);
        v1.get()[0] = 1.;
        v1.get()[1] = 2.;
        v1.get()[2] = 3.;
        v2.get()[0] = 2.;
        v2.get()[1] = 3.;
        v2.get()[2] = 4.;
        double r = linalg::dot(v1.get(), v2.get(), 3);

        // 1*2 + 2*3 + 3*4 = 20
        if (r != 20.)
            ;//LOG2("ERROR: test_linalg #1, dot = [", r, "], expected 20\n")
        else
            ++success_tests;
        ++total_tests;
    }

    // 2
    {
        double m[] = {1, 2, 3, 1, 2, 3, 1, 2, 3};   // 3*3 matrix
        ptr<double> v = linalg::zeros(3);
        ptr<double> r = linalg::zeros(3);
        v.get()[0] = 2.;
        v.get()[1] = 3.;
        v.get()[2] = 4.;
        linalg::dot_m_to_v(m, v.get(), r.get(), 3, 3);

        // [20, 20, 20]
        if (r.get()[0] != 20. || r.get()[1] != 20. || r.get()[2] != 20.)
            ;//LOG3("ERROR: test_linalg #2, dot_m_to_v [%f, %f, %f], expected [20,20,20]", r.get()[0], r.get()[1], r.get()[2])
        else
            ++success_tests;
        ++total_tests;

        linalg::dot_v_to_m(v.get(), m, r.get(), 3, 3);
        // [9, 18, 27]
        if (r.get()[0] != 9. || r.get()[1] != 18. || r.get()[2] != 27.)
            ;//LOG3("ERROR: test_linalg #2, dot_v_to_m [%f, %f, %f], expected [9,18,27]", r.get()[0], r.get()[1], r.get()[2])
        else
            ++success_tests;
        ++total_tests;
    }

    // 3
    {
        double m[] = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};   // 4*3 matrix
        ptr<double> v = linalg::zeros(4);
        ptr<double> r = linalg::zeros(3);
        v.get()[0] = 2.;
        v.get()[1] = 3.;
        v.get()[2] = 4.;
        v.get()[3] = 5.;

        linalg::dot_v_to_m(v.get(), m, r.get(), 4, 3);
        // [14, 28, 42]
        if (r.get()[0] != 14. || r.get()[1] != 28. || r.get()[2] != 42.)
            ;//LOG3("ERROR: test_linalg #3, dot_v_to_m [%f, %f, %f], expected [14,28,42]", r.get()[0], r.get()[1], r.get()[2])
        else
            ++success_tests;
        ++total_tests;
    }

    // 4
    {
        ptr<double> v = linalg::zeros(4);
        v.get()[0] = 2.;
        v.get()[1] = 3.;
        v.get()[2] = 4.;
        v.get()[3] = 5.;

        linalg::pow(2., v.get(), 4);
        if (v.get()[0] != 4. || v.get()[1] != 9. || v.get()[2] != 16. || v.get()[3] != 25.)
            ;//LOG4("ERROR: test_linalg #4, pow [%f, %f, %f, %f], expected [4,9,16,25]", v.get()[0], v.get()[1], v.get()[2], v.get()[3])
        else
            ++success_tests;
        ++total_tests;
    }

    // 5
    {
        ptr<double> v = linalg::zeros(4);
        ptr<double> r = linalg::zeros(4);
        v.get()[0] = 2.;
        v.get()[1] = 3.;
        v.get()[2] = 4.;
        v.get()[3] = 5.;

        linalg::pow(2., v.get(), r.get(), 4);
        if (r.get()[0] != 4. || r.get()[1] != 9. || r.get()[2] != 16. || r.get()[3] != 25.)
            ;//LOG4("ERROR: test_linalg #5, pow [%f, %f, %f, %f], expected [4,9,16,25]", r.get()[0], r.get()[1], r.get()[2], r.get()[3])
        else
            ++success_tests;
        ++total_tests;
    }

    // 6
    {
        double v[] = {1,2,3,4,5};
        double y[] = {5,4,3,2,1};
        ptr<double> d = linalg::zeros(5);
        linalg::sub(v, y, d.get(), 5);

        if (d.get()[0] != -4. || d.get()[1] != -2. || d.get()[2] != 0. || d.get()[3] != 2. || d.get()[4] != 4.) {
            ;//LOG5("ERROR: test_linalg #6, sub [%f, %f, %f, %f, %f], expected [-4,-2,0,2,4]", d.get()[0], d.get()[1], d.get()[2], d.get()[3], d.get()[4])
        }
        else
            ++success_tests;
        ++total_tests;
    }

    // 7
    {
        double v[] = {1,2,3,4,5};
        double y[] = {5,4,3,2,1};
        ptr<double> d = linalg::zeros(5);
        ptr<double> tmp = linalg::zeros(5);
        linalg::sub(v, y, d.get(), 5);
        linalg::pow(2., d.get(), tmp.get(), 5);

        if (tmp.get()[0] != 16. || tmp.get()[1] != 4. || tmp.get()[2] != 0. || tmp.get()[3] != 4. || tmp.get()[4] != 16.) {
            ;//LOG5("ERROR: test_linalg #7, pow [%f, %f, %f, %f, %f], expected [16,4,0,4,16]\n", tmp.get()[0], tmp.get()[1], tmp.get()[2], tmp.get()[3], tmp.get()[4])
        }
        else
            ++success_tests;
        ++total_tests;
    }

    // 8
    {
        double v[] = {2, 4, 5, 6};
        linalg::div(2., v, v, 4);

        if (v[0] != 1. || v[1] != 2. || v[2] != 2.5 || v[3] != 3.) {
            ;//LOG4("ERROR: test_linalg #8, div [%f, %f, %f, %f], expected [1,2,2.5,3]\n", v[0], v[1], v[2], v[3])
        }
        else
            ++success_tests;
        ++total_tests;
    }

    // 9
    {
        double v[] = {2, 4, 5, 6};
        double s = linalg::sum(v, 4);

        if (s != 17.) {
            ;//LOG1("ERROR: test_linalg #9, sum [%f], expected [17]\n", s)
        }
        else
            ++success_tests;
        ++total_tests;
    }

    //LOG3("Finished: test_linalg, ", success_tests, " of ", total_tests)
}


void test_optimize() {
    int total_tests = 0;
    int success_tests = 0;

    // theta, X, Y, cost, grad, rows, columns

    // 1
    {
        double X[] = {1, 2, 2, 3, 3, 4, 5, 6};
        double Y[] = {5, 8, 11, 17};
        double E = 25.5;
        double grad[] = {22.25, 28.75};
        double theta[] = {2, 3};

        double tmp_E;
        ptr<double> tmp_grad = linalg::zeros(2);

        optimize::quadratic_cost(theta, X, Y, &tmp_E, tmp_grad.get(), 4, 2);

        if (tmp_E != E ||
            tmp_grad.get()[0] != grad[0] || tmp_grad.get()[1] != grad[1]) {
            ;//LOG3("ERROR: test_optimize #1, E [%f] grad [%f, %f], expected 25.5, [22,28]", tmp_E, tmp_grad.get()[0], tmp_grad.get()[1])
        }
        else {
            ++success_tests;
        }

        ++total_tests;
    }

    // 2
    {
        // theta, X, rows, columns, Y, func, max_iterations
        double X[] = {1, 2, 1, 3, 1, 4, 1, 6};
        double Y[] = {6, 8, 10, 14};
        double theta[] = {2, 3};

        // 1.9421305   2.01346032

        optimize::minimize_gc(theta, X, 4, 2, Y, optimize::quadratic_cost, 100);

        if (!equal(theta[0], 1.942131) || !equal(theta[1], 2.013460)) {
            ;//LOG2("ERROR: test_optimize #2, theta [%f, %f], expected [1.942131, 2.013460]\n", theta[0], theta[1])
        }
        else {
            ++success_tests;
        }

        ++total_tests;
    }

    // 3
    {
        int rows = 50;
        int columns = 2;
        double X[] = { 1.0 ,  0.788572620526 ,  1.0 ,  0.670062200437 ,   1.0 ,  0.22974371904 ,
                       1.0 ,  0.27750360394 ,   1.0 ,  0.964301310895 ,   1.0 ,  0.521839466661 ,
                       1.0 ,  0.562981501482 ,  1.0 ,  0.742224344835 ,   1.0 ,  0.0267960623868 ,
                       1.0 ,  0.709410160026 ,  1.0 ,  0.474850653355 ,   1.0 ,  0.107401833692 ,
                       1.0 ,  0.9592266594 ,    1.0 ,  0.57853034708 ,    1.0 , 0.106855436018 ,
                       1.0 ,  0.948170938477 ,  1.0 ,  0.0382418963743 ,  1.0 ,  0.595708312752 ,
                       1.0 ,  0.48211740177 ,   1.0 ,  0.990291370923 ,   1.0 ,  0.0710195982107 ,
                       1.0 ,  0.252565382027 ,  1.0 ,  0.33819629206 ,    1.0 ,  0.171346989635 ,
                       1.0 ,  0.806395278587 ,  1.0 ,  0.43690043386 ,    1.0 ,  0.100040312716 ,
                       1.0 ,  0.40566336606 ,   1.0 ,  0.581286299126 ,   1.0 ,  0.387883846167 ,
                       1.0 ,  0.89174123726 ,   1.0 ,  0.412533167397 ,   1.0 ,  0.654967086259 ,
                       1.0 ,  0.557370668731 ,  1.0 ,  0.39619481655 ,    1.0 ,  0.945446811912 ,
                       1.0 ,  0.636496325157 ,  1.0 ,  0.49096120676 ,    1.0 ,  0.0143092755874 ,
                       1.0 ,  0.950819347394 ,  1.0 ,  0.0743227488296 ,  1.0 ,  0.498676352345 ,
                       1.0 ,  0.63929124517 ,   1.0 ,  0.387228425033 ,   1.0 ,  0.550470131576 ,
                       1.0 ,  0.876536806187 ,  1.0 ,  0.510434958354 ,   1.0 ,  0.981577982404 ,
                       1.0 ,  0.463691740528 ,  1.0 ,  0.5504326445  };
        double Y[] = { 1.0 ,  1.0 ,  0.0 ,  0.0 ,  1.0 ,  1.0 ,  1.0 ,  1.0 ,  0.0 ,  1.0 ,  0.0 ,  0.0 ,  1.0 ,  1.0 ,  0.0 ,  1.0 ,  0.0 ,  1.0 ,  0.0 ,  1.0 ,  0.0 ,  0.0 ,  0.0 ,  0.0 ,  1.0 ,  0.0 ,  0.0 ,  0.0 ,  1.0 ,  0.0 ,  1.0 ,  0.0 ,  1.0 ,  1.0 ,  0.0 ,  1.0 ,  1.0 ,  0.0 ,  0.0 ,  1.0 ,  0.0 ,  0.0 ,  1.0 ,  0.0 ,  1.0 ,  1.0 ,  1.0 ,  1.0 ,  0.0 ,  1.0  };
        double theta[] = {0.12086746, 0.27889032};
        double bm_theta[] = {-6.00811867, 12.03530739};

        optimize::minimize_gc(theta, X, rows, columns, Y, optimize::logistic_cost, 100);

        if (!equal(theta[0], bm_theta[0]) || !equal(theta[1], bm_theta[1])) {
            ;//LOG4("ERROR: test_optimize #2, theta [%f, %f], expected [%f, %f]\n", theta[0], theta[1], bm_theta[0], bm_theta[1])
        }
        else {
            ++success_tests;
        }

        ++total_tests;
    }

    //LOG2("Finished: test_optimize, %d of %d\n", success_tests, total_tests)
}

#endif  // TEST_UTILS





#if defined TEST_FAST

void read_data();

int main(int argc, char* argv[]) {

    read_data();

    return 0;
}

template<class T>
void vectors_reserve(T v[4], int size) {
    v[0].reserve(size);
    v[1].reserve(size);
    v[2].reserve(size);
    v[3].reserve(size);
}

template<class T>
void vectors_clear(T v[4]) {
    v[0].clear();
    v[1].clear();
    v[2].clear();
    v[3].clear();
}

void read_img(int size, vector<int>& data, vector<string>& headers, vector<double>& wcs) {
    string line;
    int v_int;
    double v_double;

    for (int i = 0; i < size; ++i) {
        std::getline(cin, line);
        v_int = std::atoi(line.c_str());
        data.push_back(v_int);
    }

    std::getline(cin, line);
    v_int = std::atoi(line.c_str());   // N headers
    for (int i = 0; i < v_int; ++i) {
        std::getline(cin, line);
        headers.push_back(line);
        //LOG << line << endl;
    }
    //LOG << "======" << endl;

    for (int i = 0; i < 8; ++i) {
        std::getline(cin, line);
        v_double = atof(line.c_str());
        wcs.push_back(v_double);
    }
}

void read_data() {
    AsteroidDetector detector;

    vector<int> imageData[4];
    vector<string> header[4];
    vector<double> wcs[4];
    vector<string> detections;

    string line;

    // train
    int cur_train_set = 0;
    int max_train_set = 5;

    for (int i = 0; i < 100; ++i) {
        vectors_clear(imageData);
        vectors_clear(header);
        vectors_clear(wcs);
        detections.clear();

        // width & hight
        std::getline(cin, line);
        int width = std::atoi(line.c_str());
        std::getline(cin, line);
        int height = std::atoi(line.c_str());
        int size = width * height;

        vectors_reserve(imageData, size);
        read_img(size, imageData[0], header[0], wcs[0]);
        read_img(size, imageData[1], header[1], wcs[1]);
        read_img(size, imageData[2], header[2], wcs[2]);
        read_img(size, imageData[3], header[3], wcs[3]);

        // detections
        std::getline(cin, line);
        int v_int = std::atoi(line.c_str());
        for (int n = 0; n < v_int; ++n) {
            std::getline(cin, line);
            detections.push_back(line);
            LOG << line << endl;
        }
        LOG << "====" << endl;

        int res = detector.trainingData(width, height, imageData[0], header[0], wcs[0], imageData[1], header[1], wcs[1], imageData[2], header[2], wcs[2], imageData[3], header[3], wcs[3], detections);

        // for debugging
        ++cur_train_set;
        if (cur_train_set >= max_train_set) {
            res = 1;
        }
        //save_train_set(cur_train_set, width, height, imageData, header, wcs, detections);
        //

        cout << res << endl;

        if (res == 1)
            break;
    }   // end of train

/*
    // test
    //int cur_test_set = 0;

    for (int i = 0; i < 20; ++i) {
        vectors_clear(imageData);
        vectors_clear(header);
        vectors_clear(wcs);

        std::getline(cin, line);
        string imageID = line;

        // width & hight
        std::getline(cin, line);
        int width = std::atoi(line.c_str());
        std::getline(cin, line);
        int height = std::atoi(line.c_str());
        int size = width * height;

        vectors_reserve(imageData, size);
        read_img(size, imageData[0], header[0], wcs[0]);
        read_img(size, imageData[1], header[1], wcs[1]);
        read_img(size, imageData[2], header[2], wcs[2]);
        read_img(size, imageData[3], header[3], wcs[3]);

        int res = detector.testingData(imageID, width, height, imageData[0], header[0], wcs[0], imageData[1], header[1], wcs[1], imageData[2], header[2], wcs[2], imageData[3], header[3], wcs[3]);
        cout << res << endl;

        //++cur_test_set;
        //save_test_set(cur_test_set, width, height, imageData, header, wcs);
    }

    // result

    vector<string> result = detector.getAnswer();

    // output result
    int size = result.size();
    cout << size << endl;
    for (int i = 0; i < size; ++i) {
        cout << result[i] << endl;
    }
*/
}


#endif // TEST_FAST
