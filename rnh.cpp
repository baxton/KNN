
//
// g++ -g  -mmmx -msse -msse2 -msse3  -D DEBUG -I. -I/c/Working/MinGW/lib/gcc/mingw32/4.8.1/include  prep_c.cpp -o prep_c.exe
// g++ -g -O2 -mmmx -msse -msse2 -msse3  -D DEBUG -I. -I/c/Working/MinGW/lib/gcc/mingw32/4.8.1/include  prep_c.cpp -o prep_c.exe
//



#include <vector>
#include <deque>
#include <algorithm>
#include <memory>
#include <pmmintrin.h>

#if defined DEBUG
  #include <iostream>
  #include <fstream>

  std::fstream log("prep_c.log", std::fstream::out);
#endif


#if defined DEBUG
  #define DBUG1(x1) {log << x1 << std::endl;}
  #define DBUG2(x1, x2) {log << x1 << x2 << std::endl;}
  #define DBUG_ARR(v) {for (int i = 0; i < v.size(); ++i) log << v[i] << ","; log << std::endl;}
  #define DBUG_ARR_STATIC(v, size) {for (int i = 0; i < size; ++i) log << v[i] << ","; log << std::endl;}
#else
  #define DBUG1(x1)
  #define DBUG2(x1, x2)
  #define DBUG_ARR(v)
  #define DBUG_ARR_STATIC(v, size)
#endif


void copy(void* dst, void* src, size_t size) {
    // size is in bytes
    // size is a multiple of 128 (32 byte allignment and pointer to float)
    float* pin = (float*)src;
    float* pout = (float*)dst;

    size_t iterations = size >> 7; // i.e. devide by 32 and by 4
    for (size_t i = 0; i < iterations; ++i) {
        __m128 z1 = _mm_load_ps(pin + 0);
        __m128 z2 = _mm_load_ps(pin + 4);
        __m128 z3 = _mm_load_ps(pin + 8);
        __m128 z4 = _mm_load_ps(pin + 12);
        __m128 z5 = _mm_load_ps(pin + 16);
        __m128 z6 = _mm_load_ps(pin + 20);
        __m128 z7 = _mm_load_ps(pin + 24);
        __m128 z8 = _mm_load_ps(pin + 28);
        _mm_store_ps(pout + 0, z1);
        _mm_store_ps(pout + 4, z2);
        _mm_store_ps(pout + 8, z3);
        _mm_store_ps(pout + 12, z4);
        _mm_store_ps(pout + 16, z5);
        _mm_store_ps(pout + 20, z6);
        _mm_store_ps(pout + 24, z7);
        _mm_store_ps(pout + 28, z8);
        pin += 32;
        pout += 32;
    }
}



using namespace std;



// this came from the problem description
#define MAX_N 1024






// 16 byte aligned
struct rect {
    enum ORIENTATION {
        O_AX = 0,
        O_AY = 1,
    };
    enum CONNECTION_POINT {
        CP_NO = 0,
        CP_TL = 1,
        CP_TR = 2,
        CP_BR = 4,
        CP_BL = 8,
    };

    rect () :
        o(O_AX),
        x(0),
        y(0),
        a(0),
        b(0),
        i(-1),
        h(0),
        w(0),
        len(0),
        used(0),
        cp(CP_NO),
        new_i(0),
        sq(0.)
    {}

    int change_orientation() {
        o = O_AY >> o;
        h = o == O_AX ? b : a;
        w = o == O_AX ? a : b;
        return o;
    }

    void init(int a, int b, int i) {
        this->a = a;
        this->b = b;
        this->i = i;
        new_i = i;
        o = a > b ? O_AX : O_AY;    // --
        h = o == O_AX ? b : a;
        w = o == O_AX ? a : b;
        len = w;    // will never change
        sq = (double)h / w;
    }

#if defined DEBUG
    ostream& print(ostream& os) const {
        os << "rect{" << (o == O_AX ? "AX" : "AY") << ", xy[" << x << "," << y << "] "
           << "ab[" << a << "," << b << "]}";
    }
#endif

    int o;  // orientation
    int x;
    int y;
    int a;
    int b;
    int i;      // index in A and B
    int h;
    int w;
    int len;
    int used;
    int cp;     // connection point associated with this rect
    int new_i;
    double sq;  // squareness h / w
} __attribute__((aligned(16)));

#if defined DEBUG
ostream& operator<< (ostream& os, const rect& r) {
    return r.print(os);
}
#endif

class connection_points {
    deque<int> storage;
    int prev_size;
public:
    connection_points() :
        storage(),
        prev_size(-1)
    {}

    bool empty() {return storage.empty();}

    void add(int i) {
        storage.push_back(i);
    }

    int next() {
        return storage.front();
    }

    // Logic:
    // 1) fix - remember current size
    // 2) add new points
    // 3) erase
    // 4) unfix
/*
    // [start, end] inclusive range
    void erase(int start, int end) {
        if (start < end) {
            if (fix >= end)
                storage.erase(storage.begin()+start, storage.begin()+end + 1);
            else {
                storage.erase(storage.begin(), storage.begin() + fix);
                end -= fix;
                storage.erase(storage.begin(), storage.begin() + end + 1);
            }
        }
        else if (start > end) {
            storage.erase(storage.begin()+start, storage.end());
            storage.erase(storage.begin(), storage.begin() + end + 1);
        }
        else {
            // erase all before fix
            storage.erase(storage.begin(), storage.begin() + fix);
        }
    }
*/
    void fix() {
        prev_size = storage.size();
    }
    void unfix() {
        prev_size = -1;
    }

} __attribute__((aligned(16)));


class Ray {
    rect rectangles_[MAX_N];
    connection_points cp_;

public:
    Ray() {}

    rect* rectangles() {
        return rectangles_;
    }

    connection_points& points() {
        return cp_;
    }

private:
    Ray(const Ray& tmp);
    Ray& operator=(const Ray&);
} __attribute__((aligned(16)));





// quick memory allocator
template<class T>
struct heap {
    enum {
        HEAP_SIZE = 1024,   // 2^10
        MASK = 1023,        // for quick mod(heap_size)
        ARRAY_SIZE_BYTES = MAX_N * sizeof(T),
    };
    int begin;
    int end;
    int count;

    T raw_heap[HEAP_SIZE] __attribute__((aligned(16)));
    int storage[HEAP_SIZE];

    T* get_object(int i) {return &raw_heap[i];}

    heap() :
        begin(0),
        end(0),
        count(HEAP_SIZE)
    {
        for (int i = 0; i < HEAP_SIZE; ++i)
            storage[i] = i;
    }

    int allocate() {
#if defined DEBUG
        if (!count)
            DBUG1("Out of memory")
#endif
        int b = storage[begin];
        --count;
        begin = ++begin & MASK;

        return b;
    }

    void release(int b) {
        ++count;
        storage[end] = b;
        end = ++end & MASK;
    }
};
heap<Ray> global_heap;

// auto handler
template<class T>
struct AutoObj {
    enum {
        NOTHING = -1,
    };

    int oh; // object handler
    int *pcnt;

    AutoObj() :
        oh(NOTHING),
        pcnt(NULL)
    {}

    AutoObj(const AutoObj& other) {
        pcnt = other.pcnt;
        oh = other.oh;
        *pcnt += 1;
    }

    AutoObj& operator= (const AutoObj& other) {
        AutoObj tmp(other);
        swap(tmp);
        return *this;
    }

    ~AutoObj() {free();}

    bool initialised() {return NOTHING != oh;}

    T* get() {return global_heap.get_object(oh);}

    void alloc() {
        pcnt = new int(1);
        if (pcnt)
            oh = global_heap.allocate();
    }

    void free() {
        if (NOTHING != oh) {
            *pcnt -= 1;
            if (0 == *pcnt) {
                global_heap.release(oh);
                delete pcnt;
            }
            oh = NOTHING;
            pcnt = NULL;
        }
    }

    void swap(AutoObj& other) {
        int tmp = oh;
        oh = other.oh;
        other.oh = tmp;

        int* ptmp = pcnt;
        pcnt = other.pcnt;
        other.pcnt = ptmp;
    }

    AutoObj<Ray> copy() {
        AutoObj<Ray> tmp;
        tmp.alloc();
        ::copy(tmp.get()->rectangles(), get()->rectangles(), MAX_N * sizeof(rect));
        //::copy(tmp.get().points(), get().points(), MAX_N * sizeof(connection_points));
        return tmp;
    }
};



//------------------------------------------------------------------------

typedef AutoObj<Ray> AutoRay;


class RectanglesAndHoles {
    enum {
        BEAM_WIDTH = 1,
    };

    int N;
    AutoRay beam[BEAM_WIDTH];

    // for sorting by squareness and length, DESC order
    struct cmp_sq_len_desc {
        bool operator() (const rect& r1, const rect& r2) {
            return (r1.sq > r2.sq) || (r1.sq == r2.sq and r1.len > r2.len);
        }
    };

public:
    RectanglesAndHoles() :
        N(0)
    {}


    void make_child(Ray& ray) {

    }

    void run_beam() {/*
        for (int b = 0; b < BEAM_WIDTH; ++b) {
            ray = beam[b];
            array c = make_child(ray);
            ray.swap(c);
        }*/
    }

    // what it does is initialize new_indices after sorting
    void update_indices(rect* rectangles) {
        for (int i = 0; i < N; ++i)
            rectangles[i].new_i = i;
    }

    int start(vector<int>& A, vector<int>&B) {
        // logic:
        //  1) initialize the 1st array
        //  2) if no more rectangles then exit
        //  3) get all N children
        //  4) select K children out of N
        //  5) for each of K children call #2

        // init 1st array and populate it
        beam[0].alloc();
        Ray* pray = beam[0].get();
        rect* rectangles = pray->rectangles();
        for (int i = 0; i < N; ++i) {
            rect& r = rectangles[i];
            r.init(A[i], B[i], i);
        }
for (int i = 0; i < 1000000; ++i)
        sort(&rectangles[0], &rectangles[N], cmp_sq_len_desc());
        update_indices(rectangles);

        // put the very first rect to [0,0]
        rectangles[0].x = 0;
        rectangles[0].y = 0;
        rectangles[0].used = 1;
        rectangles[0].cp = rect::CP_TR;


        run_beam();

        // return index of the best ray
        return 0;   // TODO get_the_best();
    }

    vector<int> place(vector<int>& A, vector<int>&B) {
        N = A.size();

        int best_ray_idx = start(A, B);

        // all rectangles which are not used will be put into a straight row
        vector<int> result(N*3, 0);
        int x = 0;
        int y = -5000;
        rect* rectangles = beam[best_ray_idx].get()->rectangles();
        for (int i = 0; i < N; ++i) {
            rect& r = rectangles[i];
            if (r.used) {
                result[r.i*3+0] = r.x;
                result[r.i*3+1] = r.y;
                result[r.i*3+2] = r.o;
            }
            else {
                result[r.i*3+0] = x;
                result[r.i*3+1] = y;
                result[r.i*3+2] = r.o;
                x += r.w;
            }
        }

        return result;
    }
};



#if defined DEBUG

void read_from_cin(vector<int>& A, vector<int>& B, int& N);

int main(int argc, const char* argv[]) {
    vector<int> A;
    vector<int> B;
    int N;

    read_from_cin(A, B, N);

    // some debug info
    DBUG2("Read N: ", N)
    DBUG1("")
    DBUG1("A")
    DBUG_ARR(A)
    DBUG1("")
    DBUG1("B")
    DBUG_ARR(B)
    DBUG1("")

    // process
    RectanglesAndHoles c = RectanglesAndHoles();
    vector<int> result = c.place(A, B);

    // output the result
    for (int i = 0; i < N; ++i) {
        cout << result[i*3] << endl;
        cout << result[i*3+1] << endl;
        cout << result[i*3+2] << endl;
    }

    DBUG1("---DONE---")

    return 0;
}

void read_from_cin(vector<int>& A, vector<int>& B, int& N) {
    cin >> N;
    A.reserve(N);
    B.reserve(N);
    int c;
    for (int i = 0; i < N; ++i) {
        cin >> c;
        A.push_back(c);
    }
    for (int i = 0; i < N; ++i) {
        cin >> c;
        B.push_back(c);
    }
}


#endif
