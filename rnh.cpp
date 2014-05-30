
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
#define MAX_N_MASK (MAX_N-1)

#define NO_X  -2000000
#define NO_Y  -2000000


#define DIR_TL  1
#define DIR_TR  2
#define DIR_BR  4
#define DIR_BL  8






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

    bool intersect(const rect& other) const {
        bool no_intersection = (x + w <= other.x) || (other.x + other.w <= x) ||
                               (y + h <= other.y) || (other.y + other.h <= y);
        return !no_intersection;
    }

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
    int storage[MAX_N] __attribute__((aligned(16)));
    int begin;
    int end;
    int count;
    int prev_size;
public:
    connection_points() :
        begin(0),
        end(0),
        count(0),
        prev_size(-1)
    {}

    bool empty() {return !count;}

    void copy(connection_points& cps) {
        ::copy(cps.storage, storage, MAX_N * sizeof(int));
        cps.begin = begin;
        cps.end = end;
        cps.count = count;
        cps.prev_size = prev_size;
    }

    void add(int i) {
#if defined DEBUG
        if (count == MAX_N)
            DBUG1("ERROR: Connection point queue is FULL")
#endif
        storage[begin] = i;
        begin = ++begin & MAX_N_MASK;
        ++count;
    }

    int next() {
        return storage[end];
    }

    void pop() {
#if defined DEBUG
        if (!count)
            DBUG1("ERROR: Connection point queue is empty")
#endif
        end = ++end & MAX_N_MASK;
        --count;
    }


} __attribute__((aligned(16)));


class Ray {
    rect rectangles_[MAX_N];
    connection_points cp_;
    int indices_sorted_[MAX_N];
    int N;

    // for sorting by squareness and length, DESC order
    struct cmp_sq_len_desc {
        const rect* rectangles_;

        cmp_sq_len_desc(const rect* rectangles) :
            rectangles_(rectangles)
        {}

        bool operator() (int a, int b) {
            const rect& r1 = rectangles_[a];
            const rect& r2 = rectangles_[b];
            return (r1.sq > r2.sq) || (r1.sq == r2.sq and r1.len > r2.len);
        }
    };

public:
    Ray() : N(0) {
        for (int i = 0; i < MAX_N; ++i) {
            indices_sorted_[i] = i;
        }
    }

    void set_N(int n) { N = n; }

    int* arg_sort() {
        std::sort(&indices_sorted_[0], &indices_sorted_[N], cmp_sq_len_desc(rectangles_));
        return indices_sorted_;
    }



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

    bool initialised() const {return NOTHING != oh;}

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
        get()->points().copy(tmp.get()->points());
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

public:
    RectanglesAndHoles() :
        N(0)
    {}


    bool has_intersection(const rect& r, const rect* rectangles) const {
        for (int i = 0; i < N; ++i) {
            const rect& cur = rectangles[i];
            if (cur.used && cur.i != r.i) {
                if (r.intersect(cur)) {
                    DBUG2("Intersect: ", cur.i)
                    return true;
                }
            }
        }
        return false;
    }

    void get_leg_tl(const rect* start, rect* rectangles, int* indices_sorted, int& current_idx, int leg_len, rect** finish,
                    int x_block=NO_X, int y_block=NO_Y, bool check_intersection=true) {

        if (!start) {
            DBUG1("ERROR: start was not provided for DIR_TL")
            return;
        }

        *finish = NULL;

        int cur_len = 0;

        const rect* prev = start;

        for (; current_idx < N; ++current_idx) {
            rect& r = rectangles[indices_sorted[current_idx]];
            if (r.used) {
                DBUG2("Skip used: ", r.i)
            }
            else {
                DBUG2("Rect in TL: ", r.i)

                //if (r.o == rect::O_AX)
                //    r.change_orientation();

                if ((x_block == NO_X || x_block < prev->x - r.w) && (y_block == NO_Y || y_block > prev->y + prev->h)) {
                    r.x = prev->x - r.w;
                    r.y = prev->y + prev->h;

                    if (check_intersection && has_intersection(r, rectangles)) {
                        DBUG1("intersection!!!")
                        break;
                    }
                    else {
                        r.used = 1;
                        prev = &r;
                        *finish = &r;
                        ++cur_len;
                        if (cur_len >= leg_len) {
                            ++current_idx;
                            break;
                        }
                    }
                }
                else{
                    break;
                }
            }
        }   // while
    }

    void get_leg_bl(const rect* start, rect* rectangles, int* indices_sorted, int& current_idx, int leg_len, rect** finish, vector<int>& indices,
                 int x_block=NO_X, int y_block=NO_Y, bool check_intersection=true) {

        if (!start) {
            DBUG1("ERROR: start was not provided for DIR_BL")
            return;
        }

        *finish = NULL;

        int cur_len = 0;

        const rect* prev = start;

        for (; current_idx < N; ++current_idx) {
            rect& r = rectangles[indices_sorted[current_idx]];
            if (r.used) {
                DBUG2("Skip used: ", r.i)
            }
            else {
                DBUG2("Rect in BL: ", r.i)

                //if (r.o == rect::O_AY)
                //    r.change_orientation();

                if ((x_block == NO_X || x_block < prev->x - r.w) && (y_block == NO_Y || y_block < prev->y - r.h)) {
                    r.x = prev->x - r.w;
                    r.y = prev->y - r.h;

                    if (check_intersection && has_intersection(r, rectangles)) {
                        DBUG1("intersection!!!")
                        break;
                    }
                    else {
                        r.used = 1;
                        prev = &r;
                        *finish = &r;
                        indices.push_back(current_idx);
                        ++cur_len;
                        if (cur_len >= leg_len) {
                            ++current_idx;
                            break;
                        }
                    }
                }
                else {
                    break;
                }
            }   // while
        }
    }



    bool get_leg(const rect* start, rect* rectangles, int* indices_sorted, int& current_idx, int leg_len, int direction, rect** finish,
                 int x_block=NO_X, int y_block=NO_Y) {
        bool result = false;
        int cur_len = 0;

        DBUG2("Direction: ", direction)

        const rect* prev = start;

        for (; current_idx < N; ++current_idx) {
            rect& r = rectangles[indices_sorted[current_idx]];
            if (r.used) {
                DBUG2("Skip: ", r.used)
                continue;
            }

            bool goon = true;

            switch (direction) {
            case DIR_TL:
                //if (r.o == rect::O_AX)
                //    r.change_orientation();
                if ((x_block == NO_X || x_block < prev->x - r.w) && (y_block == NO_Y || y_block > prev->y + prev->h)) {
                    r.x = prev->x - r.w;
                    r.y = prev->y + prev->h;
                }
                else
                    goon = false;
                break;

            case DIR_TR:
                //if (r.o == rect::O_AY)
                //    r.change_orientation();
                if ((x_block == NO_X || x_block > prev->x + prev->w) && (y_block == NO_Y || y_block > prev->y + prev->h)) {
                    r.x = prev->x + prev->w;
                    r.y = prev->y + prev->h;
                }
                else
                    goon = false;
                break;

            case DIR_BR:
                //if (r.o == rect::O_AX)
                //    r.change_orientation();
                if ((x_block == NO_X || x_block > prev->x + prev->w) && (y_block == NO_Y || y_block < prev->y - r.h)) {
                    r.x = prev->x + prev->w;
                    r.y = prev->y - r.h;
                }
                else
                    goon = false;
                break;

            case DIR_BL:
                //if (r.o == rect::O_AY)
                //    r.change_orientation();
                if ((x_block == NO_X || x_block < prev->x - r.w) && (y_block == NO_Y || y_block < prev->y - r.h)) {
                    r.x = prev->x - r.w;
                    r.y = prev->y - r.h;
                }
                else
                    goon = false;
                break;
            }

            if (goon) {
                if (has_intersection(r, rectangles)) {
                    DBUG1("intersection!!!")
                    break;
                }
                else {
                    r.used = 1;
                    prev = &r;
                    *finish = &r;
                    ++cur_len;
                    if (cur_len >= leg_len) {
                        ++current_idx;
                        break;
                    }
                }
            }
            else{
                break;
            }
        }   // while

        return (cur_len == leg_len);
    }


    void make_romb(rect* start, rect* rectangles, int* indices_sorted, int* start_idx, int leg_len, rect** ret_bottom, rect** ret_left, rect** ret_top, rect** ret_right) {

        int current_idx = *start_idx;

        rect* bottom = start;
        rect* left = *ret_left;
        rect* top = *ret_top;
        rect* right = *ret_right;

        vector<int> l1;
        vector<int> l2;
        vector<int> l3;
        vector<int> l4;

        rect* finish = NULL;
        if (!left) {
            get_leg_tl(start, rectangles, indices_sorted, current_idx, leg_len, &finish, NO_X, NO_Y, false);
            left = finish;
        }
        else {
            finish = left;
        }

        if (!top) {
            get_leg(finish, rectangles, indices_sorted, current_idx, leg_len, DIR_TR, &finish, start->x+start->w, NO_Y);
            top = finish;
        }
        else {
            finish = top;
        }

        if (!right) {
            get_leg(finish, rectangles, indices_sorted, current_idx, leg_len, DIR_BR, &finish, NO_X, left->y);
            right = finish;
        }
        else {
            finish = right;
        }

        vector<int> indices_bl;
        indices_bl.reserve(leg_len);

        rect* tmp = NULL;
        get_leg_bl(finish, rectangles, indices_sorted, current_idx, leg_len, &tmp, indices_bl, start->x + start->w, start->y);

        if (tmp)
            finish_romb_b(rectangles, indices_sorted, current_idx, &top, &right, &bottom, &left, tmp, indices_bl);

        *ret_bottom = bottom;
        *ret_left = left;
        *ret_top = top;
        *ret_right = right;

        *start_idx = current_idx;
    }

    void finish_romb_b(rect* rectangles, int* indices_sorted, int& current_idx,
                       rect** ret_top, rect** ret_right, rect** ret_bottom, rect** ret_left, rect* finish, vector<int>& indices_finish) {
        rect* top = *ret_top;
        rect* right = *ret_right;
        rect* bottom = *ret_bottom;
        rect* left = *ret_left;

        DBUG2("Bottom: ", bottom->i)
        DBUG2("Right: ", right->i)
        DBUG2("Finish: ", finish->i)

        // converge
        int y_block = (bottom->y + finish->y) / 2;
        rect* tmp_b = bottom;
        while (tmp_b->y + tmp_b->h < y_block) {
            rect* r = &rectangles[indices_sorted[current_idx++]];
            if (rect::O_AX == r->o)
                r->change_orientation();
#if defined DEBUG
            if (r->used)
                DBUG2("ERROR: used rect: ", r->i);
#endif

            r->x = tmp_b->x + tmp_b->w;
            r->y = tmp_b->y + tmp_b->h;

            if (!has_intersection(*r, rectangles)) {
                DBUG2("Add to bottom: ", r->i)
                r->used = 1;
                tmp_b = r;
            }
            else {
                break;
            }
        }

//        while (finish->y > tmp_b->y + tmp_b->h) {

//        }


        // connect
        //int
        //if (bottom->y < finish->y)

        *ret_bottom = bottom;
        *ret_left = left;
        *ret_top = top;
        *ret_right = right;
    }

    void run_beam() {
        Ray* pray = beam[0].get();
        rect* rectangles = pray->rectangles();

        rect* ret_bottom = NULL;
        rect* ret_left   = NULL;
        rect* ret_top    = NULL;
        rect* ret_right  = NULL;

        rect* start = &rectangles[0];
        start->x = 0;
        start->y = 0;
        start->used = 1;

        int start_idx = 1;

        int leg_len = 80;

        int* indices_sorted = pray->arg_sort();

        make_romb(start, rectangles, indices_sorted, &start_idx, leg_len, &ret_bottom, &ret_left, &ret_top, &ret_right);
        for (int i = 0; i < 0; ++i) {
            if (!ret_right || ! ret_top)
                break;

            start = ret_right;
            rect* ret_bottom2 = NULL;
            rect* ret_left2   = ret_top;
            rect* ret_top2    = NULL;
            rect* ret_right2  = NULL;

            make_romb(start, rectangles, indices_sorted, &start_idx, leg_len, &ret_bottom2, &ret_left2, &ret_top2, &ret_right2);

            ret_bottom = ret_bottom2;
            ret_top = ret_top2;
            ret_right = ret_right2;
            ret_left = ret_left2;
        }
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
        pray->set_N(N);
        rect* rectangles = pray->rectangles();
        for (int i = 0; i < N; ++i) {
            rect& r = rectangles[i];
            r.init(A[i], B[i], i);
        }

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
        int y = -50000;
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
