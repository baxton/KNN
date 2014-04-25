

#include <cstring>
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <ctime>


using namespace std;

// types
typedef unsigned long long UINT; // 8 bytes

// consts
#define BYTE_BITS 8
#define BUFFER_ALIGNMENT 4





//////////////////////////////////////////////////////////////////////////////////////
// BOARD
//////////////////////////////////////////////////////////////////////////////////////

struct Board {
    int board[225];
} __attribute__((aligned(4))) ;

//////////////////////////////////////////////////////////////////////////////////////
// END BOARD
//////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////
// MEMORY
//////////////////////////////////////////////////////////////////////////////////////

#define NO_BOARD (UINT)-1

#define MEGA_BUFFER_SIZE 500000

Board* mega_buffer;

list<UINT>& get_free_boards() {
    static list<UINT> free_boards;
    return free_boards;
}

UINT alloc_board() {
    list<UINT>& boards = get_free_boards();
    if (!boards.empty()) {
        UINT i = boards.front();
        boards.pop_front();
        return i;
    }

    cout << "CRIT: not enough boards" << endl;    

    return (UINT)-1; 
}

void free_board(UINT i) {
    get_free_boards().push_front(i);
}

Board* get_board(UINT i) {
    return &mega_buffer[i];
}


class PtrBoard {
    UINT handler_;
public:
    PtrBoard() : handler_(NO_BOARD) {}
    ~PtrBoard() {free();}

    PtrBoard& operator= (PtrBoard& other) {
        if (this != &other) {
            free();
            handler_ = other.handler_;
            other.handler_ = NO_BOARD;
        }
        return *this;
    }

    int* board() {
        return get_board(handler_)->board;
    }
    
    void alloc() {
        free();
        handler_ = alloc_board();
    }

    void free() {
        if (NO_BOARD != handler_) {
            free_board(handler_);
            handler_ = NO_BOARD;
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////
// END MEMORY
//////////////////////////////////////////////////////////////////////////////////////



const int masks[] = {0x00000000,
                     0x01010101,
                     0x02020202,
                     0x03030303,
                     0x04040404,
                     0x05050505
                };


//////////////////////////////////////////////////////////////////////////////////////
// INIT
//////////////////////////////////////////////////////////////////////////////////////

struct Init {
    Init() {

        // for now I ignore freeing
        mega_buffer = (Board*)malloc(MEGA_BUFFER_SIZE * sizeof(Board));
        
        //
        list<UINT>& boards = get_free_boards();
        for (UINT i = 0; i < MEGA_BUFFER_SIZE; ++i) {
            boards.push_front(i);
        }
    }

    ~Init() {
        // free resources here
    }
} init;

//////////////////////////////////////////////////////////////////////////////////////
// END INIT
//////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////
// GENERATOR
//////////////////////////////////////////////////////////////////////////////////////
 
class Generator {
    UINT colors_;
    UINT prev_;
    int line_;
    

public:
    Generator(UINT colors, UINT startSeed) :
        colors_(colors),
        prev_(startSeed),
        line_(0)
    {}

    Generator(const Generator& other) {
        colors_ = other.colors_;
        prev_ = other.prev_;
        line_ = other.line_;
    }

    Generator& operator=(const Generator& other) {
        if (this != &other) {
            colors_ = other.colors_;
            prev_ = other.prev_;
            line_ = other.line_;
        }
        return *this;
    }

    void swap(Generator& other) {
        UINT tmp = colors_;
        colors_ = other.colors_;
        other.colors_ = tmp;

        tmp = prev_;
        prev_ = other.prev_;
        other.prev_ = tmp;

        int btmp = line_;
        line_ = other.line_;
        other.line_ = btmp;
    }

    void generate_list(char* list, size_t size) {
        for (int i = 0; i < size; ++i) {
            if (line_) 
                prev_ = (prev_ * 48271) % 2147483647;
            ++line_;
            list[i] = (char)(prev_ % colors_);
        }
    }

    int next() {
        if (line_)
            prev_ = (prev_ * 48271) % 2147483647;
        ++line_;
        return prev_ % colors_;
    }

    int get_line() const {return line_;}
};

//////////////////////////////////////////////////////////////////////////////////////
// END GENERATOR
//////////////////////////////////////////////////////////////////////////////////////





void fill_from_vector(const vector<string>& v, int* board) {
    size_t n = v.size();
    int* p = (int*)board;
    int idx = 0;
    for (size_t i = 0; i < n-1; ++i) {
        for (size_t c = 0; c < n-1; ++c) {
            p[idx++] = (v[i+0][c+0] - 0x30)          | ((v[i+0][c+1] - 0x30) << 8) |
                       ((v[i+1][c+0] - 0x30) << 16)  | ((v[i+1][c+1] - 0x30) << 24);
        }
    }
}


bool validate_board(int* board, int n, bool print=false) {
    int tmp1[n*n];

    int idx = 0;
    int i = 0;
    int size = (n-1)*(n-1);
    for (; i < size; ++i) {
        unsigned char* p = (unsigned char*)&board[i];
        tmp1[idx++] = (int)p[0];
        if (n-2 == i%(n-1)) {
            tmp1[idx++] = (int)p[1];
        }
    }
    i -= n-1;
    for (; i < size; ++i) {
        unsigned char* p = (unsigned char*)&board[i];
        tmp1[idx++] = (int)p[2];
        if (n-2 == i%(n-1))
            tmp1[idx++] = (int)p[3];
    }
    
    if (print) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                cout << tmp1[i*n+j];
            }
            cout << endl;
        }
        cout << endl;
    }

    PtrBoard tmp2;
    tmp2.alloc();
   
    int* p = (int*)tmp2.board();
    idx = 0;
    for (size_t i = 0; i < n-1; ++i) {
        for (size_t c = 0; c < n-1; ++c) {
            p[idx++] = tmp1[i*n+c]             | (tmp1[i*n+c+1] << 8) |
                       (tmp1[(i+1)*n+c] << 16) | (tmp1[(i+1)*n+c+1] << 24);
        }
    }

    for (int i = 0; i < size; ++i) {
        if (tmp2.board()[i] != board[i]) {
            cout << "ERROR: validation [" << i << "] [" << tmp2.board()[i] << "] [" << board[i] << "]" << endl;
            return false;
        }
    }

    return true; 
}


void print_smart(int* board, int n) {
    int i = 0;
    int size = (n-1)*(n-1);
    for (; i < size; ++i) {
        unsigned char* p = (unsigned char*)&board[i];
        cout << (int)p[0];
        if (n-2 == i%(n-1)) {
            cout << (int)p[1];  
            cout << endl;
        }
    }
    i -= n-1;
    for (; i < size; ++i) {
        unsigned char* p = (unsigned char*)&board[i];
        cout << (int)p[2];
        if (n-2 == i%(n-1)) 
            cout << (int)p[3];
    }
    cout << endl << endl;
}

int search_quare(int* board, int colors, int size, int start) {
    for (int i = start; i < size; ++i) {
        for (int c = 0; c < colors; ++c) {
            if (masks[c] == board[i])
                return i;
        }
    }
    return -1;
}

int process_board(int* board, int n, Generator& g, int colors, int size) {
    int total_score = 0;
    int idx = search_quare(board, colors, size, 0);
    while (0 <= idx) {
        ++total_score;
        
        int new_plate = 0;  
        g.generate_list((char*)&new_plate, 4);

        int i = idx / (n-1);
        int j = idx - (i*(n-1));
//cout << "idx: " << idx << " [" << i << ", " << j << "]" << endl;
//cout << "New: " << new_plate << endl;
        int b;
        int start;

        if (0 == i && 0 == j) {
            b = idx;
            start = b;
            board[b] = new_plate;
            ++b;
            board[b] = (board[b] & 0xFF00FF00) | ((new_plate & 0xFF00FF00) >> 8);
            b+=n-2;
            board[b] = (board[b] & 0xFFFF0000) | ((new_plate & 0xFFFF0000) >> 16);
            ++b;
            board[b] = (board[b] & 0xFFFFFF00) | ((new_plate & 0xFF000000) >> 24);
        }
        else if (0 == i && j == n-2) {
            b = idx-1;  
            start = b;  
            board[b] = (board[b] & 0x00FF00FF) | ((new_plate & 0x00FF00FF) << 8);
            ++b;
            board[b] = new_plate;
            b+=n-2;
            board[b] = (board[b] & 0xFFFF00FF) | ((new_plate & 0x00FF0000) >> 8);
            ++b;
            board[b] = (board[b] & 0xFFFF0000) | ((new_plate & 0xFFFF0000) >> 16);
        }
        else if (i == n-2 && j == n-2) {
            b = idx - n;
            start = b;
            board[b] = (board[b] & 0x00FFFFFF) | ((new_plate & 0x000000FF) << 24);
            ++b;
            board[b] = (board[b] & 0x0000FFFF) | ((new_plate & 0x0000FFFF) << 16);
            b+=n-2;
            board[b] = (board[b] & 0x00FF00FF) | ((new_plate & 0x00FF00FF) << 8);
            ++b;
            board[b] = new_plate;
        }
        else if (i == n-2 && 0 == j) {
            b = idx - (n-1);
            start = b;
            board[b] = (board[b] & 0x0000FFFF) | ((new_plate & 0x0000FFFF) << 16);
            ++b;
            board[b] = (board[b] & 0xFF00FFFF) | ((new_plate & 0x0000FF00) << 8);
            b+=n-2;
            board[b] = new_plate;
            ++b;
            board[b] = (board[b] & 0xFF00FF00) | ((new_plate & 0xFF00FF00) >> 8);
        }
        else if (0 == i) {
            b = idx - 1;
            start = b;
            board[b] = (board[b] & 0x00FF00FF) | ((new_plate & 0x00FF00FF) << 8);
            ++b;
            board[b] = new_plate;
            ++b;
            board[b] = (board[b] & 0xFF00FF00) | ((new_plate & 0xFF00FF00) >> 8);
            b+=n-3;
            board[b] = (board[b] & 0xFFFF00FF) | ((new_plate & 0x000000FF) << 8);
            ++b;
            board[b] = (board[b] & 0xFFFF0000) | ((new_plate & 0xFFFF0000) >> 16);
            ++b;
            board[b] = (board[b] & 0xFFFFFF00) | ((new_plate & 0xFF000000) >> 24);
        }
        else if (0 == j) {
            b = idx - (n-1);
            start = b;
            board[b] = (board[b] & 0x0000FFFF) | ((new_plate & 0x0000FFFF) << 16);
            ++b;
            board[b] = (board[b] & 0xFF00FFFF) | ((new_plate & 0x0000FF00) << 8);
            b+=n-2;
            board[b] = new_plate;
            ++b;
            board[b] = (board[b] & 0xFF00FF00) | ((new_plate & 0xFF00FF00) >> 8);
            b+=n-2;
            board[b] = (board[b] & 0xFFFF0000) | ((new_plate & 0xFFFF0000) >> 16);
            ++b;
            board[b] = (board[b] & 0xFFFFFF00) | ((new_plate & 0xFF000000) >> 24);
        }
        else if (j == n-2) {
            b = idx - n;
            start = b;
            board[b] = (board[b] & 0x00FFFFFF) | ((new_plate & 0x000000FF) << 24);
            ++b;
            board[b] = (board[b] & 0x0000FFFF) | ((new_plate & 0x0000FFFF) << 16);
            b+=n-2;
            board[b] = (board[b] & 0x00FF00FF) | ((new_plate & 0x00FF00FF) << 8);            
            ++b;
            board[b] = new_plate;
            b+=n-2;
            board[b] = (board[b] & 0xFFFF00FF) | ((new_plate & 0x00FF0000) >> 8);
            ++b;
            board[b] = (board[b] & 0xFFFF0000) | ((new_plate & 0xFFFF0000) >> 16);
        }
        else if (i == n-2) {
            b = idx - n;
            start = b;
            board[b] = (board[b] & 0x00FFFFFF) | ((new_plate & 0x000000FF) << 24);
            ++b;
            board[b] = (board[b] & 0x0000FFFF) | ((new_plate & 0x0000FFFF) << 16);
            ++b;
            board[b] = (board[b] & 0xFF00FFFF) | ((new_plate & 0x0000FF00) << 8);
            b+=n-3;
            board[b] = (board[b] & 0x00FF00FF) | ((new_plate & 0x00FF00FF) << 8);
            ++b;
            board[b] = new_plate;
            ++b;
            board[b] = (board[b] & 0xFF00FF00) | ((new_plate & 0xFF00FF00) >> 8);
        }
        else {
            b = idx - n;
            start = b;
            board[b] = (board[b] & 0x00FFFFFF) | ((new_plate & 0x000000FF) << 24);
            ++b;
            board[b] = (board[b] & 0x0000FFFF) | ((new_plate & 0x0000FFFF) << 16);
            ++b;
            board[b] = (board[b] & 0xFF00FFFF) | ((new_plate & 0x0000FF00) << 8);
            b+=n-3;
            board[b] = (board[b] & 0x00FF00FF) | ((new_plate & 0x00FF00FF) << 8);
            ++b;
            board[b] = new_plate;
//cout << "full: " << b << endl;
            ++b;
            board[b] = (board[b] & 0xFF00FF00) | ((new_plate & 0xFF00FF00) >> 8);
            b+=n-3;
            board[b] = (board[b] & 0xFFFF00FF) | ((new_plate & 0x00FF0000) >> 8);
            ++b;
            board[b] = (board[b] & 0xFFFF0000) | ((new_plate & 0xFFFF0000) >> 16);
            ++b;
            board[b] = (board[b] & 0xFFFFFF00) | ((new_plate & 0xFF000000) >> 24);
//print_smart(board, n);
        }

        // next
        idx = search_quare(board, colors, size, start);

    }
    return total_score;
}


void move_right(int i, int j, int* board, int* new_board, int n) {

    if (0 == i && 0 == j) {
        int v1 = board[0] & 0x000000FF;
        int v2 = (board[0] & 0x0000FF00) >> 8;
        new_board[0] = (board[0] & 0xFFFF0000) | v2 | (v1 << 8);
        new_board[1] = (board[1] & 0xFFFFFF00) | v1;
        int e = (n-1)*(n-1);
        for (int i = 2; i < e; ++i)
            new_board[i] = board[i];
    }
    else if (0 == i && n-2 == j) {
        int b = 0;
        int e = n-3;
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = board[b+1] & 0x000000FF;
        int v2 = (board[b+1] & 0x0000FF00) >> 8;
        new_board[b] = (board[b] & 0xFFFF00FF) | (v2 << 8);
        ++b;
        new_board[b] = (board[b] & 0xFFFF0000) | v2 | (v1 << 8);
        ++b;
        e = (n-1)*(n-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
    }
    else if (n-1 == i && 0 == j) {
        int b = 0;
        int e = (n-1)*(n-1)-(n-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = (board[b] & 0x00FF0000) >> 16;
        int v2 = (board[b] & 0xFF000000) >> 24;
        new_board[b] = (board[b] & 0x0000FFFF) | (v2 << 16) | (v1 << 24);
        ++b;
        new_board[b] = (board[b] & 0xFF00FFFF) | (v1 << 16);
        ++b;
        e = (n-1)*(n-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
    }
    else if (n-1 == i && n-2 == j) {
        int b = 0;
        int e = (n-1)*(n-1) - 2;
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = (board[b+1] & 0x00FF0000) >> 16;
        int v2 = (board[b+1] & 0xFF000000) >> 24;
        new_board[b] = (board[b] & 0x00FFFFFF) | (v2 << 24);
        ++b;
        new_board[b] = (board[b] & 0x0000FFFF) | (v2 << 16) | (v1 << 24);
    }
    else if (0 == i) {
        int b = 0;
        int e = j-1;
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = board[b+1] & 0x000000FF;
        int v2 = (board[b+1] & 0x0000FF00) >> 8;
        new_board[b] = (board[b] & 0xFFFF00FF) | (v2 << 8);
        ++b;
        new_board[b]   = (board[b]   & 0xFFFF0000) | v2 | (v1 << 8);
        ++b;
        new_board[b] = (board[b] & 0xFFFFFF00) | v1;
        ++b;
        e = (n-1)*(n-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
    }
    else if (n-2 == j) {
        int b = 0;
        int e = (i-1)*(n-1) + (j-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = board[b+n] & 0x000000FF;
        int v2 = (board[b+n] & 0x0000FF00) >> 8;
        new_board[b] = (board[b] & 0x00FFFFFF) | (v2 << 24);
        ++b;
        new_board[b] = (board[b] & 0x0000FFFF) | (v2 << 16) | (v1 << 24);
        ++b;
        e = b+n-3;
        for (; b < e; ++b)
            new_board[b] = board[b];
        new_board[b] = (board[b] & 0xFFFF00FF) | (v2 << 8);
        ++b;
        new_board[b] = (board[b] & 0xFFFF0000) | v2 | (v1 << 8);
        ++b;
        e = (n-1)*(n-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
    }
    else if (n-1 == i) {
        int b = 0;
        int size = (n-1)*(n-1);
        int e = size - (n-j);
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = (board[b+1] & 0x00FF0000) >> 16;
        int v2 = (board[b+1] & 0xFF000000) >> 24;
        new_board[b] = (board[b] & 0x00FFFFFF) | (v2 << 24);
        ++b;
        new_board[b] = (board[b] & 0x0000FFFF) | (v2 << 16) | (v1 << 24);
        ++b;
        new_board[b] = (board[b] & 0xFF00FFFF) | (v1 << 16);
        ++b;
        e = size;
        for (; b < e; ++b)
            new_board[b] = board[b];
    }
    else if (0 == j) {
        int b = 0;
        int size = (n-1)*(n-1);
        int e = (i-1)*(n-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = (board[b] & 0x00FF0000) >> 16;
        int v2 = (board[b] & 0xFF000000) >> 24;
        new_board[b] = (board[b] & 0x0000FFFF) | (v2 << 16) | (v1 << 24);
        ++b;
        new_board[b] = (board[b] & 0xFF00FFFF) | (v1 << 16);
        ++b;
        e = b + n-3;
        for (; b < e; ++b)
            new_board[b] = board[b];
        new_board[b] = (board[b] & 0xFFFF0000) | v2 | (v1 << 8);
        ++b;
        new_board[b] = (board[b] & 0xFFFFFF00) | v1;
        ++b;
        e = size;
        for (; b < e; ++b)
            new_board[b] = board[b];
    }
    else {
        int b = 0;
        int size = (n-1)*(n-1);
        int e = (i-1)*(n-1) + (j-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = board[b+n] & 0x000000FF;
        int v2 = (board[b+n] & 0x0000FF00) >> 8;
        new_board[b] = (board[b] & 0x00FFFFFF) | (v2 << 24);
        ++b;
        new_board[b] = (board[b] & 0x0000FFFF) | (v2 << 16) | (v1 << 24);
        ++b;
        new_board[b] = (board[b] & 0xFF00FFFF) | (v1 << 16);
        ++b;
        e = b + n-4;
        for (; b < e; ++b)
            new_board[b] = board[b];
        new_board[b] = (board[b] & 0xFFFF00FF) | (v2 << 8);
        ++b;
        new_board[b] = (board[b] & 0xFFFF0000) | v2 | (v1 << 8);
        ++b;
        new_board[b] = (board[b] & 0xFFFFFF00) | v1;
        ++b;
        e = size;
        for (; b < e; ++b)
            new_board[b] = board[b];
    }
}


void move_down(int i, int j, int* board, int* new_board, int n) {
    if (0 == i && 0 == j) {
        int b = 0;
        int v1 = board[b] & 0x000000FF;
        int v2 = (board[b] & 0x00FF0000) >> 16;
        new_board[b] = (board[b] & 0xFF00FF00) | v2 | (v1 << 16);
        ++b;
        int e = b + n - 2;
        for (; b < e; ++b)
            new_board[b] = board[b];
        new_board[b] = (board[b] & 0xFFFFFF00) | v1;
        ++b;
        int size = (n-1)*(n-1);
        e = size;
        for (; b < e; ++b)
            new_board[b] = board[b];
    }
    else if (0 == i && n-1 == j) {
        int b = 0;
        int e = n-2;
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = (board[b] & 0x0000FF00) >> 8;
        int v2 = (board[b] & 0xFF000000) >> 24;
        new_board[b] = (board[b] & 0x00FF00FF) | (v2 << 8) | (v1 << 24);
        ++b;
        e = b + n-2;
        for (; b < e; ++b)
            new_board[b] = board[b];
        new_board[b] = (board[b] & 0xFFFF00FF) | (v1 << 8);
        ++b;
        int size = (n-1)*(n-1);
        e = size;
        for (; b < e; ++b)
            new_board[b] = board[b];
    }
    else if (n-2 == i && n-1 == j) {
        int b = 0;
        int e = (i-1)*(n-1) + (j-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = (board[b+n-1] & 0x0000FF00) >> 8;
        int v2 = (board[b+n-1] & 0xFF000000) >> 24;
        new_board[b] = (board[b] & 0x00FFFFFF) | (v2 << 24);
        ++b;
        e = b + n-2;
        for (; b < e; ++b)
            new_board[b] = board[b];
        new_board[b] = (board[b] & 0x00FF00FF) | (v2 << 8) | (v1 << 24);
    }
    else if (n-2 == i && 0 == j) {
        int b = 0;
        int e = (i-1)*(n-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = board[b+n-1] & 0x000000FF;
        int v2 = (board[b+n-1] & 0x00FF0000) >> 16;
        new_board[b] = (board[b] & 0xFF00FFFF) | (v2 << 16);
        ++b;
        e = b+n-2;
        for (; b < e; ++b)
            new_board[b] = board[b];
        new_board[b] = (board[b] & 0xFF00FF00) | v2 | (v1 << 16);
        ++b;
        e = (n-1)*(n-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
    }
    else if (0 == i) {
        int b = 0;
        int e = j-1;
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = (board[b] & 0x0000FF00) >> 8;
        int v2 = (board[b] & 0xFF000000) >> 24;
        new_board[b] = (board[b] & 0x00FF00FF) | (v2 << 8) | (v1 << 24);
        ++b;
        new_board[b] = (board[b] & 0xFF00FF00) | v2 | (v1 << 16);
        ++b;
        e = b+n-3;
        for (; b < e; ++b)
            new_board[b] = board[b];
        new_board[b] = (board[b] & 0xFFFF00FF) | (v1 << 8);
        ++b;
        new_board[b] = (board[b] & 0xFFFFFF00) | v1;
        ++b;
        e = (n-1)*(n-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
    }
    else if (n-1 == j) {
        int b = 0;
        int e = (i-1)*(n-1) + (j-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = (board[b+n-1] & 0x0000FF00) >> 8;
        int v2 = (board[b+n-1] & 0xFF000000) >> 24;
        new_board[b] = (board[b] & 0x00FFFFFF) | (v2 << 24);
        ++b;
        new_board[b] = (board[b] & 0xFF00FFFF) | (v2 << 16);
        ++b;
        e = b+n-3;
        for (; b < e; ++b)
            new_board[b] = board[b];
        new_board[b] = (board[b] & 0x00FF00FF) | (v2 << 8) | (v1 << 24);
        ++b;
        new_board[b] = (board[b] & 0xFF00FF00) | v2 | (v1 << 16);
        ++b;
        e = b+n-3;
        for (; b < e; ++b)
            new_board[b] = board[b];
        new_board[b] = (board[b] & 0xFFFF00FF) | (v1 << 8);
        ++b;
        new_board[b] = (board[b] & 0xFFFFFF00) | v1;
        ++b;
        e = (n-1)*(n-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
    }
    else if (0 == j) {
        int b = 0;
        int e = (i-1)*(n-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = board[b+n-1] & 0x000000FF;
        int v2 = (board[b+n-1] & 0x00FF0000) >> 16;
        new_board[b] = (board[b] & 0xFF00FFFF) | (v2 << 16);
        ++b;
        e = b+n-2;
        for (; b < e; ++b)
            new_board[b] = board[b];
        new_board[b] = (board[b] & 0xFF00FF00) | v2 | (v1 << 16);
        ++b;
        e = b+n-2;
        for (; b < e; ++b)
            new_board[b] = board[b];
        new_board[b] = (board[b] & 0xFFFFFF00) | v1;
        ++b;
        e = (n-1)*(n-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
    }
    else if (n-2 == i) {
        int b = 0;
        int e = (i-1)*(n-1) + (j-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = (board[b+n-1] & 0x0000FF00) >> 8;
        int v2 = (board[b+n-1] & 0xFF000000) >> 24;
        new_board[b] = (board[b] & 0x00FFFFFF) | (v2 << 24);
        ++b;
        new_board[b] = (board[b] & 0xFF00FFFF) | (v2 << 16);
        ++b;
        e = b+n-3;
        for (; b < e; ++b)
            new_board[b] = board[b];
        new_board[b] = (board[b] & 0x00FF00FF) | (v2 << 8) | (v1 << 24);
        ++b;
        new_board[b] = (board[b] & 0xFF00FF00) | v2 | (v1 << 16);
        ++b;
        e = (n-1)*(n-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
    }
    else {
        int b = 0;
        int e = (i-1)*(n-1) + (j-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
        int v1 = (board[b+n-1] & 0x0000FF00) >> 8;
        int v2 = (board[b+n-1] & 0xFF000000) >> 24;
        new_board[b] = (board[b] & 0x00FFFFFF) | (v2 << 24);
        ++b;
        new_board[b] = (board[b] & 0xFF00FFFF) | (v2 << 16);
        ++b;
        e = b+n-3;
        for (; b < e; ++b)
            new_board[b] = board[b];
        new_board[b] = (board[b] & 0x00FF00FF) | (v2 << 8) | (v1 << 24);
        ++b;
        new_board[b] = (board[b] & 0xFF00FF00) | v2 | (v1 << 16);
        ++b;
        e = b+n-3;
        for (; b < e; ++b)
            new_board[b] = board[b];
        new_board[b] = (board[b] & 0xFFFF00FF) | (v1 << 8);
        ++b;
        new_board[b] = (board[b] & 0xFFFFFF00) | v1;
        ++b;
        e = (n-1)*(n-1);
        for (; b < e; ++b)
            new_board[b] = board[b];
    }
}



class SquareRemover {
    enum {
        CNT_MOVES  = 10000,
        CNT_RESULT = 30000
    };


    int colors_;
    int start_seed_;
    int n_;
    int size_;
    int max_depth_;
    
    int max_best_score_;
    int best_deep_;

public:

    int short_process(PtrBoard& orig_board, Generator& orig_gen, std::vector<int>& moves, int beg_i, int end_i, int beg_j, int end_j, int cur_i, int cur_j, int cur_dir, int depth, int accumulated_score, int step) {

        int limit = n_ - 1;
        int size_moves = depth;

        // best
        PtrBoard best_board;
        Generator best_gen(orig_gen);
        int best_score = -1;
        std::vector<int> best_moves;

        // cur
        PtrBoard cur_board;
        cur_board.alloc();

        //
        --depth;
        
        for (int i = beg_i; i < end_i; i+=step) {
            for (int j = beg_j; j < end_j; j+=step) {
                if (0 <= i && i < n_ && 0 <= j && j < limit && !(i == cur_i && j == cur_j && 1 == cur_dir)) {
                    Generator cur_gen(orig_gen);
                    std::vector<int> cur_moves;
                    cur_moves.reserve(size_moves);

                    cur_moves.push_back(i);
                    cur_moves.push_back(j);
                    cur_moves.push_back(1);
//cout << "before move right [" << i << ", " << j << "]" << endl;
//print_smart(orig_board.board(), n_);
                    move_right(i, j, orig_board.board(), cur_board.board(), n_);
//validate_board(cur_board.board(), n_, true);
                    int cur_score = 0;
//cout << "before process_board" << endl;
                    cur_score = process_board(cur_board.board(), n_, cur_gen, colors_, size_);
//validate_board(cur_board.board(), n_, true);

                    if (depth > 0 && ((accumulated_score + cur_score) >= max_best_score_ || depth > best_deep_+2)) 
                        cur_score += short_process(cur_board, cur_gen, cur_moves, i-1, i+2, j-1, j+3, i, j, 1, depth, accumulated_score + cur_score, 1);
    
                    if (best_score < cur_score) {
                        best_board = cur_board;
                        cur_board.alloc();
                        best_gen = cur_gen;
                        best_moves.swap(cur_moves);
                        best_score = cur_score;

                        if (accumulated_score + best_score > max_best_score_) {
                            best_deep_ = depth;
                            max_best_score_ = accumulated_score + best_score;
                        }
                    }
                }
                if (0 <= i && i < limit && 0 <= j && j < n_ && !(i == cur_i && j == cur_j && 2 == cur_dir)) {
                    Generator cur_gen(orig_gen);
                    std::vector<int> cur_moves;
                    cur_moves.reserve(size_moves);

                    cur_moves.push_back(i);
                    cur_moves.push_back(j);
                    cur_moves.push_back(2);
//cout << "before move down [" << i << ", " << j << "]" << endl;
//print_smart(orig_board.board(), n_);
                    move_down(i, j, orig_board.board(), cur_board.board(), n_);
//validate_board(cur_board.board(), n_, true);
                    int cur_score = 0;
//cout << "before process_board" << endl;
                    cur_score = process_board(cur_board.board(), n_, cur_gen, colors_, size_);
//validate_board(cur_board.board(), n_, true);
                    if (depth > 0 && ((accumulated_score + cur_score) >= max_best_score_ || depth > best_deep_+2))
                        cur_score += short_process(cur_board, cur_gen, cur_moves, i-1, i+3, j-1, j+2, i, j, 2, depth, accumulated_score + cur_score, 1);

                    if (best_score < cur_score) {
                        best_board = cur_board;
                        cur_board.alloc();
                        best_gen = cur_gen;
                        best_moves.swap(cur_moves);
                        best_score = cur_score;

                        if (accumulated_score + best_score > max_best_score_) {
                            best_deep_ = depth;
                            max_best_score_ = best_score + accumulated_score;
                        }
                    }
                }
            }
        }

        if (best_score == 0) {
        }

        orig_board = best_board;
        orig_gen = best_gen;
        moves.insert(moves.end(), best_moves.begin(), best_moves.end());

        return best_score;
    }

    std::vector<int> playIt(int& colors, std::vector<std::basic_string<char> >& init_board, int& start_seed) {
        // timing
        clock_t t = clock();


        // init
        colors_     = colors;
        start_seed_ = start_seed;
        n_          = init_board.size();
        size_       = (n_-1)*(n_-1);
        max_depth_  = n_ <=6 ? 5 : 4;
        int step = n_ / 2;

        Generator g(colors_, start_seed_);

        std::vector<int> result;
        result.reserve(CNT_RESULT);

        // init board
        PtrBoard board;
        board.alloc();
        fill_from_vector(init_board, board.board());

        int total_score = process_board(board.board(), n_, g, colors_, size_);
        int total_score1,
            total_score2;

        while (result.size() < CNT_RESULT) {

            PtrBoard tmp1; tmp1.alloc(); memcpy(tmp1.board(), board.board(), n_*n_*sizeof(int));
            Generator cur_g1(g);
            std::vector<int> cur_moves1;
            total_score1 = start_one(tmp1, cur_g1, cur_moves1, 0, n_, 0, n_, step); 

            PtrBoard tmp2; tmp2.alloc(); memcpy(tmp2.board(), board.board(), n_*n_*sizeof(int));
            Generator cur_g2(g);
            std::vector<int> cur_moves2;
            total_score2 = start_one(tmp2, cur_g2, cur_moves2, 1, n_, 1, n_, step);

            if (total_score1 > total_score2) {
                result.insert(result.end(), cur_moves1.begin(), cur_moves1.end());
                total_score += total_score1;
                board = tmp1;
                g = cur_g1;
            }
            else {
                result.insert(result.end(), cur_moves2.begin(), cur_moves2.end());
                total_score += total_score2;
                board = tmp2;
                g = cur_g2;
            }
        }

        // end timing
        t = clock() - t;
        cout << "Time: " << (((float)t) / CLOCKS_PER_SEC) << endl;
        cout << "Score: " << total_score << endl;
        cout << "Res: " << result.size() << endl;

        int d = result.size() - 30000;
        while (d > 0) {
            result.pop_back();
            --d;
        }

        //
        return result;
    }

    int start_one(PtrBoard& board, Generator& g, std::vector<int>& cur_moves, int beg_i, int end_i, int beg_j, int end_j, int step) {
        cur_moves.reserve(max_depth_*3);
        max_best_score_ = 0;
        best_deep_ = 0;
        return short_process(board, g, cur_moves, beg_i, end_i, beg_j, end_j, -1, -1, -1, max_depth_, 0, step);
    }

};




int main(int argc, const char* argv[]) {
    std::vector<std::string> v;

    v.push_back("01110301124204");
    v.push_back("23410342140311");
    v.push_back("11221043243240");
    v.push_back("41331040000212");
    v.push_back("30341210240431");
    v.push_back("13102302012121");
    v.push_back("14443123422433");
    v.push_back("11330124421024");
    v.push_back("04111002142314");
    v.push_back("13243424303310");
    v.push_back("32302334141333");
    v.push_back("40010223344143");
    v.push_back("30420121014442");
    v.push_back("02034041313410");

    int colors = 5;
    //int N = 14;
    int startSeed = 857377961;


    SquareRemover sr;
    std::vector<int> result = sr.playIt(colors, v, startSeed);

    cout << "Result: ";
    for (int i = 0; i < 30 && i < result.size(); ++i) {
        cout << result[i] << " ";
    }
    cout << endl;

    Generator g(5, 857377961);
    PtrBoard b;
    b.alloc();
    fill_from_vector(v, b.board());
    int total_score = process_board(b.board(), 14, g, colors, 169);
//    print_smart(b.board(), 14);
//    cout << "---" << endl;
    for (int i = 0; i < result.size(); i+=3) {
        if (result[i] < 0 || 13 < result[i])
            cout << "ERROR: " << i << " " << result[i] << endl;
        if (result[i+1] < 0 || 13 < result[i+1])
            cout << "ERROR: " << (i+1) << " " << result[i+1] << endl;
        //cout << result[i] << " " << result[i+1] << " " << result[i+2] << endl;
        if (result[i+2] == 1) {
            PtrBoard nb;
            nb.alloc();
            move_right(result[i], result[i+1], b.board(), nb.board(), 14);
            memcpy(b.board(), nb.board(), 169*sizeof(int));

            total_score += process_board(b.board(), 14, g, colors, 169);
        }
        else {
            PtrBoard nb;
            nb.alloc();
            move_down(result[i], result[i+1], b.board(), nb.board(), 14);
            memcpy(b.board(), nb.board(), 169*sizeof(int));

            total_score += process_board(b.board(), 14, g, colors, 169);
        }
//        print_smart(b.board(), 14);
//        cout << "---" << endl;
    }
    cout << "Control result: " << total_score << endl;

    return 0;
    
/*
    for (int i = 0; i < 8; ++i)
        cout << v[i] << endl;
    cout << "====================================" << endl;

    UINT board_idx = alloc_board();
    Board& board = mega_buffer[board_idx];

    fill_from_vector(v, board.board, &board.hash1, &board.hash2, &board.hash3, hash_vals1, hash_vals2, hash_vals3);
    //print_board(mega_buffer);
    print_smart(board.board, 8);
    cout << "Hash: " << board.hash1 << " " << board.hash2 << " " << board.hash3 << endl;


    clock_t t = clock();


    UINT new_board_idx = alloc_board();
    Board& new_board = mega_buffer[new_board_idx];
    

    bloom_add(board.hash1, board.hash2, board.hash3, bloom, BLOOM_SIZE);

    move_down(6, 5, board.board, new_board.board, 8);
    new_board.hash1 = board.hash1;
    new_board.hash2 = board.hash2;
    new_board.hash3 = board.hash3;

    update_hash(new_board.board, 6, 5, 7, 5, 8, &new_board.hash1, &new_board.hash2, &new_board.hash3, hash_vals1, hash_vals2, hash_vals3);
    print_smart(new_board.board, 8);
    cout << "New hash: " << new_board.hash1 << " " << new_board.hash2 << " " << new_board.hash3 << endl;

    bool exists = bloom_exists(new_board.hash1, new_board.hash2, new_board.hash3, bloom, BLOOM_SIZE);
    cout << "Found in bloom: [" << exists << "] " << new_board.hash1 << " " << new_board.hash2 << " " << new_board.hash3 << endl;

    move_down(6, 5, new_board.board, board.board, 8);
    board.hash1 = new_board.hash1;
    board.hash2 = new_board.hash2;
    board.hash3 = new_board.hash3;

    cout << "--" << endl;
    print_smart(board.board, 8);

    update_hash(board.board, 6, 5, 7, 5, 8, &board.hash1, &board.hash2, &board.hash3, hash_vals1, hash_vals2, hash_vals3);
    cout << "New hash: " << board.hash1 << " " << board.hash2 << " " << board.hash3 << endl;

    exists = bloom_exists(board.hash1, board.hash2, board.hash3, bloom, BLOOM_SIZE);
    cout << "Found in bloom: [" << exists << "] " << board.hash1 << " " << board.hash2 << " " << board.hash3 << endl;
    
    free_board(new_board_idx);



    ////////////////////
    Generator g(5, 123);
    int score = process_board(board.board, 8, g, 5, 49, &board.hash1, &board.hash2, &board.hash3, hash_vals1, hash_vals2, hash_vals3);
    cout << "Score: " << score << endl;
    print_smart(board.board, 8);

    v.clear();
    v.push_back("30421131");
    v.push_back("40044240");
    v.push_back("01342042");
    v.push_back("41231332");
    v.push_back("21220220");
    v.push_back("40110402");
    v.push_back("04430124");
    v.push_back("23432243");

    new_board_idx = alloc_board();
    Board* ctrl = get_board(new_board_idx);
    fill_from_vector(v, ctrl->board, &ctrl->hash1, &ctrl->hash2, &ctrl->hash3, hash_vals1, hash_vals2, hash_vals3);
    cout << "Repl hash: " << board.hash1 << " " << board.hash2 << " " << board.hash3 << endl;
    cout << "Ctrl hash: " << ctrl->hash1 << " " << ctrl->hash2 << " " << ctrl->hash3 << endl;



    t = clock() - t;
    cout << "Time: " << (((float)t) / CLOCKS_PER_SEC) << endl;
    cout << "=====================================" << endl;
//    print_smart(new_buffer, 8);

    return 0;
*/
}


