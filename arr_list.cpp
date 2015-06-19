


#include <iostream>
#include <memory>


using namespace std;


const int NOVAL = -1;


template<class T>
class ArrList {

    

    struct node {
        T data;
        int prev;
        int next;

        node() :
            data(),
            prev(NOVAL),
            next(NOVAL)
        {}
    };


    int size_;
    auto_ptr<node> storage_;
    int count_;
    int head_;
    int tail_;

    ArrList(const ArrList&);
    ArrList& operator= (ArrList&);

    
    //
    // private utility methods
    // here I _do_not_ check parameters
    //

    // this method should be used along with connect_to_tail
    void disconnect(int idx) {

        if (!count_)
            return;
    
        int prev = storage_.get()[idx].prev;
        if (NOVAL != prev) {
            storage_.get()[prev].next = storage_.get()[idx].next;
            if (tail_ == idx) 
                tail_ = prev;
        }

        int next = storage_.get()[idx].next;
        if (NOVAL != next) {
            storage_.get()[next].prev = storage_.get()[idx].prev;
            if (head_ == idx)
                head_ = next;
        }


        --count_;

        storage_.get()[idx].prev = NOVAL;
        storage_.get()[idx].next = NOVAL;
    }

    // this method should be used alongwith disconnect
    void connect_to_tail(int idx) {

        if (!count_) {
            // empty list
            head_ = idx;
        }
        else {
            storage_.get()[tail_].next = idx;
            storage_.get()[idx].prev = tail_;
        }

        tail_ = idx;
        ++count_;
    }


    void touch(int idx) {
        disconnect(idx);
        connect_to_tail(idx);
    }

public:

    ArrList(int size) :
        size_(size),
        storage_(new node[size_]),
        count_(0),
        head_(NOVAL),
        tail_(NOVAL)
    {}

    ~ArrList(){}

    int count() const {return count_;}

    template<class K>
    bool get(const K& key, const T** data) {
        int p = head_;

        while (NOVAL != p) {
            if (storage_.get()[p].data.key == key) {
                *data = &storage_.get()[p].data;
                touch(p);
                break;
            }

            p = storage_.get()[p].next;
        }

        return p != NOVAL;
    }

    void add(T& data) {
        if (count_ == size_) {
            storage_.get()[head_].data = data;
            touch(head_);
        }
        else {
            // not fully filled
            storage_.get()[count_].data = data;
            connect_to_tail(count_);
        }
    }

    ostream& print(ostream& os) const {
        int p = head_;

        while (NOVAL != p) {
            os << "(" << storage_.get()[p].data << ")" << (NOVAL != storage_.get()[p].next ? "->" : "");
            p = storage_.get()[p].next;
        }

        return os;
    }

};

template<class T>
ostream& operator<< (ostream& os, const ArrList<T>& l) {
    return l.print(os);
}








struct DATA {
    int key;
    int a;
};

ostream& operator<< (ostream& os, const DATA& d) {
    os << d.key << ":" << d.a;
    return os;
}


int main(int argc, const char* argv[]) {

    ArrList<DATA> hm(5);

    DATA d;

    for (int i = 0; i < 10; ++i) {
        d.key = i;
        d.a = i * 3;
        hm.add(d);

        cout << hm << endl;
    }

    cout << endl;

    for (int i = 7; i < 10; ++i) {
        const DATA* pd = NULL;
        hm.get(i, &pd);
        cout << hm << endl;
    }
        
    cout << endl;

    for (int i = 0; i < 15; ++i) {
        const DATA* pd = NULL;
        if (hm.get(i, &pd))
            cout << i << ": found (" << *pd << ")" << endl;
        else
            cout << i << ": not found" << endl;
    }

    return 0;
}
