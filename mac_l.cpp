

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <iomanip>

using namespace std;





//
//
//

union MAC_ADDR {

    unsigned char mac[6];

    struct {
        unsigned char m1;
        unsigned char m2;
        unsigned char m3;
        unsigned char m4;
        unsigned char m5;
        unsigned char m6;
    } addr;

    void test () {};
};

ostream& operator<< (ostream& os, const MAC_ADDR& mac) {
    os << "MAC: ";

    for (int i = 0; i < 6; ++i)
        os << setw(2) << setfill('0') << hex << (int)mac.mac[i] << (i < 5 ? ":" : "");

    return os;
}

void copy_mac(MAC_ADDR& dst, const MAC_ADDR& src) {
    memcpy(&dst, &src, sizeof(MAC_ADDR));
}

bool equal_mac(const MAC_ADDR& mac1, const MAC_ADDR& mac2) {
    return mac1.addr.m1 == mac2.addr.m1 &&
           mac1.addr.m2 == mac2.addr.m2 &&
           mac1.addr.m3 == mac2.addr.m3 &&
           mac1.addr.m4 == mac2.addr.m4 &&
           mac1.addr.m5 == mac2.addr.m5 &&
           mac1.addr.m6 == mac2.addr.m6;
}



//
// MAC table entry
//

struct MAC_TABLE_ENTRY {
    MAC_ADDR mac;
    size_t last_access;
};






//
// tries to read 6 bytes separated by colon, example: "10:a3:fe:8b:a7:2c"
//
bool parse_mac(const char* in_str, MAC_ADDR* mac) {

    enum {
        FIRST_DIG,
        SECOND_DIG,
        COLON,
        DONE,
    } state = FIRST_DIG;

    bool result = true;

    MAC_ADDR mac_local;
    int count = 0;
    unsigned char tmp = 0;

    const unsigned char* p = (const unsigned char*) in_str;

    while (*p && result && state != DONE) {
        switch (state) {
        case FIRST_DIG:
            if ('0' <= *p && *p <= '9') {
                tmp = (*p - 0x30) << 4;
            }
            else if ('a' <= *p && *p <= 'f') {
                tmp = (*p - 'a' + 0x0A) << 4;
            }
            else if ('A' <= *p && *p <= 'F') {
                tmp = (*p - 'A' + 0x0A) << 4;
            }
            else {
                result = false;
            }
            state = SECOND_DIG;
            break;

        case SECOND_DIG:
            if ('0' <= *p && *p <= '9') {
                tmp |= (*p - 0x30);
            }
            else if ('a' <= *p && *p <= 'f') {
                tmp |= (*p - 'a' + 0x0A);
            }
            else if ('A' <= *p && *p <= 'F') {
                tmp |= (*p - 'A' + 0x0A);
            }
            else {
                result = false;
            }
            mac_local.mac[count++] = tmp;
            if (6 == count) {
                state = DONE;
            }
            else {
                state = COLON;
            }
            break;

        case COLON:
            if (':' == *p) {
                state = FIRST_DIG;
            }
            else {
                result = false;
            }
            break;

        default:
            throw runtime_error("Invalid MAC parser state");
        }

        ++p;
    }

    if (DONE == state)
        copy_mac(*mac, mac_local);
    else
        result = false;

    return result;
} 


//
// Utility API function for parsing input string
// Example of input:
// 1 10:a3:fe:8b:a7:2c 08:6e:90:55:3a:97
// 2 08:6e:90:55:3a:97 10:a3:fe:8b:a7:2c
// 213 10:a3:fe:8b:a7:2c 08:6e:90:55:3a:97
//
// Notes: as it's an API I _do_not_ check parameters here, like checking for NULLs
//        the parser only checks input format.
//        any validation will be done by the caller
//
// In case of any error on parsing, the function throws exception
//
void parse_input(const char* in_str, int* port, MAC_ADDR* mac_s, MAC_ADDR* mac_d) {

    enum {
        WAIT_PORT,
        PORT,
        MAC_DST,
        MAC_SRC,
        DONE
    } state = PORT;


    const char* p = in_str;
    int port_local = 0;
    MAC_ADDR mac_dst;
    MAC_ADDR mac_src;

    while (*p && DONE != state) {
        switch (state) {
        case WAIT_PORT:
            if (isdigit(*p)) {
                port_local = *p - 0x30;
            }
            else {
                throw runtime_error("Invalid port format");
            }
            break;

        case PORT:
            if (isdigit(*p)) {
                port_local = port_local * 10 + (*p - 0x30);
            }
            else if (' ' == *p) {
                state = MAC_DST;
            }
            else {
                throw runtime_error("Invalid port format");
            }
            break;

        case MAC_DST:
            if (!parse_mac(p, &mac_dst)) 
                throw runtime_error("Invalid destination MAC format");
            state = MAC_SRC;
            p += 17;
            break;

        case MAC_SRC:
            if (!parse_mac(p, &mac_src))
                throw runtime_error("Invalid source MAC format");
            state = DONE;
            p += 17;
            break;

        default:
            throw runtime_error("Unexpected input parser state");
        }

        ++p;
    }

    if (DONE != state) {
        throw runtime_error("Invalid input string format - cannot parse");
    }


    // now I can exception-safely
    // change whatever passed by the caller
    *port = port_local;
    copy_mac(*mac_s, mac_src);
    copy_mac(*mac_d, mac_dst);
}






// ---------------------------------------------------------------
// TESTING
// ---------------------------------------------------------------



void test_parsing_input() {
    const char* in_str = "213 10:a3:fe:8b:a7:2c 08:6e:90:55:3a:97";

    int port;
    MAC_ADDR mac_src;
    MAC_ADDR mac_dst;

    cout << "Parsing: " << in_str << endl;

    parse_input(in_str, &port, &mac_src, &mac_dst);

    cout << port << endl
         << mac_src << endl
         << mac_dst << endl;
    
}



// ---------------------------------------------------------------
// MAC Learning
// ---------------------------------------------------------------


void :wq



int main(int argc, const char* argv[]) {

    vector<MAC_TABLE_ENTRY> mac_table;


    try {
        
        test_parsing_input();


    }
    catch (const exception& ex) {
        cout << "ERROR: " << ex.what() << endl;
    }
    catch (...) {
        cout << "ERROR: unexpected" << endl;
    }


    return 0;
}
