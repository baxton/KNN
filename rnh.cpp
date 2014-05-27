
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
516 lines yanked                                                                                                    1,0-1         Top
