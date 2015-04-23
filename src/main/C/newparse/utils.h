// Read a file and parse each line according to the format file.
// Output integer matrices with lookup maps in either ascii text 
// or matlab binary form. 

#include <assert.h>
#include <math.h>
#include <string>
#include <iostream> 
#include <iomanip>
#include <fstream> 
#include <sstream> 
#include <vector> 
#include <algorithm>
#include <stdexcept>
#include <time.h>
#include <cstring>
#include "gzstream.h"

#ifdef __GNUC__
#include <stdint.h>
typedef uint64_t uint64;
typedef int64_t int64;
typedef int32_t int32;
#else
typedef unsigned __int64 uint64;
typedef __int64 int64;
typedef __int32 int32;
#endif

struct eqstr {
  bool operator()(const char* s1, const char* s2) const {
    return strcmp(s1, s2) == 0;
  }
};

struct ltstr {
  bool operator()(const char* s1, const char* s2) const {
    return strcmp(s1, s2) == 0;
  }
};

struct strhashfn {
  size_t operator()(const char* str) const {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++)) {
      hash = ((hash << 5) + hash) + c;
    }
    return hash;
  }
};

class hashcompare {
 public :
  static const size_t bucket_size = 4;
  static const size_t min_buckets = 1 << 10;

  hashcompare() {}

  hashcompare(size_t t) {}

  size_t operator()(const char* str) const {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++)) {
      hash = ((hash << 5) + hash) + c;
    }
    return hash;
  }
  
  bool operator()(const char *left, const char *right) const
  {
    return (strcmp(left, right) < 0);
  }
};

#if defined __GNUC__ 
#if (__GNUC__ < 4 || __GNUC_MINOR__ < 4)
#include <ext/hash_map>
typedef __gnu_cxx::hash_map<const char*, int, __gnu_cxx::hash<const char*>, eqstr> strhash;
#else
#include <unordered_map>
typedef std::unordered_map<const char*, int, strhashfn, eqstr> strhash;
#endif
#else // no __GNUC__
#if _MSC_VER >= 1600
#include <unordered_map>
typedef std::unordered_map<const char*, int, strhashfn, eqstr> strhash;
#else // _MSC_VER < 1600
#include <hash_map>
typedef stdext::hash_map<const char*, int, hashcompare> strhash;
#endif
#endif

typedef struct quadint {
  uint64 top;
  uint64 bottom;
} qint;


template <class T>
class vpair {
public:
  int ind;
  T val;
  vpair(int i, T v): ind(i), val(v) {};
  vpair(): ind(0), val(0) {};
  ~vpair() {};
};

template <class T>
bool indxlt(const vpair<T> & v1, const vpair<T> & v2) {
  return (v1.ind < v2.ind);
}

template <class T>
bool valult(const vpair<T> & v1, const vpair<T> & v2) {
  return (v1.val < v2.val);
}

using namespace std;

typedef vector<char *> unhash;
typedef vpair<float> fpair;
typedef vpair<int> ipair;
typedef vector<fpair> sfvector;
typedef vector<ipair> ipvector;
typedef vector<int> ivector;
typedef vector<int64> divector;
typedef vector<qint> qvector;
typedef vector<ivector> imatrix;
typedef vector<divector> dimatrix;
typedef vector<float> fvector;
typedef vector<double> dvector;
typedef vector<string> svector;
typedef vector<fvector> fmatrix;

void strtoupper(char * str);

void strtolower(char * str);

int checkword(char * str, strhash &htab, ivector &wcount, ivector &tokens, unhash &unh);

int parsedt(char * str);

int parsemdt(char * str);

int parsemdate(char * str);

int parsecmdate(char * str);


istream * open_in(string ifname);

ostream * open_out(string ofname, int level);

istream * open_in_buf(string ifname, char * buffer, int buffsize);

ostream * open_out_buf(string ofname, int buffsize);

void closeos(ostream *ofs);

int writeIntVec(ivector & im, string fname, int buffsize);

int writeDIntVec(divector & im, string fname, int buffsize);

int writeQIntVec(qvector & im, string fname, int buffsize);

int writeFVec(fvector & im, string fname, int buffsize);

int writeDVec(dvector & im, string fname, int buffsize);

int writeIntVecs(imatrix & im, string fname, int buffsize);

int writeDIntVecs(dimatrix & im, string fname, int buffsize);

int writeSBVecs(unhash & unh, string fname, int buffsize);

void addtok(int tok, ivector &tokens);

extern "C" {

void addtok(int tok);

int checkword(char * str);

int parsedate(char * str);

}

