#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"
#include "gzstream.h"

extern int yylex(void);
extern FILE*   yyin;

void indxsort(sfvector & sfv) {
  sort(sfv.begin(), sfv.end(), indxlt<float>);
}

void valusort(sfvector & sfv) {
  sort(sfv.begin(), sfv.end(), valult<float>);
}

void indxsort(sfvector::iterator p1, sfvector::iterator p2) {
  sort(p1, p2, indxlt<float>);
}

void valusort(sfvector::iterator p1, sfvector::iterator p2) {
  sort(p1, p2, valult<float>);
}

void indxsorts(sfvector & sfv) {
  stable_sort(sfv.begin(), sfv.end(), indxlt<float>);
}

void valusorts(sfvector & sfv) {
  stable_sort(sfv.begin(), sfv.end(), valult<float>);
}

void indxsorts(sfvector::iterator p1, sfvector::iterator p2) {
  stable_sort(p1, p2, indxlt<float>);
}

void valusorts(sfvector::iterator p1, sfvector::iterator p2) {
  stable_sort(p1, p2, valult<float>);
}


void strtoupper(char * str) {
  if (str)
    while (*str) {
      *str = toupper(*str);
      str++;
    }
}

void strtolower(char * str) {
  if (str)
    while (*str) {
      *str = tolower(*str);
      str++;
    }
}

int checkword(char * str, strhash &htab, ivector &wcount, ivector &tokens, unhash &unh) {
  strtolower(str);
  int userno;
  if (htab.count(str)) {
    userno = htab[str];
    wcount[userno-1]++;
  } else {
    try {
      char * newstr = new char[strlen(str)+1];
      strcpy(newstr, str);
      wcount.push_back(1);
      unh.push_back(newstr);
      userno = unh.size();
      htab[newstr] = userno;
    } catch (bad_alloc) {
      cerr << "stringIndexer:checkstr: allocation error" << endl;
      throw;
    }
  }
  //  fprintf(stderr, "token %s (%d)\n", str, userno);
  tokens.push_back(userno);
  return userno;
}

void addtok(int tok, ivector &tokens) {
  tokens.push_back(tok | 0x80000000);
}


// Return a unix local time for dates of form MM-DD-YYYY
int parsemdt(char * str) {
  struct tm tnow;
  int day=0, month=1, year=1900;
  char * next;
  time_t tt;
  const char * delims = "-/: ";
  next = strpbrk(str, delims);
  if (next) {
    *next++ = 0;
    sscanf(str, "%d", &month);
    str = next;
    next = strpbrk(str, delims);
    if (next) {
      *next++ = 0;
      sscanf(str, "%d", &day);
      sscanf(next, "%d", &year);
    }
  }
  tnow.tm_year = year-1900;
  tnow.tm_mon = month-1;
  tnow.tm_mday = day;
  tnow.tm_isdst = 0;
  tnow.tm_sec = 0;
  tnow.tm_min = 0;
  tnow.tm_hour = 0;
  tnow.tm_isdst = 0;
  tt = mktime(&tnow);
  tnow.tm_year = 70;   // Jan 2, 1970 in GMT
  tnow.tm_mon = 0;
  tnow.tm_mday = 2;
  tnow.tm_isdst = 0;
  tt = tt - mktime(&tnow) + 24*3600;
  return tt;
}

// For dates of form YYYY-MM-DD
int parsedt(char * str) {
  struct tm tnow;
  int day=0, month=1, year=1900;
  char * next;
  time_t tt;
  const char * delims = "-/: ";
  next = strpbrk(str, delims);
  if (next) {
    *next++ = 0;
    sscanf(str, "%d", &year);
    str = next;
    next = strpbrk(str, delims);
    if (next) {
      *next++ = 0;
      sscanf(str, "%d", &month);
      sscanf(next, "%d", &day);
    }
  }
  tnow.tm_year = year-1900;
  tnow.tm_mon = month-1;
  tnow.tm_mday = day;
  tnow.tm_isdst = 0;
  tnow.tm_sec = 0;
  tnow.tm_min = 0;
  tnow.tm_hour = 0;
  tnow.tm_isdst = 0;
  tt = mktime(&tnow);
  tnow.tm_year = 70;   // Jan 2, 1970 in GMT
  tnow.tm_mon = 0;
  tnow.tm_mday = 2;
  tnow.tm_isdst = 0;
  tt = tt - mktime(&tnow) + 24*3600;
  return tt;
}


// For dates of form MM/DD/YYYY HH:MM:SS or MM/DD/YY HH:MM:SS
// nyear (2 or 4) is the number of year digits
int parsegdate(char * str, int nyear) {
  struct tm tnow;
  int i, fields[6];
  float secs;
  char * next;
  time_t tt;
  const char * delims = "-/: ";
  fields[0]=1900;
  fields[1]=1;
  fields[2]=0;
  fields[3]=0;
  fields[4]=0;
  fields[5]=0;
  next = strpbrk(str, delims);
  for (i = 0; i < 5 && next && *next; i++) {
    *next++ = 0;
    next += strspn(next, delims);
    sscanf(str, "%d", &fields[i]);
    //    printf("%d ", fields[i]);
    str = next;
    next = strpbrk(str, delims);
  }
  if (str)
    sscanf(str, "%f", &secs);
  //  printf("%s %f %d\n", str, secs, (int)secs);
  if (nyear == 4) {
    tnow.tm_year = fields[2]-1900;
  } else {
    tnow.tm_year = fields[2];
  }
  tnow.tm_mon = fields[0]-1;
  tnow.tm_mday = fields[1];
  tnow.tm_hour = fields[3];
  tnow.tm_min = fields[4];
  tnow.tm_sec = (int)secs;
  tnow.tm_isdst = 0;
  tt = mktime(&tnow);
  tnow.tm_year = 70;   // Jan 2, 1970 in GMT
  tnow.tm_mon = 0;
  tnow.tm_mday = 2;
  tnow.tm_hour = 0;
  tnow.tm_min = 0;
  tnow.tm_sec = 0;
  tnow.tm_isdst = 0;
  tt = tt - mktime(&tnow) + 24*3600;
  return tt;
}

// For dates of form MM/DD/YYYY HH:MM:SS 
int parsemdate(char * str) {
  return parsegdate(str, 4);
}

// For dates of form MM/DD/YY HH:MM:SS 
int parsecmdate(char * str) {
  return parsegdate(str, 2);
}

// For dates of form YYYY/MM/DD HH:MM:SS
int parsedate(char * str) {
  struct tm tnow;
  int i, fields[6];
  float secs;
  char * next;
  time_t tt;
  const char * delims = "T-+/: ";
  fields[0]=1900;
  fields[1]=1;
  fields[2]=0;
  fields[3]=0;
  fields[4]=0;
  fields[5]=0;
  next = strpbrk(str, delims);
  for (i = 0; i < 5 && next && *next; i++) {
    *next++ = 0;
    next += strspn(next, delims);
    sscanf(str, "%d", &fields[i]);
    str = next;
    next = strpbrk(str, delims);
  }
  if (str)
    sscanf(str, "%f", &secs);
  tnow.tm_year = fields[0]-1900;
  tnow.tm_mon = fields[1]-1;
  tnow.tm_mday = fields[2];
  tnow.tm_hour = fields[3];
  tnow.tm_min = fields[4];
  tnow.tm_sec = (int)secs;
  tnow.tm_isdst = 0;
  tt = mktime(&tnow);
  tnow.tm_year = 100;   // 
  tnow.tm_mon = 0;
  tnow.tm_mday = 1;
  tnow.tm_hour = 0;
  tnow.tm_min = 0;
  tnow.tm_sec = 0;
  tnow.tm_isdst = 0;
  tt = tt - mktime(&tnow) + 24*3600;
  return tt;
}


istream * open_in(string ifname) {
  istream * ifstr = NULL;
  int opened = 0;
  if (ifname.rfind(".gz") == ifname.length() - 3) {
    ifstr = new igzstream(ifname.c_str(), ios_base::in);
    opened = ((igzstream *)ifstr)->rdbuf()->is_open(); 
  }
  else {
    ifstr = new ifstream(ifname.c_str());
    opened = ((ifstream *)ifstr)->is_open(); 
  }
  if (!*ifstr || !opened) {
    cerr << "Couldnt open input file " << ifname << endl;
    throw;
  }
  return ifstr;
}

ostream * open_out(string ofname) {
  ostream * ofstr = NULL;
  int opened = 0;
  if (ofname.rfind(".gz") == ofname.length() - 3) {
    ofstr = new ogzstream(ofname.c_str());
    opened = ((ogzstream *)ofstr)->rdbuf()->is_open(); 
  } else {
    ofstr = new ofstream(ofname.c_str(), std::ofstream::binary);
    opened = ((ofstream *)ofstr)->is_open(); 
  }
  if (!*ofstr || !opened) {
    cerr << "Couldnt open output file " << ofname << endl;
    throw;
  }
  return ofstr;
}

istream * open_in_buf(string ifname, char * buffer, int buffsize) {
  istream * ifstr = NULL;
  int opened = 0;
  if (ifname.rfind(".gz") == ifname.length() - 3) {
    ifstr = new igzstream(ifname.c_str(), ios_base::in);
    opened = ((igzstream *)ifstr)->rdbuf()->is_open(); 
    ((igzstream *)ifstr)->rdbuf()->pubsetbuf(buffer, buffsize);
  }
  else {
    ifstr = new ifstream(ifname.c_str());
    opened = ((ifstream *)ifstr)->is_open(); 
    ((ifstream *)ifstr)->rdbuf()->pubsetbuf(buffer, buffsize);
  }
  if (!*ifstr || !opened) {
    cerr << "Couldnt open input file " << ifname << endl;
    throw;
  }
  return ifstr;
}

ostream * open_out_buf(string ofname, int buffsize) {
  char *buffer = new char[buffsize];
  ostream * ofstr = NULL;
  int opened = 0;
  if (ofname.rfind(".gz") == ofname.length() - 3) {
    ofstr = new ogzstream(ofname.c_str(), ios_base::out);
    opened = ((ogzstream *)ofstr)->rdbuf()->is_open(); 
    ((ogzstream *)ofstr)->rdbuf()->pubsetbuf(buffer, buffsize);
  } else {
    ofstr = new ofstream(ofname.c_str(), std::ofstream::binary);
    opened = ((ofstream *)ofstr)->is_open(); 
#ifdef __GNUC__
    ((ofstream *)ofstr)->rdbuf()->pubsetbuf(buffer, buffsize);
#endif
  }
  if (!*ofstr || !opened) {
    cerr << "Couldnt open output file " << ofname << endl;
    throw;
  }
  delete [] buffer;
  return ofstr;
}

void closeos(ostream *ofs) {
  ofstream * of1 = dynamic_cast<ofstream *>(ofs);
  if (of1) of1->close();
  ogzstream * of2 = dynamic_cast<ogzstream *>(ofs);
  if (of2) of2->close();
}

int writeIntVec(ivector & im, string fname, int buffsize) {
  int fmt, nrows, ncols, nnz;

  ostream *ofstr = open_out_buf(fname.c_str(), buffsize);
  fmt = 110;
  nrows = im.size();
  ncols = 1;
  nnz = nrows;
  ofstr->write((const char *)&fmt, 4);
  ofstr->write((const char *)&nrows, 4);
  ofstr->write((const char *)&ncols, 4);
  ofstr->write((const char *)&nnz, 4);
  ofstr->write((const char *)&im[0], 4 * nrows);
  closeos(ofstr);
  return 0;
}

int writeDIntVec(divector & im, string fname, int buffsize) {
  int fmt, nrows, ncols, nnz;
  ostream *ofstr = open_out_buf(fname.c_str(), buffsize);
  fmt = 120;
  nrows = im.size();
  ncols = 1;
  nnz = nrows;
  ofstr->write((const char *)&fmt, 4);
  ofstr->write((const char *)&nrows, 4);
  ofstr->write((const char *)&ncols, 4);
  ofstr->write((const char *)&nnz, 4);
  ofstr->write((const char *)&im[0], 8 * nrows);
  closeos(ofstr);
  return 0;
}

int writeQIntVecx(qvector & im, string fname, int buffsize) {
  int fmt, nrows, ncols, nnz;
  ostream *ofstr = open_out_buf(fname.c_str(), buffsize);
  fmt = 170;
  nrows = im.size();
  ncols = 1;
  nnz = nrows;
  ofstr->write((const char *)&fmt, 4);
  ofstr->write((const char *)&nrows, 4);
  ofstr->write((const char *)&ncols, 4);
  ofstr->write((const char *)&nnz, 4);
  ofstr->write((const char *)&im[0], 16 * nrows);
  closeos(ofstr);
  return 0;
}


int writeQIntVec(qvector & qv, string fname, int buffsize) {
  int j, nrows, fmt, ncols, nnz;
  uint64 v;
  ostream *ofstr = open_out_buf(fname, buffsize);
  ncols = qv.size();
  nrows = 4;
  nnz = nrows * ncols;
  fmt = 110;
  ofstr->write((const char *)&fmt, 4);
  ofstr->write((const char *)&nrows, 4);
  ofstr->write((const char *)&ncols, 4);
  ofstr->write((const char *)&nnz, 4);
  for (j = 0; j < qv.size(); j++) {
    v = qv[j].bottom;
    ofstr->write((const char *)&v, 8);
    v = qv[j].top;
    ofstr->write((const char *)&v, 8);
  }
  closeos(ofstr);
  return 0;
}


int writeFVec(fvector & im, string fname, int buffsize) {
  int fmt, nrows, ncols, nnz;
  ostream *ofstr = open_out_buf(fname.c_str(), buffsize);
  fmt = 130;
  nrows = im.size();
  ncols = 1;
  nnz = nrows;
  ofstr->write((const char *)&fmt, 4);
  ofstr->write((const char *)&nrows, 4);
  ofstr->write((const char *)&ncols, 4);
  ofstr->write((const char *)&nnz, 4);
  ofstr->write((const char *)&im[0], 4 * nrows);
  closeos(ofstr);
  return 0;
}

int writeDVec(dvector & im, string fname, int buffsize) {
  int fmt, nrows, ncols, nnz;
  ostream *ofstr = open_out_buf(fname.c_str(), buffsize);
  fmt = 140;
  nrows = im.size();
  ncols = 1;
  nnz = nrows;
  ofstr->write((const char *)&fmt, 4);
  ofstr->write((const char *)&nrows, 4);
  ofstr->write((const char *)&ncols, 4);
  ofstr->write((const char *)&nnz, 4);
  ofstr->write((const char *)&im[0], 8 * nrows);
  closeos(ofstr);
  return 0;
}

int writeDIntVecs(dimatrix & im, string fname, int buffsize) {
  int i, j, nrows, fmt, ncols, nnz;
  int64 v;
  ostream *ofstr = open_out_buf(fname.c_str(), buffsize);
  ncols = 2;
  nnz = 0;
  for (i = 0, nrows = 0; i < im.size(); i++) nrows += im[i].size();
  fmt = 110;
  ofstr->write((const char *)&fmt, 4);
  ofstr->write((const char *)&nrows, 4);
  ofstr->write((const char *)&ncols, 4);
  ofstr->write((const char *)&nnz, 4);
  for (i = 0; i < im.size(); i++) {
    for (j = 0; j < im[i].size(); j++) {
      ofstr->write((const char *)&i, 4);
    }
  }
  for (i = 0; i < im.size(); i++) {
    divector & ivv = im[i];
    for (j = 0; j < ivv.size(); j++) {
      v = ivv[j];
      ofstr->write((const char *)&v, 8);
    }
  }
  closeos(ofstr);
  return 0;
}


int writeIntVecs(imatrix & im, string fname, int buffsize) {
  int i, j, nrows, fmt, ncols, nnz, v;
  ostream *ofstr = open_out_buf(fname.c_str(), buffsize);
  ncols = 2;
  nnz = 0;
  for (i = 0, nrows = 0; i < im.size(); i++) nrows += im[i].size();
  fmt = 110;
  ofstr->write((const char *)&fmt, 4);
  ofstr->write((const char *)&nrows, 4);
  ofstr->write((const char *)&ncols, 4);
  ofstr->write((const char *)&nnz, 4);
  for (i = 0; i < im.size(); i++) {
    for (j = 0; j < im[i].size(); j++) {
      ofstr->write((const char *)&i, 4);
    }
  }
  for (i = 0; i < im.size(); i++) {
    ivector & ivv = im[i];
    for (j = 0; j < ivv.size(); j++) {
      v = ivv[j];
      ofstr->write((const char *)&v, 4);
    }
  }
  closeos(ofstr);
  return 0;
}


int writeSBVecs(unhash & unh, string fname, int buffsize) {
  int i, s, fmt, nrows, ncols, nnz;
  ostream *ofstr = open_out_buf(fname.c_str(), buffsize);
  fmt = 301; // 3=sparse(no rows), 0=byte, 1=int
  ncols = unh.size();
  ivector cols;
  cols.push_back(0);
  for (i=0, nrows=0, nnz=0; i<ncols; i++) {
    s = strlen(unh[i]);
    nrows = max(nrows, s);
    nnz = nnz + s;
    cols.push_back(nnz);
  }
  ofstr->write((const char *)&fmt, 4);
  ofstr->write((const char *)&nrows, 4);
  ofstr->write((const char *)&ncols, 4);
  ofstr->write((const char *)&nnz, 4);
  ofstr->write((const char *)&cols[0], 4 * (ncols+1));
  for (i=0; i<ncols; i++) {
    ofstr->write(unh[i], cols[i+1] - cols[i]);
  }
  closeos(ofstr);
  return 0;
}

