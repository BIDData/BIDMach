// Basic VW reader. Not quite fully functional. Does namespaces but not weights or tags.

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits>
#include "utils.h"
#include "gzstream.h"

#define BUFSIZE 1048576

class stringIndexer {
 public:
  ivector count;
  unhash unh;
  strhash htab;
  int size;
  char * linebuf;
  stringIndexer() : htab(0), unh(0), count(0), size(0) {
    linebuf = new char[BUFSIZE];
  };
  stringIndexer(const stringIndexer & si);
  ~stringIndexer();
  stringIndexer(char *);
  int writeMap(string fname, string suffix) {
    writeIntVec(count, fname + ".cnt.imat" + suffix, BUFSIZE);
    return writeSBVecs(unh, fname + ".sbmat" + suffix, BUFSIZE);
  }
  int checkword(char *str);
};


int stringIndexer::
checkword(char * str) {
  int userno;
  char * newstr;
  if (htab.count(str)) {
    userno = htab[str];
    count[userno-1]++;
  } else {
    try {
      newstr = new char[strlen(str)+1];
      strcpy(newstr, str);
      userno = ++size;
      htab[newstr] = userno;
      count.push_back(1);
      unh.push_back(newstr);
    } catch (std::bad_alloc) {
      cerr << "stringIndexer:checkstr: allocation error" << endl;
      throw;
    }
  }
  return userno;
}

stringIndexer::
~stringIndexer() {
  int i;
  for (i=0; i<unh.size(); i++) {
    if (unh[i]) {
      delete [] unh[i];
      unh[i] = NULL;
    }
  }
  if (linebuf) {
    delete [] linebuf;
    linebuf = NULL;
  }
}


int parseLine(char * line, const char * delim1, const char * delim2, const char *delim3,
              const char * delim4, stringIndexer &si, stringIndexer &ns, fvector &labels, imatrix &imat, ivector &nsv, fvector &fvec)  {
  char *here, *next, *fpos;
  int label, indx;
  float fval;
  ivector iv;

  here = line;
  next = strpbrk(here, delim1);     // get the label(s)
  *(next++) = 0;
  sscanf(here, "%d", &label);
  here = next;
  labels.push_back(label);

  next = strpbrk(here, delim2);     // get the tag
  *(next++) = 0;                   
  // Do nothing with the tag for now
  here = next;                      // points just after the "|"
  int nsi = 0;
  if (*here != *delim3) {           // first char is a non-space, must be namespace id
    next = strpbrk(here, delim3);   // go to space after the namespace id
    *(next++) = 0;                  // skip that space and zero it
    nsi = ns.checkword(here);       // lookup the NS id. 
    here = next;
  }
  while (here != NULL) {
    next = strpbrk(here, delim3);   // get the next non-space blob
    if (next) {
      *(next++) = 0;
    }
    // Actually parse in here
    if (*here == *delim2) {         // starts with "|", namespace change
      nsi = ns.checkword(here+1);
      here = next;            
    } else {                        // feature/value pair
      fpos = strpbrk(here, delim4); // fget a value, if there is one
      fval = 1.0;
      if (fpos) {
        sscanf(fpos+1, "%f", &fval);
        *fpos = 0;
      }
      indx = si.checkword(here);    // lookup the feature name
      iv.push_back(indx-1);
      if (nsi > 0) nsv.push_back(nsi-1);
      fvec.push_back(fval);
      here = next;
    }
  }
  imat.push_back(iv);
  return 0;
}


void writefiles(string fname, imatrix &imat, fvector &fvec, fvector &labels, ivector &nsv, string suffix, int &ifile, int membuf) {
  char num[6];
  sprintf(num, "%05d", ifile);
  writeIntVecs(imat, fname+"inds"+num+".imat"+suffix, membuf);
  writeFVec(fvec, fname+"vals"+num+".fmat"+suffix, membuf);
  writeFVec(labels, fname+"labels"+num+".fmat"+suffix, membuf);
  if (nsv.size() > 0) {
    writeIntVec(nsv, fname+"namespace"+num+".imat"+suffix, membuf);
  }
  ifile++;
  imat.clear();
  fvec.clear();
  labels.clear();
  nsv.clear();
}

const char usage[] = 
"\nParser for generalized libsvm files. Arguments are\n"
"   -i <infile>     input file[s] to read\n"
"   -o <outd>       output files will be written to <outd><infile>.imat[.gz]\n"
"   -d <dictfile>   dictionary will be written to <dictfile>.sbmat[.gz]\n"
"                   and word counts will be written to <dictfile>.imat[.gz]\n"
"   -s N            set buffer size to N.\n"
"   -n N            split files at N lines.\n"
"   -c              produce compressed (gzipped) output files.\n\n"
;

int main(int argc, char ** argv) {
  int jmax, iarg=1, membuf=1048576, nsplit=100000, nfiles = 0;
  long long numlines;
  char *here, *linebuf, *readbuf;
  char *ifname = NULL;
  string odname="", dictname = "", suffix = "";
  string delim1=" ", delim2="|", delim3=" ", delim4=":";
  while (iarg < argc) {
    if (strncmp(argv[iarg], "-i", 2) == 0) {
      ifname = argv[++iarg];
    } else if (strncmp(argv[iarg], "-o", 2) == 0) {
      odname = argv[++iarg];
    } else if (strncmp(argv[iarg], "-d", 2) == 0) {
      dictname = argv[++iarg];
    } else if (strncmp(argv[iarg], "-s", 2) == 0) {
      membuf = strtol(argv[++iarg],NULL,10);
    } else if (strncmp(argv[iarg], "-n", 2) == 0) {
      nsplit = strtol(argv[++iarg],NULL,10);
    } else if (strncmp(argv[iarg], "-c", 2) == 0) {
      suffix=".gz";
    } else if (strncmp(argv[iarg], "-?", 2) == 0) {
      printf("%s", usage);
      return 1;
    } else if (strncmp(argv[iarg], "-h", 2) == 0) {
      printf("%s", usage);
      return 1;
    } else {
      cout << "Unknown option " << argv[iarg] << endl;
      exit(1);
    }
    iarg++;
  }
  imatrix imat;
  fvector fvec;
  fvector labels;
  ivector nsv;
  istream * ifstr;
  stringIndexer si; // for tokens
  stringIndexer ns; // for namespaces
  linebuf = new char[membuf];
  readbuf = new char[membuf];
  if (dictname.size() == 0) dictname = odname+"dict";
  here = strtok(ifname, " ,");

  while (here != NULL) {
    ifstr = open_in_buf(ifname, readbuf, membuf);
    numlines = 0;
    while (!ifstr->bad() && !ifstr->eof()) {
      ifstr->getline(linebuf, membuf-1);
      linebuf[membuf-1] = 0;
      if (ifstr->fail()) {
        ifstr->clear();
        ifstr->ignore(std::numeric_limits<long>::max(),'\n');
      }
      if (strlen(linebuf) > 0) {
        jmax++;
        numlines++;
        try {
          parseLine(linebuf, delim1.c_str(), delim2.c_str(), delim3.c_str(), delim4.c_str(),
                    si, ns, labels, imat, nsv, fvec);
        } catch (int e) {
          cerr << "Continuing" << endl;
        }
      }
      if ((numlines % 100000) == 0) {
        cout<<"\r"<<numlines<<" lines processed";
        cout.flush();
      }
      if ((numlines % nsplit) == 0) {
        writefiles(odname, imat, fvec, labels, nsv, suffix, nfiles, membuf);
      }
    }
    if ((numlines % nsplit) != 0) {
      writefiles(odname, imat, fvec, labels, nsv, suffix, nfiles, membuf);
    }
    if (ifstr) delete ifstr;
    cout<<"\r"<<numlines<<" lines processed";
    cout.flush();
    string rname = here;
    if (strstr(here, ".gz") - here == strlen(here) - 3) {
      rname = rname.substr(0, strlen(here) - 3);
    }
    here = strtok(NULL, " ,");
  }
  fprintf(stderr, "\nWriting Dictionary\n");
  si.writeMap(dictname, suffix);
  if (ns.size > 0) writeSBVecs(ns.unh, dictname + "_namespace.sbmat" + suffix, BUFSIZE);
}
