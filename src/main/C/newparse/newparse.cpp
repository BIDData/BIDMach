// Read a file and parse each line according to the format file.
// Output integer matrices with lookup maps in either ascii text 
// or matlab binary form. 

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"
#include "gzstream.h"

ivector wcount;
ivector tokens;
unhash unh;
strhash htab;

extern "C" {
extern int yylex(void);
extern FILE*   yyin;

int numlines=0;
}

int checkword(char *str) {
  return checkword(str, htab, wcount, tokens, unh);
}

void addtok(int tok) {
  addtok(tok, tokens);
}

const char usage[] = 
"\nParser for specialized input files. Arguments are\n"
"   -i <infile>     input file[s] to read\n"
"   -o <outd>       output files will be written to <outd><infile>.imat[.gz]\n"
"   -d <dictfile>   dictionary will be written to <dictfile>.sbmat[.gz]\n"
"                   and word counts will be written to <dictfile>.imat[.gz]\n"
"   -s N            set buffer size to N.\n"
"   -c              produce compressed (gzipped) output files.\n\n"
;

int main(int argc, char ** argv) {
  int pos, iarg=1, membuf=1048576;
  char *here;
  char *ifname = NULL;
  string odname="", dictname = "", suffix = "";
  while (iarg < argc) {
    if (strncmp(argv[iarg], "-i", 2) == 0) {
      ifname = argv[++iarg];
    } else if (strncmp(argv[iarg], "-o", 2) == 0) {
      odname = argv[++iarg];
    } else if (strncmp(argv[iarg], "-d", 2) == 0) {
      dictname = argv[++iarg];
    } else if (strncmp(argv[iarg], "-s", 2) == 0) {
      membuf = strtol(argv[++iarg],NULL,10);
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
  if (dictname.size() == 0) dictname = odname+"dict";
  here = strtok(ifname, " ,");
  while (here != NULL) {
    if (strstr(here, ".gz") - here == strlen(here) - 3) {
#if defined __CYGWIN__ || ! defined __GNUC__
      printf("cant use compressed files in cygwin\n");
      exit(1);
#else 
      yyin = popen( (string("gunzip -c ")+here).c_str(), "r" );
#endif
    } else {
      yyin = fopen( here, "r" );
    }
    fprintf(stderr, "\nScanning %s\n", here);
    fflush(stderr);
    yylex();
    if (strstr(here, ".gz") - here == strlen(here) - 3) {
#if ! defined __CYGWIN__ && defined __GNUC__
      pclose(yyin);
#endif
    } else {
      fclose(yyin);
    }
    fprintf(stderr, "\r%05d lines", numlines);
    fflush(stderr);
    string rname = here;
    if (strstr(here, ".gz") - here == strlen(here) - 3) {
      rname = rname.substr(0, strlen(here) - 3);
    } 
    pos = rname.rfind('/');
    if (pos == string::npos) pos = rname.rfind('\\');
    if (pos != string::npos) rname = rname.substr(pos+1, rname.size());
    writeIntVec(tokens, odname+rname+".imat"+suffix, membuf);
    tokens.clear();
    numlines = 0;
    here = strtok(NULL, " ,");
  }
  fprintf(stderr, "\nWriting Dictionary\n");
  writeIntVec(wcount, dictname+".imat"+suffix, membuf);
  writeSBVecs(unh, dictname+".sbmat"+suffix, membuf);
}
