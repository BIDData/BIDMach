// Read a file and parse each line according to the format file.
// Output integer matrices with lookup maps in either ascii text 
// or matlab binary form. 

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"
#include "gzstream.h"

extern int yylex(void);
extern FILE*   yyin;

ivector wcount;
ivector tokens;
unhash unh;
strhash htab;
int numlines=0;

int checkword(char *str) {
  return checkword(str, htab, wcount, tokens, unh);
}

void addtok(int tok) {
   addtok(tok, tokens);
}

int main(int argc, char ** argv) {
  int iarg=1, membuf=1048576;
  char *here;
  char *ifname = NULL;
  string odname="", dictname = "";
  while (iarg < argc) {
    if (strncmp(argv[iarg], "-i", 2) == 0) {
      ifname = argv[++iarg];
    } else if (strncmp(argv[iarg], "-o", 2) == 0) {
      odname = argv[++iarg];
    } else if (strncmp(argv[iarg], "-d", 2) == 0) {
      dictname = argv[++iarg];
    } else if (strncmp(argv[iarg], "-s", 2) == 0) {
      membuf = strtol(argv[++iarg],NULL,10);
    } else {
      cout << "Unknown option " << argv[iarg] << endl;
      exit(1);
    }
    iarg++;
  }
  if (dictname.size() == 0) dictname = odname;
  here = strtok(ifname, " ,");
  while (here != NULL) {
    if (strstr(here, ".gz") - here == strlen(here) - 3) {
#ifdef __CYGWIN__
      printf("cant use compressed files in cygwin\n");
      exit(1);
#else 
      yyin = popen( (string("gunzip -c ")+here).c_str(), "r" );
#endif
    } else {
      yyin = fopen( here, "r" );
    }
    fprintf(stderr, "\nScanning %s\n", here);
    yylex();
    if (strstr(here, ".gz") - here == strlen(here) - 3) {
#ifndef __CYGWIN__
      pclose(yyin);
#endif
    } else {
      fclose(yyin);
    }
    fprintf(stderr, "\r%05d lines", numlines);
    writeIntVec(tokens, odname+here, membuf);
    tokens.clear();
    numlines = 0;
    here = strtok(NULL, " ,");
  }
  fprintf(stderr, "\nWriting Dictionary\n");
  writeIntVec(wcount, dictname+"wcount.gz", membuf);
  writeSBVecs(unh, dictname+"dict.gz", membuf);
}
