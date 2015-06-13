// Read a file and parse each line according to the format file.
// Output integer matrices with lookup maps in either ascii text 
// or matlab binary form. 

#include <time.h>
#include <limits.h>
#include "utils.h"

#define BUFSIZE 1048576

enum ftypes {
  ftype_int = 1,
  ftype_dint,
  ftype_qhex,
  ftype_float,
  ftype_double,
  ftype_word,
  ftype_string,
  ftype_date,
  ftype_cmdate,
  ftype_mdate,
  ftype_dt,
  ftype_mdt,
  ftype_group,
  ftype_igroup,
  ftype_digroup
};

class fieldtype {
 public:
  ivector iv;
  divector div;
  qvector qv;
  fvector fv;
  dvector dv;
  imatrix im;
  dimatrix dim;
 fieldtype() : iv(0), div(0), fv(0), dv(0), im(0), dim(0) {};
  int writeInts(string fname) {return writeIntVec(iv, fname, BUFSIZE);}
  int writeDInts(string fname) {return writeDIntVec(div, fname, BUFSIZE);}
  int writeQInts(string fname) {return writeQIntVec(qv, fname, BUFSIZE);}
  int writeFloats(string fname) {return writeFVec(fv, fname, BUFSIZE);}
  int writeDoubles(string fname) {return writeDVec(dv, fname, BUFSIZE);}
  int writeIVecs(string fname) {return writeIntVecs(im, fname, BUFSIZE);}
  int writeIntVecsTxt(string fname);
  int writeDIVecs(string fname) {return writeDIntVecs(dim, fname, BUFSIZE);}
};

typedef vector<fieldtype> ftvector;

class stringIndexer {
 public:
  strhash htab;
  unhash unh;
  ivector count;
  int size;
  char * linebuf;
  stringIndexer(int sz) : htab(sz), unh(sz), count(sz), size(sz) {
    linebuf = new char[BUFSIZE];
  };
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
  ivector checkstring(char *, const char *);
  ivector checkstrings(char **, const char *, int);
  ivector checkgroup(char ** here, const char * delim2, int len);
  divector checkdgroup(char ** here, const char * delim2, int len);
  stringIndexer shrink(int);
  stringIndexer shrink_to_size(int);  
  ivector indexMap(stringIndexer &);
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

ivector stringIndexer::
checkstring(char * here, const char * delim2) {
  char * next;
  ivector out(0);
  here += strspn(here, delim2);
  //  strtoupper(here);
  while (here && *here) {
    next = strpbrk(here, delim2);
    if (next) {
      *(next++) = 0;
      next += strspn(next, delim2);
    }
    out.push_back(checkword(here));
    here = next;
  }    
  return out;
}

ivector stringIndexer::
checkstrings(char ** here, const char * delim2, int len) {
  int i;
  char * next;
  ivector out(0);
  //  strtoupper(*here);
  for (i = 0; i < len; i++) {
    next = strpbrk(*here, delim2);
    if (next) {
      *(next++) = 0;
    }
    if (strlen(*here) > 0)
      out.push_back(checkword(*here));
    else
      out.push_back(0);
    *here = next;
  }    
  return out;
}

ivector stringIndexer::
checkgroup(char ** here, const char * delim2, int len) {
  int i;
  char * next;
  ivector out(0);
  int64 val;
  next = strpbrk(*here, delim2);
  if (next == *here) (*here)++;
  for (i = 0; i < len && *here; i++) {
    next = strpbrk(*here, delim2);
    if (next) {
      *(next++) = 0;
    }
    if (strlen(*here) > 0) {
      sscanf(*here, "%lld", &val);
      out.push_back(val);
    } else
      out.push_back(0);
    *here = next;
  }    
  return out;
}


divector stringIndexer::
checkdgroup(char ** here, const char * delim2, int len) {
  int i;
  char * next;
  divector out(0);
  int64 val;
  next = strpbrk(*here, delim2);
  if (next == *here) (*here)++;
  for (i = 0; i < len && *here; i++) {
    next = strpbrk(*here, delim2);
    if (next) {
      *(next++) = 0;
    }
    if (strlen(*here) > 0) {
      sscanf(*here, "%lld", &val);
      out.push_back(val);
    } else
      out.push_back(0);
    *here = next;
  }    
  return out;
}

// Need a deep copy constructor for stringIndexer for this to work
stringIndexer stringIndexer::
shrink(int threshold) {
  int i, newc;
  char * newstr;
  for (i = 0, newc = 0; i < size; i++) {
    if (count[i] >= threshold) {
      newc++;
    }
  }
  stringIndexer retval(newc);
  try {
    for (i = 0, newc = 0; i < size; i++) {
      if (count[i] >= threshold) {
	newstr = new char[strlen(unh[i])+1];
	strcpy(newstr, unh[i]);
	retval.count[newc] = count[i];
	retval.unh[newc] = newstr;
	retval.htab[newstr] = ++newc;
      }
    }
  } catch (std::bad_alloc) {
    cerr << "stringIndexer:shrink: allocation error" << endl;
    throw;
  }
  return retval;
}

stringIndexer stringIndexer::
shrink_to_size(int newsize) {
  int i, newc, minth=1, maxth=0, midth;
  for (i = 0, newc = 0; i < size; i++) {
    if (count[i] >= maxth) {
      maxth = count[i];
    }
  }
  while (maxth > (minth+1)) {
    midth = (maxth + minth) >> 1;
    for (i = 0, newc = 0; i < size; i++) {
      if (count[i] >= midth) {
	newc++;
      }
    }
    if (newc > newsize) {
      minth = midth;
    } else {
      maxth = midth;
    }
  }
  return shrink(maxth);
}

ivector stringIndexer::
indexMap(stringIndexer & A) {
  int i;
  ivector remap(size);
  for (i = 0; i < size; i++) {
    if (A.htab.count(unh[i])) {
      remap[i] = A.htab[unh[i]];
    } else {
      remap[i] = 0;
    }
  }
  return remap;
}
// Deep copy constructor
stringIndexer::
stringIndexer(const stringIndexer & si) : 
  htab(0), unh(si.size), count(si.size), size(si.size) 
{
  int i;
  char *str, *newstr;
  for (i = 0; i < size; i++) {
    str = si.unh[i];
    newstr = new char[strlen(str)+1];
    strcpy(newstr, str);
    unh[i] = newstr;
    htab[newstr] = i+1;
    count[i] = si.count[i];
  }
  linebuf = new char[BUFSIZE];
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


typedef vector<stringIndexer> srivector;

int parseLine(char * line, int membuf, int lineno, const char * delim1, ivector & tvec, 
	       svector & delims, srivector & srv, ftvector & out, int grpsize) {
  int i, ival;
  int64 dival;
  qint qval;
  float fval;
  double dval;
  char * here, * next; 

  here = line;
  for (i = 0; i < tvec.size(); i++) {
    next = strpbrk(here, delim1);
    if (!next && i < tvec.size()-1) {
      cerr << "parseLine: format error line " << lineno << endl;
      cerr << "  contents: " << line << " ... " << here << endl;
      throw 10;
    }
    if (next && *next) *(next++) = 0;
    switch (tvec[i]) {
    case ftype_int:
      sscanf(here, "%d", &ival);
      out[i].iv.push_back(ival);
      break;
    case ftype_dint:
      sscanf(here, "%lld", &dival);
      out[i].div.push_back(dival);
      break;
    case ftype_qhex:
      sscanf(here, "%16llx%16llx", &qval.top, &qval.bottom);
      out[i].qv.push_back(qval);
      break;
    case ftype_float:
      sscanf(here, "%f", &fval);
      out[i].fv.push_back(fval);
      break;
    case ftype_double:
      sscanf(here, "%lf", &dval);
      out[i].dv.push_back(dval);
      break;
    case ftype_word:
      here += strspn(here, " ");
      out[i].iv.push_back(srv[i].checkword(here));
      break;
    case ftype_string:
      out[i].im.push_back(srv[i].checkstring(here, delims[i].c_str()));
      break;
    case ftype_dt:  
      ival = parsedt(here);
      if (ival < 0)
	printf("\nWarning: bad dt on line %d\n", lineno);
      out[i].iv.push_back(ival);
      break;
    case ftype_mdt:
      ival = parsemdt(here);
      if (ival < 0)
	printf("\nWarning: bad mdt on line %d\n", lineno);
      out[i].iv.push_back(ival);
      break;
    case ftype_date:
      ival = parsedate(here);
      if (ival < 0)
	printf("\nWarning: bad date on line %d\n", lineno);
      out[i].iv.push_back(ival);
      break;
    case ftype_mdate:
      ival = parsemdate(here);
      if (ival < 0)
	printf("\nWarning: bad mdate on line %d\n", lineno, here);
      out[i].iv.push_back(ival);
      break;
    case ftype_cmdate:
      ival = parsecmdate(here);
      if (ival < 0)
	printf("\nWarning: bad cmdate on line %d\n", lineno, here);
      out[i].iv.push_back(ival);
      break;
    case ftype_group:
      *(next-1) = *delim1;
      out[i].im.push_back(srv[i].checkstrings(&here, delim1, grpsize));
      next = here;
      break;
    case ftype_igroup:
      //      *(next-1) = *delim1;
      out[i].im.push_back(srv[i].checkgroup(&here, delims[i].c_str(), grpsize));
      next = here;
      break;
    case ftype_digroup:
      //      *(next-1) = *delim1;
      out[i].dim.push_back(srv[i].checkdgroup(&here, delims[i].c_str(), grpsize));
      next = here;
      break;
    default:
      break;
    }
    here = next;
  }
  return 0;
}


string getfield(char * line, const char * delim1, int k);

int parseFormat(string ffname, ivector & tvec, svector & dnames, svector & delims, int *grpsize) {
  ifstream ifstr(ffname.c_str(), ios::in);
  if (ifstr.fail() || ifstr.eof()) {
    cerr << "couldnt open format file" << endl;
    throw;
  }
  char *next, *third, *newstr, *linebuf = new char[80];
  while (!ifstr.bad() && !ifstr.eof()) {
    ifstr.getline(linebuf, 80);
    if (strlen(linebuf) > 1) {
      next = strchr(linebuf, ' ');
      *next++ = 0;
      third = strchr(next, ' ');
      if (third)
	*third++ = 0;
      dnames.push_back(next);
      if (strncmp(linebuf, "int", 3) == 0) {
	tvec.push_back(ftype_int);
	delims.push_back("");
      } else if (strncmp(linebuf, "dint", 4) == 0) {
	tvec.push_back(ftype_dint);
	delims.push_back("");
      } else if (strncmp(linebuf, "qhex", 4) == 0) {
	tvec.push_back(ftype_qhex);
	delims.push_back("");
      } else if (strncmp(linebuf, "float", 5) == 0) {
	tvec.push_back(ftype_float);
	delims.push_back("");
      } else if (strncmp(linebuf, "double", 6) == 0) {
	tvec.push_back(ftype_double);
	delims.push_back("");
      } else if (strncmp(linebuf, "word", 4) == 0) {
	tvec.push_back(ftype_word);
	delims.push_back("");
      } else if (strncmp(linebuf, "string", 6) == 0) {
	tvec.push_back(ftype_string);
	ifstr.getline(linebuf, 80);
        newstr = new char[strlen(linebuf)+1];
        strcpy(newstr, linebuf);
	delims.push_back(newstr);
      } else if (strncmp(linebuf, "date", 4) == 0) {
	tvec.push_back(ftype_date);
	delims.push_back("");
      } else if (strncmp(linebuf, "mdate", 5) == 0) {
	tvec.push_back(ftype_mdate);
	delims.push_back("");
      } else if (strncmp(linebuf, "cmdate", 6) == 0) {
	tvec.push_back(ftype_cmdate);
	delims.push_back("");
      } else if (strncmp(linebuf, "dt", 2) == 0) {
	tvec.push_back(ftype_dt);
	delims.push_back("");
      } else if (strncmp(linebuf, "mdt", 3) == 0) {
	tvec.push_back(ftype_mdt);
	delims.push_back("");
      } else if (strncmp(linebuf, "group", 5) == 0) {
	sscanf(third, "%d", grpsize);
	tvec.push_back(ftype_group);
	delims.push_back("");
      } else if (strncmp(linebuf, "igroup", 6) == 0) {
	sscanf(third, "%d", grpsize);
	tvec.push_back(ftype_igroup);
	ifstr.getline(linebuf, 80);
	delims.push_back(linebuf);
      } else if (strncmp(linebuf, "digroup", 7) == 0) {
	sscanf(third, "%d", grpsize);
	tvec.push_back(ftype_digroup);
	ifstr.getline(linebuf, 80);
	delims.push_back(linebuf);
      } else {
        cerr << "couldnt parse format file line " << tvec.size()+1 << endl;
	throw;
      }
    }
  }
  return tvec.size();
  delete [] linebuf;
}


const char usage[] = 
"\nParser for Tabular input files. Arguments are\n"
"   -i <infile>     input file to read\n"
"   -f <fmtfile>    format file\n"
"   -o <outfile>    output file prefix. Multiple output files are written with names\n"
"          <outfile><varname>.<type>[.gz]\n"
"   -d <dstring>    delimiter string for input fields. Defaults to tab.\n"
"   -s N            set buffer size to N.\n"
"   -c              produce compressed (gzipped) output files.\n\n"
;

int main(int argc, char ** argv) {
  int iline=0, i, iarg=1, nfields=1, jmax=0, writetxt=0, writemat=0, grpsize=1, pos=0, epos=0;
  int membuf=1048576;
  char * here, *linebuf, *readbuf, *thisfile, *filestart;
  string ifname = "", thisfname = "", ofname = "", ofprefix = "", ffname = "", fdelim="\t", suffix="";
  if (argc < 2) {
    printf("%s", usage);
    return 1;
  }
  while (iarg < argc) {
    if (strncmp(argv[iarg], "-d", 2) == 0) {
      fdelim = argv[++iarg];
    } else if (strncmp(argv[iarg], "-c", 2) == 0) {
      suffix=".gz";
    } else if (strncmp(argv[iarg], "-f", 2) == 0) {
      ffname = argv[++iarg];
    } else if (strncmp(argv[iarg], "-i", 2) == 0) {
      ifname = argv[++iarg];
    } else if (strncmp(argv[iarg], "-o", 2) == 0) {
      ofprefix = argv[++iarg];
    } else if (strncmp(argv[iarg], "-s", 2) == 0) {
      membuf = strtol(argv[++iarg],NULL,10);
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
  ivector tvec(0);
  svector delims(0);
  svector dnames(0);
  nfields = parseFormat(ffname, tvec, dnames, delims, &grpsize);
  srivector srv(nfields);
  ftvector ftv(nfields);

  istream * ifstr;
  linebuf = new char[membuf];
  readbuf = new char[membuf];

  pos = ifname.find_first_of(" ,");
  thisfname = ifname.substr(0, pos);

  while (thisfname.length() > 0) {
    cout << "Processing " << thisfname << endl;
    cout.flush();
    ifstr = open_in_buf(thisfname, readbuf, membuf);

    epos = thisfname.find_last_of("/");
    ofname = ofprefix + thisfname.substr(epos+1);

    while (!ifstr->bad() && !ifstr->eof()) {
      ifstr->getline(linebuf, membuf-1);
      linebuf[membuf-1] = 0;
      if (ifstr->fail()) {
        ifstr->clear();
        ifstr->ignore(LONG_MAX,'\n');
      }
      if (strlen(linebuf) > 0) {
        jmax++;
        try {
          parseLine(linebuf, membuf, ++iline, fdelim.c_str(), tvec, delims, srv, ftv, grpsize);
        } catch (int e) {
          cerr << "Continuing" << endl;
        }
      }
      if ((jmax % 100000) == 0) {
        cout<<"\r"<<jmax<<" lines processed";
        cout.flush();
      }
    }
    if (ifstr) delete ifstr;
    cout<<"\r"<<jmax<<" lines processed"<< endl;
    cout.flush();

    for (i = 0; i < nfields; i++) {
      switch (tvec[i]) {
      case ftype_int: case ftype_dt: case ftype_mdt: case ftype_date: case ftype_mdate: case ftype_cmdate:
        ftv[i].writeInts(ofname + "_" + dnames[i] + ".imat" + suffix);
        ftv[i].iv.clear();
        break;
      case ftype_dint:
        ftv[i].writeDInts(ofname + "_" + dnames[i] + ".dimat" + suffix);
        ftv[i].div.clear();
        break;
      case ftype_qhex:
        ftv[i].writeQInts(ofname + "_" + dnames[i] + ".imat" + suffix);
        ftv[i].qv.clear();
        break;
      case ftype_float:
        ftv[i].writeFloats(ofname + "_" + dnames[i] + ".fmat" + suffix);
        ftv[i].fv.clear();
        break;
      case ftype_double:
        ftv[i].writeDoubles(ofname + "_" + dnames[i] + ".dmat" + suffix);
        ftv[i].dv.clear();
        break;
      case ftype_word:
        ftv[i].writeInts(ofname + "_" + dnames[i] + ".imat" + suffix);
        ftv[i].iv.clear();
        break;
      case ftype_string: case ftype_group: 
        ftv[i].writeIVecs(ofname + "_" + dnames[i] + ".imat" + suffix);
        ftv[i].im.clear();
        break;
      case ftype_igroup:
        ftv[i].writeIVecs(ofname + "_" + dnames[i] + ".imat" + suffix);
        ftv[i].im.clear();
        break;
      case ftype_digroup:
        ftv[i].writeDIVecs(ofname + "_" + dnames[i] + ".lmat" + suffix);
        ftv[i].dim.clear();
        break;
      default:
        break;
      }    
    }
    epos = ifname.find_first_of(" ,", pos+1);
    if (epos > 0) {
      thisfname = ifname.substr(pos+1, epos-pos-1);
      pos = epos; 
    } else {
      thisfname = ifname.substr(pos+1);
      pos = ifname.length()-1;
    }
    jmax = 0;
  }
  printf("\n");
  for (i = 0; i < nfields; i++) {
    switch (tvec[i]) {
    case ftype_int: case ftype_dt: case ftype_mdt: case ftype_date: case ftype_mdate: case ftype_cmdate:
    case ftype_dint: case ftype_qhex: case ftype_float: case ftype_double:
      break;
    case ftype_word:
      srv[i].writeMap(ofprefix + dnames[i], suffix);
      break;
    case ftype_string: case ftype_group: 
      srv[i].writeMap(ofprefix + dnames[i], suffix);
      break;
    case ftype_igroup:
      break;
    case ftype_digroup:
      break;
    default:
      break;
    }    
  }
  if (linebuf) delete [] linebuf;
  return 0;
}
