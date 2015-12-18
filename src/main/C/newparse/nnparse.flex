/* Scanner for neural net datasources */

%{
  extern int checkword(char *);
  extern void addtok(int tok);
  extern int parsedate(char * str);
  extern int numlines;
  extern int doparagraphids;
  extern int sentenceid;
  extern int paragraphid;

  #define YY_USER_ACTION doparagraphids=1;  // Macro happens at initialization
%}

%option never-interactive
%option noyywrap

LETTER	   [a-zA-Z_]
DIGIT	   [0-9]
PUNCT	   [;:,.?!]

%% 

{LETTER}+    {
  int iv = checkword(yytext);
	}

"<"{LETTER}+">"    {
  int iv = checkword(yytext);
	}

"</"{LETTER}+">"    {
  int iv = checkword(yytext);
	}

{PUNCT}	  {
  int iv = checkword(yytext);
	  }

"..""."*  {
  char ell[] = "...";
  int iv  = checkword(ell);
	  }

". "  {
  sentenceid++;
  char ell[] = ".";
  int iv  = checkword(ell);
	  }

"? "  {
  sentenceid++;
  char ell[] = "?";
  int iv  = checkword(ell);
	  }

"! "  {
  sentenceid++;
  char ell[] = "!";
  int iv  = checkword(ell);
	  }

[\n]	  {
	  numlines++;
	  paragraphid++;
	  sentenceid = 0;
	  if (numlines % 1000000 == 0) {
	  fprintf(stderr, "\r%05d lines", numlines);
      fflush(stderr);
	  }	  
	  }

.         {}

%%

