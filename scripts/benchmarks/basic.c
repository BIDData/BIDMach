#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv) {
  int n, nreps, irep, i, j, ibase;
  struct timeval tp1, tp2;
  double t1, t2, tdiff;
  double flops;
  sscanf(argv[1], "%d", &n);
  sscanf(argv[2], "%d", &nreps);
  float *a;
  float *b;
  float *c;
  a = (float *)malloc(n*n*sizeof(float));
  b = (float *)malloc(n*n*sizeof(float));
  c = (float *)malloc(n*n*sizeof(float));
  gettimeofday(&tp1, NULL);
  for (irep = 0; irep < nreps; irep++) {
    for (i = 0; i < n; i++) {
      ibase = i * n;
      for (j = 0; j < n; j++) {
	c[j + ibase] = a[j + ibase] + b[j + ibase];
      }
    }
  }
  gettimeofday(&tp2, NULL);
  t1 = tp1.tv_sec + 1.0e-6*tp1.tv_usec;
  t2 = tp2.tv_sec + 1.0e-6*tp2.tv_usec;
  tdiff = t2 - t1;
  flops = 1.0 * n * n * nreps;
  printf("time %f, Mflops %f %f\n", tdiff/nreps, flops/tdiff/1.0e6, c[1000000-1]);    
}

  
  
