void setsizes(int N, dim3 *gridp, int *nthreadsp);

int apply_links(float *A, int *L, float *C, int nrows, int ncols);

int apply_preds(float *A, int *L, float *C, int nrows, int ncols);

int apply_lls(float *A, float *B, int *L, float *C, int nrows, int ncols);

int apply_derivs(float *A, float *B, int *L, float *C, int nrows, int ncols);

int veccmp(int *A, int *B, int *C);

int hammingdists(int *a, int *b, int *w, int *op, int *ow, int n);

int LDA_Gibbs(int nrows, int nnz, float *A, float *B, float *AN, float *BN, int *Cir, int *Cic, float *P, float nsamps);

int LDA_Gibbs1(int nrows, int nnz, float *A, float *B, int *Cir, int *Cic, float *P, int *Ms, int *Us, int k);

int LDA_Gibbsv(int nrows, int nnz, float *A, float *B, float *AN, float *BN, int *Cir, int *Cic, float *P, float* nsamps);

int treePack(float *fdata, int *treenodes, int *icats, long long *out, int *fieldlens, int nrows, int ncols, int ntrees, int nsamps, int seed);

int treePackInt(int *fdata, int *treenodes, int *icats, long long *out, int *fieldlens, int nrows, int ncols, int ntrees, int nsamps, int seed);

int minImpurity(long long *keys, int *counts, int *outv, int *outf, float *outg, int *outc, int *jc, int *fieldlens, 
                int nnodes, int ncats, int nsamps, int impType);

int floatToInt(int n, float *in, int *out, int nbits);

int findBoundaries(long long *keys, int *jc, int n, int njc, int shift);

int mergeInds(long long *keys, long long *okeys, int *counts, int n, int *cspine);

int getMergeIndsLen(long long *keys, int n, int *cspine);

int hashmult(int nrows, int nfeats, int ncols, float *A, float *Bdata, int *Bir, int *Bjc, float *C);

int hashmultT(int nrows, int nfeats, int ncols, float *A, float *Bdata, int *Bir, int *Bjc, float *C);
