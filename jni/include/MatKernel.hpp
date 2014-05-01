
int apply_links(float *A, int *L, float *C, int nrows, int ncols);

int apply_preds(float *A, int *L, float *C, int nrows, int ncols);

int apply_lls(float *A, float *B, int *L, float *C, int nrows, int ncols);

int apply_derivs(float *A, float *B, int *L, float *C, int nrows, int ncols);

int veccmp(int *A, int *B, int *C);

int hammingdists(int *a, int *b, int *w, int *op, int *ow, int n);

int LDA_Gibbs(int nrows, int nnz, float *A, float *B, float *AN, float *BN, int *Cir, int *Cic, float *P, float nsamps);

int LDA_Gibbs1(int nrows, int nnz, float *A, float *B, int *Cir, int *Cic, float *P, int *Ms, int *Us, int k);

int treePack(int *fdata, int *treenodes, int *icats, int *jc, long long *out, int *fieldlens, int nrows, int ncols, int ntrees, int nsamps);

int minImpurity(long long *keys, int *counts, int *outv, int *outf, float *outg, int *jc, int *fieldlens, 
                int nnodes, int ncats, int nsamps, int impType);

