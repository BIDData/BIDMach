
#define SYNC_STREAM cudaStreamDefault

void setsizes(int N, dim3 *gridp, int *nthreadsp);

void setsizesLean(int N, dim3 *gridp, int *nthreadsp);

int apply_links(float *A, int *L, float *C, int nrows, int ncols);

int apply_preds(float *A, int *L, float *C, int nrows, int ncols);

int apply_lls(float *A, float *B, int *L, float *C, int nrows, int ncols);

int apply_derivs(float *A, float *B, int *L, float *C, int nrows, int ncols);

int apply_dlinks(double *A, int *L, double *C, int nrows, int ncols);

int apply_dpreds(double *A, int *L, double *C, int nrows, int ncols);

int apply_dlls(double *A, double *B, int *L, double *C, int nrows, int ncols);

int apply_dderivs(double *A, double *B, int *L, double *C, int nrows, int ncols);

int veccmp(int *A, int *B, int *C);

int hammingdists(int *a, int *b, int *w, int *op, int *ow, int n);

int LDA_Gibbs(int nrows, int nnz, float *A, float *B, float *AN, float *BN, int *Cir, int *Cic, float *P, float nsamps);

int LDA_GibbsBino(int nrows, int nnz, float *A, float *B, float *AN, float *BN, int *Cir, int *Cic, float *Cv, float *P, int nsamps);

int LDA_Gibbs1(int nrows, int nnz, float *A, float *B, int *Cir, int *Cic, float *P, int *Ms, int *Us, int k);

int LDA_Gibbsv(int nrows, int nnz, float *A, float *B, float *AN, float *BN, int *Cir, int *Cic, float *P, float* nsamps);

int treePack(float *fdata, int *treenodes, int *icats, long long *out, int *fieldlens, int nrows, int ncols, int ntrees, int nsamps, int seed);

int treePackfc(float *fdata, int *treenodes, float *fcats, long long *out, int *fieldlens, int nrows, int ncols, int ntrees, int nsamps, int seed);

int treePackInt(int *fdata, int *treenodes, int *icats, long long *out, int *fieldlens, int nrows, int ncols, int ntrees, int nsamps, int seed);

int minImpurity(long long *keys, int *counts, int *outv, int *outf, float *outg, int *outc, int *jc, int *fieldlens, 
                int nnodes, int ncats, int nsamps, int impType);

int treeWalk(float *fdata, int *inodes, float *fnodes, int *itrees, int *ftrees, int *vtrees, float *ctrees,
             int nrows, int ncols, int ntrees, int nnodes, int getcat, int nbits, int nlevels);

int floatToInt(int n, float *in, int *out, int nbits);

int jfeatsToIfeats(int itree, int *inodes, int *jfeats, int *ifeats, int n, int nfeats, int seed);

int findBoundaries(long long *keys, int *jc, int n, int njc, int shift);

int hashmult(int nrows, int nfeats, int ncols, int bound1, int bound2, float *A, float *Bdata, int *Bir, int *Bjc, float *C, int transpose);

int hashcross(int nrows, int nfeats, int ncols, float *A, float *Bdata, int *Bir, int *Bjc, float *Cdata, int *Cir, int *Cjc, float *D, int transpose);

int multinomial(int nrows, int ncols, float *A, int *B, float *Norm, int nvals);

int multinomial2(int nrows, int ncols, float *A, int *B, int nvals);

int multADAGrad(int nrows, int ncols, int nnz, float *A, float *Bdata, int *Bir, int *Bic, float *MM, float *Sumsq, 
                float *Mask, int maskrows, float *lrate, int lrlen, float *vexp, int vexplen, float *texp, int texplen, 
                float istep, int addgrad, float epsilon, int biasv, int nbr);

int multADAGradTile(int nrows, int ncols, int y, int x, int nnz, float *A, int lda, float *Bdata, int *Bir, int *Bic, float *MM, 
                    float *Sumsq, float *Mask, int maskrows, float *lrate, int lrlen, float *vexp, int vexplen, 
                    float *texp, int texplen, float istep, int addgrad, float epsilon, int biasv, int nbr);

int multGradTile(int nrows, int ncols, int y, int x, int nnz, float *A, int lda, float *Bdata, int *Bir, int *Bic, float *MM, 
                 float *Mask, int maskrows, float *lrate, int lrlen, float limit, int biasv, int nbr);

int hashmultADAGrad(int nrows, int nfeats, int ncols, int bound1, int bound2, float *A, float *Bdata, int *Bir, int *Bjc, int transpose, 
                    float *MM, float *Sumsq, float *Mask, int maskrows, float *lrate, int lrlen, 
                    float *vexp, int vexplen, float *texp, int texplen, float istep, int addgrad, float epsilon);

int word2vecPos(int nrows, int ncols, int shift, int *W, int *LB, int *UB, float *A, float *B, float lrate, float vexp);

int word2vecNeg(int nrows, int ncols, int nwa, int nwb, int *WA, int *WB, float *A, float *B, float lrate, float vexp);

int word2vecNegFilt(int nrows, int ncols, int nwords, int nwa, int nwb, int *WA, int *WB, float *A, float *B, float lrate, float vexp);

int word2vecEvalPos(int nrows, int ncols, int shift, int *W, int *LB, int *UB, float *A, float *B, float *Retval);

int word2vecEvalNeg(int nrows, int ncols, int nwa, int nwb, int *WA, int *WB, float *A, float *B, float *Retval);

int word2vecFwd(int nrows, int ncols, int nwa, int nwb, int *WA, int *WB, float *A, float *B, float *C);

int word2vecBwd(int nrows, int ncols, int nwa, int nwb, int *WA, int *WB, float *A, float *B, float *C, float lrate);

int apply_fwd(float *A, float *B, int ifn, int n);

int apply_deriv(float *A, float *B, float *C, int ifn, int n);

int ADAGrad(int nrows, int ncols, float *mm, float *um, float *ssq, float *mask, int maskr, float nw, float *ve, int nve,
            float *ts, int nts, float *lrate, int nlrate, float langevin, float eps, int doupdate);

int ADAGradm(int nrows, int ncols, float *mm, float *um, float *ssq, float *momentum, float mu, float *mask, int maskr,
	     float nw, float *ve, int nve, float *ts, int nts, float *lrate, int nlrate, float langevin, float eps, int doupdate);

int ADAGradn(int nrows, int ncols, float *mm, float *um, float *ssq, float *momentum, float mu, float *mask, int maskr,
	     float nw, float *ve, int nve, float *ts, int nts, float *lrate, int nlrate, float langevin, float eps, int doupdate);

int lstm_fwd(float *inC, float *LIN1, float *LIN2, float *LIN3, float *LIN4, float *outC, float *outH, int n);

int lstm_bwd(float *inC, float *LIN1, float *LIN2, float *LIN3, float *LIN4, float *doutC, float *doutH, 
             float *dinC, float *dLIN1, float *dLIN2, float *dLIN3, float *dLIN4, int n);

int pairembed(int *r1, int *r2, long long *res, int n);

int pairMultTile(int nrows, int ncols, int bound1, int bound2, float *A, int lda, float *A2, int lda2, 
                 float *Bdata, int *Bir, int *Bjc, int broff, int bcoff, float *C, int ldc, int transpose);

int pairMultADAGradTile(int nrows, int ncols, int bound1, int bound2, float *A, int lda, int aroff, int acoff,
                        float *Bdata, int *Bir, int *Bjc, int broff, int bcoff, int transpose, 
                        float *MM, int ldmm, float *Sumsq, float *Mask, int maskrows, float *lrate, int lrlen, 
                        float *vexp, int vexplen, float *texp, int texplen, float istep, int addgrad, float epsilon);

int linComb(float *X, float wx, float *Y, float wy, float *Z, int len);
