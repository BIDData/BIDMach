package edu.berkeley.bid;
import jcuda.Pointer;

public final class CUMACH {

    private CUMACH() {}

    static {
        LibUtils.loadLibrary("bidmachcuda");
    } 
    
    public static native int applylinks(Pointer A, Pointer L, Pointer C, int nrows, int ncols);
    
    public static native int applypreds(Pointer A, Pointer L, Pointer C, int nrows, int ncols);
    
    public static native int applylls(Pointer A, Pointer B, Pointer L, Pointer C, int nrows, int ncols);
    
    public static native int applyderivs(Pointer A, Pointer B, Pointer L, Pointer C, int nrows, int ncols);
    
    public static native int applydlinks(Pointer A, Pointer L, Pointer C, int nrows, int ncols);
    
    public static native int applydpreds(Pointer A, Pointer L, Pointer C, int nrows, int ncols);
    
    public static native int applydlls(Pointer A, Pointer B, Pointer L, Pointer C, int nrows, int ncols);
    
    public static native int applydderivs(Pointer A, Pointer B, Pointer L, Pointer C, int nrows, int ncols);
    
    public static native int LDAgibbs(int nr, int nnz, Pointer A, Pointer B, Pointer AN, Pointer BN, Pointer Cir, Pointer Cic, Pointer P, float nsamps);

    public static native int LDAgibbsBino(int nr, int nnz, Pointer A, Pointer B, Pointer AN, Pointer BN, Pointer Cir, Pointer Cic, Pointer Cv, Pointer P, int nsamps);

    public static native int LDAgibbsx(int nr, int nnz, Pointer A, Pointer B, Pointer Cir, Pointer Cic, Pointer P, Pointer Ms, Pointer Us, int k);
    
    public static native int LDAgibbsv(int nr, int nnz, Pointer A, Pointer B, Pointer AN, Pointer BN, Pointer Cir, Pointer Cic, Pointer P, Pointer nsamps);
    
    public static native int treeprod(Pointer trees, Pointer feats, Pointer tpos, Pointer otvs, int nrows, int ncols, int ns, int tstride, int ntrees);

    public static native int treesteps(Pointer trees, Pointer feats, Pointer tpos, Pointer otpos, int nrows, int ncols, int ns, int tstride, int ntrees, int tdepth);
    
    public static native int veccmp(Pointer A, Pointer B, Pointer C);
    
    public static native int hammingdists(Pointer A, Pointer B, Pointer W, Pointer OP, Pointer OW, int n);

    public static native int treePack(Pointer id, Pointer tn, Pointer icats, Pointer out, Pointer fl, int nrows, int ncols, int ntrees, int nsamps, int seed);

    public static native int treePackfc(Pointer id, Pointer tn, Pointer icats, Pointer out, Pointer fl, int nrows, int ncols, int ntrees, int nsamps, int seed);

    public static native int treePackInt(Pointer id, Pointer tn, Pointer icats, Pointer out, Pointer fl, int nrows, int ncols, int ntrees, int nsamps, int seed);
    
    public static native int treeWalk(Pointer fdata, Pointer inodes, Pointer fnodes, Pointer itrees, Pointer ftrees, Pointer vtrees, Pointer ctrees,
    		int nrows, int ncols, int ntrees, int nnodes, int getcat, int nbits, int nlevels);

    public static native int minImpurity(Pointer keys, Pointer counts, Pointer outv, Pointer outf, Pointer outg, Pointer outc, Pointer jc, Pointer fieldlens, int nnodes, int ncats, int nsamps, int impType);

    public static native int findBoundaries(Pointer keys, Pointer jc, int n, int njc, int shift);
    
    public static native int floatToInt(int n, Pointer in, Pointer out, int nbits);
    
    public static native int jfeatsToIfeats(int itree, Pointer inodes, Pointer jfeats, Pointer ifeats, int n, int nfeats, int seed);
    
    public static native int mergeInds(Pointer keys, Pointer okeys, Pointer counts, int n, Pointer cspine);
    
    public static native int getMergeIndsLen(Pointer keys, int n, Pointer cspine);
    
    public static native int hashMult(int nrows, int nfeats, int ncols, int bound1, int bound2, Pointer jA, Pointer jBdata, Pointer jBir, Pointer jBjc, Pointer jC, int transpose);
    
    public static native int hashCross(int nrows, int nfeats, int ncols, Pointer jA, Pointer jBdata, Pointer jBir, Pointer jBjc, Pointer jCdata, Pointer jCir, Pointer jCjc, Pointer D, int transpose);
    
    public static native int cumsumc(int nrows, int ncols, Pointer jA, Pointer jB);
    
    public static native int multinomial(int nrows, int ncols, Pointer jA, Pointer jB, Pointer jNorm, int nvals);
    
    public static native int multinomial2(int nrows, int ncols, Pointer jA, Pointer jB, int nvals);
  
//    public static native int multADAGrad(int nrows, int ncols, int nnz, Pointer A, Pointer Bdata, Pointer Bir, Pointer Bic, Pointer MM, Pointer Sumsq, 
//    		Pointer Mask, int maskrows, Pointer lrate, int lrlen, Pointer vexp, int vexplen, Pointer texp, int texplen, float istep, int addgrad, float epsilon);
    
    public static native int multADAGrad(int nrows, int ncols, int nnz, Pointer A, Pointer Bdata, Pointer Bir, Pointer Bic, Pointer MM, Pointer Sumsq, Pointer Mask, int maskrows, 
    		Pointer lrate, int lrlen, Pointer vexp, int vexplen, Pointer texp, int texplen, float istep, int addgrad, float epsilon, int biasv, int nbr);
    
    public static native int multADAGradTile(int nrows, int ncols, int y, int x, int nnz, Pointer A, int lda, Pointer Bdata, Pointer Bir, Pointer Bic, Pointer MM, Pointer Sumsq, Pointer Mask, int maskrows, 
    		Pointer lrate, int lrlen, Pointer vexp, int vexplen, Pointer texp, int texplen, float istep, int addgrad, float epsilon, int biasv, int nbr);
    
    public static native int multGradTile(int nrows, int ncols, int y, int x, int nnz, Pointer A, int lda, Pointer Bdata, Pointer Bir, Pointer Bic, Pointer MM, Pointer Mask, int maskrows, 
    		Pointer lrate, int lrlen, float limit, int biasv, int nbr);

    public static native int hashmultADAGrad(int nrows, int nfeats, int ncols, int bound1, int bound2, Pointer A, Pointer Bdata, Pointer Bir, Pointer Bjc, int transpose,
    		Pointer MM, Pointer Sumsq, Pointer Mask, int maskrows, Pointer lrate, int lrlen, Pointer vexp, int vexplen, Pointer texp, int texplen, float istep, int addgrad, float epsilon);   
    
    public static native int word2vecPos(int nrows, int ncols, int shift, Pointer W, Pointer LB, Pointer UB, Pointer A, Pointer B, float lrate, float vexp);
     
    public static native int word2vecNeg(int nrows, int ncols, int nwa, int nwb, Pointer WA, Pointer WB, Pointer A, Pointer B, float lrate, float vexp);
   
    public static native int word2vecNegFilt(int nrows, int ncols, int nwords, int nwa, int nwb, Pointer WA, Pointer WB, Pointer A, Pointer B, float lrate, float vexp);
    
    public static native int word2vecEvalPos(int nrows, int ncols, int shift, Pointer W, Pointer LB, Pointer UB, Pointer A, Pointer B, Pointer retVal);
    
    public static native int word2vecEvalNeg(int nrows, int ncols, int nwa, int nwb, Pointer WA, Pointer WB, Pointer A, Pointer B, Pointer retVal);
    
    public static native int word2vecFwd(int nrows, int ncols, int nwa, int nwb, Pointer WA, Pointer WB, Pointer A, Pointer B, Pointer C);
    
    public static native int word2vecBwd(int nrows, int ncols, int nwa, int nwb, Pointer WA, Pointer WB, Pointer A, Pointer B, Pointer C, float lrate);
    
    public static native int applyfwd(Pointer A, Pointer B, int ifn, int n);
    
    public static native int applyderiv(Pointer A, Pointer B, Pointer C, int ifn, int n);
    
    public static native int ADAGrad(int nrows, int ncols, Pointer mm, Pointer um, Pointer ssq, Pointer mask, int maskr, float nw, Pointer ve, int nve,
        Pointer ts, int nts, Pointer lrate, int nlrate, float langevin, float eps, int doupdate);
    
    public static native int ADAGradm(int nrows, int ncols, Pointer mm, Pointer um, Pointer ssq, Pointer momentum, float mu, Pointer mask, int maskr, float nw, Pointer ve, int nve,
            Pointer ts, int nts, Pointer lrate, int nlrate, float langevin, float eps, int doupdate);
    
    public static native int ADAGradn(int nrows, int ncols, Pointer mm, Pointer um, Pointer ssq, Pointer momentum, float mu, Pointer mask, int maskr, float nw, Pointer ve, int nve,
            Pointer ts, int nts, Pointer lrate, int nlrate, float langevin, float eps, int doupdate);
    
    public static native int LSTMfwd(Pointer inC, Pointer LIN1, Pointer LIN2, Pointer LIN3, Pointer LIN4, Pointer outC, Pointer outH, int n);
    
    public static native int LSTMbwd(Pointer inC, Pointer LIN1, Pointer LIN2, Pointer LIN3, Pointer LIN4, Pointer doutC, Pointer doutH, 
    		Pointer dinC, Pointer dLIN1, Pointer dLIN2, Pointer dLIN3, Pointer dLIN4, int n);
    
    public static native int pairembed(Pointer r1, Pointer r2, Pointer res, int n);
    
    public static native int pairMultTile(int nrows, int bncols, int brows1, int brows2, Pointer A, int lda, Pointer A2, int lda2,
    		   Pointer Bdata, Pointer Bir, Pointer Bjc, int broff, int bcoff, Pointer C, int ldc, int transpose);
    
    public static native int pairMultADAGradTile(int nrows, int bncols, int brows1, int brows2, Pointer A, int lda, int aroff, int acoff, 
 		   Pointer Bdata, Pointer Bir, Pointer Bjc, int broff, int bcoff, int transpose, 
 		   Pointer MM, int ldmm, Pointer Sumsq, Pointer Mask, int maskrows, Pointer lrate, int lrlen, Pointer vexp, int vexplen, Pointer texp, int texplen, 
 		   float istep, int addgrad, float epsilon);   
    



}
