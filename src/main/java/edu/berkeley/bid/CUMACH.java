package edu.berkeley.bid;
import jcuda.*;

public final class CUMACH {

    private CUMACH() {}

    static {
        jcuda.LibUtils.loadLibrary("bidmachcuda");
    } 
    
    public static native int applylinks(Pointer A, Pointer L, Pointer C, int nrows, int ncols);
    
    public static native int applypreds(Pointer A, Pointer L, Pointer C, int nrows, int ncols);
    
    public static native int applylls(Pointer A, Pointer B, Pointer L, Pointer C, int nrows, int ncols);
    
    public static native int applyderivs(Pointer A, Pointer B, Pointer L, Pointer C, int nrows, int ncols);
    
    public static native int LDAgibbs(int nr, int nnz, Pointer A, Pointer B, Pointer AN, Pointer BN, Pointer Cir, Pointer Cic, Pointer P, float nsamps);

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
    
    public static native int hashMult(int nrows, int nfeats, int ncols, Pointer jA, Pointer jBdata, Pointer jBir, Pointer jBjc, Pointer jC, int transpose);
    
    public static native int hashCross(int nrows, int nfeats, int ncols, Pointer jA, Pointer jBdata, Pointer jBir, Pointer jBjc, Pointer jCdata, Pointer jCir, Pointer jCjc, Pointer D, int transpose);
    
}
