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
    
    public static native int treeprod(Pointer trees, Pointer feats, Pointer tpos, Pointer otvs, int nrows, int ncols, int ns, int tstride, int ntrees);

    public static native int treesteps(Pointer trees, Pointer feats, Pointer tpos, Pointer otpos, int nrows, int ncols, int ns, int tstride, int ntrees, int tdepth);
    
    public static native int veccmp(Pointer A, Pointer B, Pointer C);
    
    public static native int hammingdists(Pointer A, Pointer B, Pointer W, Pointer OP, Pointer OW, int n);

    public static native int treePack(Pointer id, Pointer tn, Pointer icats, Pointer jc, Pointer out, Pointer fl, int nrows, int ncols, int ntrees, int nsamps);


}
