package edu.berkeley.bid;

public final class CPUMACH {

  private CPUMACH() {}

  static {
      LibUtils.loadLibrary("bidmachcpu", true);
  } 
 
  
  public static native void word2vecPos(int nrows, int ncols, int shift, int [] W, int [] LB, int [] UB, float [] A, float [] B, float lrate, float vexp, int nthreads);
 
  public static native void word2vecPosSlice(int nrows, int ncols, int shift, int [] W, int [] LB, int [] UB, float [] [] MM, float lrate, float vexp, int nthreads,
  		 int islice, int nslices, int maxCols, int nHead, int dualMode, int doHead);
  
  public static native void word2vecNeg(int nrows, int ncols, int nwa, int nwb, int [] WA, int [] WB, float [] A, float [] B, float lrate, float vexp, int nthreads);
 
  public static native void word2vecNegSlice(int nrows, int ncols, int nwa, int nwb, int [] WA, int [] WB, float [] [] MM, float lrate, float vexp, int nthreads,
  		int islice, int nslices, int maxCols, int nHead, int dualMode, int doHead);
  
  public static native void word2vecFwd(int nrows, int ncols, int nwa, int nwb, int [] WA, int [] WB, float [] A, float [] B, float [] C);
  
  public static native void word2vecBwd(int nrows, int ncols, int nwa, int nwb, int [] WA, int [] WB, float [] A, float [] B, float [] DA, float [] DB, float [] C, float lrate);

  public static native double word2vecEvalNeg(int nrows, int ncols, int nwa, int nwb, int [] WA, int [] WB, float [] A, float [] B, int nthreads);
  
  public static native double word2vecEvalPos(int nrows, int ncols, int shift, int [] W, int [] LB, int [] UB, float [] A, float [] B, int nthreads);

  public static native void testarrays(float [] [] a);
  
  public static native int applyfwd(float [] A, float [] B, int ifn, int n, int nthreads);
  
  public static native int applyderiv(float [] A, float [] B, float [] C, int ifn, int n, int nthreads);
  
  public static native void multADAGrad(int nrows, int ncols, int nnz, float [] A, float [] Bdata, int [] Bir, int [] Bjc, 
  		float [] MM, float [] Sumsq, float [] Mask, int maskrows, float [] lrate, int lrlen,
  		float [] vexp, int vexplen, float [] jtexp, int texplen, float istep, int addgrad, float epsilon, int biasv, int nbr);
  
  public static native void multADAGradTile(int nrows, int ncols, int y, int x, int nnz, float [] A, int lda, float [] Bdata, int [] Bir, int [] Bjc, 
	  		float [] MM, float [] Sumsq, float [] Mask, int maskrows, float [] lrate, int lrlen,
	  		float [] vexp, int vexplen, float [] jtexp, int texplen, float istep, int addgrad, float epsilon, int biasv, int nbr);
  
  public static native void pairMultADAGradTile(int nrows, int ncols, int bound1, int bound2, float [] A, int lda, int aroff, int acoff, 
      float [] Bdata, int [] jBir, int [] jBjc, int broff, int bcoff, float [] MM, int ldmm, float [] Sumsq, float [] Mask, int maskrows, 
      float [] lrate, int lrlen, float [] vexp, int vexplen, float [] texp, int texplen, float istep, int addgrad, float epsilon, int biasv, int nbr);
}

