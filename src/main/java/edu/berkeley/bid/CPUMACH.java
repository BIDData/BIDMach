package edu.berkeley.bid;

public final class CPUMACH {

  private CPUMACH() {}

  static {
      jcuda.LibUtils.loadLibrary("bidmachcpu");
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
}

