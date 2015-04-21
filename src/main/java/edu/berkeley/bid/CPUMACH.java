package edu.berkeley.bid;

public final class CPUMACH {

  private CPUMACH() {}

  static {
      jcuda.LibUtils.loadLibrary("bidmachcpu");
  } 
  
  public static native  void word2vecFwd(int nrows, int ncols, int nwa, int nwb, int [] WA, int [] WB, float [] A, float [] B, float [] C);
  
  public static native  void word2vecBwd(int nrows, int ncols, int nwa, int nwb, int [] WA, int [] WB, float [] A, float [] B, float [] DA, float [] DB, float [] C, float lrate);
  
  public static native  void word2vecNeg(int nrows, int ncols, int nwa, int nwb, int [] WA, int [] WB, float [] A, float [] B, float lrate, int nthreads);
  
  public static native int word2vecPos(int nrows, int ncols, int shift, int [] W, int [] LB, int [] UB, float [] A, float [] B, float lrate, int nthreads);

}

