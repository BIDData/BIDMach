package edu.berkeley.bid;

public final class CPUMACH {

  private CPUMACH() {}

  static {
      jcuda.LibUtils.loadLibrary("bidmachcpu");
  } 
  
  public static native  void word2vecFwd(int nrows, int ncols, int nwa, int nwb, int [] WA, int [] WB, float [] A, float [] B, float [] C);
  
  public static native  void word2vecBwd(int nrows, int ncols, int nwa, int nwb, int [] WA, int [] WB, float [] A, float [] B, float [] DA, float [] DB, float [] C, float lrate);
  
  public static native  void word2vec(int nrows, int ncols, int nwa, int nwb, int [] WA, int [] WB, float [] A, float [] B, float lrate);

}

