package edu.berkeley.bvlc;

public final class LAYER {

    static {
        LibUtils.loadLibrary("caffe");
    } 

    private LAYER() {}

    protected LAYER(long shptr) {
        _shptr = shptr;
    }

    public int num_blobs() {if (_shptr != 0) return num_blobs(_shptr); else throw new RuntimeException("Layer uninitialized");}

    public BLOB blob(int i) {
    	if (_shptr == 0) {
    		throw new RuntimeException("Layer uninitialized");
    	} else {
    		int n = num_blobs();
    		if (i < 0 || i >= n) {
    			throw new RuntimeException("Layer blob index "+i+" out of range (0, "+(n-1)+")");
    		}
    		return new BLOB(blob(_shptr, i));
    	}
    }

    @Override
    protected void finalize() {
        if (_shptr != 0) clearLayer(_shptr);
        _shptr = 0;
    }

    private long _shptr = 0;

    private static native int num_blobs(long ref);

    private static native long blob(long ref, int i);

    private static native int clearLayer(long ref);
    
    private static native void pushMemoryData(long ref, float[] A, float[] B, int num, int channels, int height, int width);

}
