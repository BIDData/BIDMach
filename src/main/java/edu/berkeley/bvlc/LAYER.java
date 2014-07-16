package edu.berkeley.bvlc;
import jcuda.*;

public final class LAYER {

    static {
        jcuda.LibUtils.loadLibrary("caffe");
    } 

    private LAYER() {}

    protected LAYER(long shptr) {
        _shptr = shptr;
    }

    public int num_blobs() {if (_shptr != 0) return num_blobs(_shptr); else throw new RuntimeException("Layer uninitialized");}

    public BLOB blob(int i) {if (_shptr != 0) return new BLOB(blob(_shptr, i)); else throw new RuntimeException("Layer uninitialized");}

    @Override
    protected void finalize() {
        if (_shptr != 0) clearLayer(_shptr);
        _shptr = 0;
    }

    private long _shptr = 0;

    private static native int num_blobs(long ref);

    private static native long blob(long ref, int i);

    private static native int clearLayer(long ref);

}
