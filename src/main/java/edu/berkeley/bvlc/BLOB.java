package edu.berkeley.bvlc;
import jcuda.*;

public final class BLOB {

    private BLOB() {}

    static {
        jcuda.LibUtils.loadLibrary("caffe");
    } 

    protected BLOB(long shptr) {
        _shptr = shptr;
    }

    public int count() {if (_shptr != 0) return count(_shptr); else throw new RuntimeException("Blob uninitialized");}

    public int num() {if (_shptr != 0) return num(_shptr); else throw new RuntimeException("Blob uninitialized");}

    public int height() {if (_shptr != 0) return height(_shptr); else throw new RuntimeException("Blob uninitialized");}

    public int width() {if (_shptr != 0) return width(_shptr); else throw new RuntimeException("Blob uninitialized");}

    public int channels() {if (_shptr != 0) return channels(_shptr); else throw new RuntimeException("Blob uninitialized");}

    public int offset(int n, int c, int h, int w) {if (_shptr != 0) return offset(_shptr, n, c, h, w); else throw new RuntimeException("Blob uninitialized");}
    
    public float [] get_data() {
        if (_shptr != 0) {
            int size = count(_shptr);
            float [] newdata = new float[size];
            get_data(_shptr, newdata, size);
            return newdata;
        } else {
        	throw new RuntimeException("Blob uninitialized");
        }
    }

    public void get_data(float [] newdata) {
        if (_shptr != 0) {
            int size = count(_shptr);
            if (size != newdata.length) {
                throw new RuntimeException("size mismatch in BLOB.getdata");
            }
            get_data(_shptr, newdata, size);
        } else {
        	throw new RuntimeException("Blob uninitialized");
        }
    }

    public void put_data(float [] newdata) {
        if (_shptr != 0) {
            int size = count(_shptr);
            if (size != newdata.length) {
                throw new RuntimeException("size mismatch in BLOB.putdata");
            }
            put_data(_shptr, newdata, size);
        } else {
        	throw new RuntimeException("Blob uninitialized");
        }
    }

    
    public float [] get_diff() {
        if (_shptr != 0) {
            int size = count(_shptr);
            float [] newdiff = new float[size];
            get_diff(_shptr, newdiff, size);
            return newdiff;
        } else {
        	throw new RuntimeException("Blob uninitialized");
        }
    }

    public void get_diff(float [] newdiff) {
        if (_shptr != 0) {
            int size = count(_shptr);
            if (size != newdiff.length) {
                throw new RuntimeException("size mismatch in BLOB.getdiff");
            }
            get_diff(_shptr, newdiff, size);
        } else {
        	throw new RuntimeException("Blob uninitialized");
        }
    }

    public void put_diff(float [] newdiff) {
        if (_shptr != 0) {
            int size = count(_shptr);
            if (size != newdiff.length) {
                throw new RuntimeException("size mismatch in BLOB.putdiff");
            }
            put_diff(_shptr, newdiff, size);
        } else {
        	throw new RuntimeException("Blob uninitialized");
        }
    }

    @Override
    protected void finalize() {
        if (_shptr != 0) clearBlob(_shptr);
        _shptr = 0;
    }

    private long _shptr = 0;

    private static native int count(long ref);

    private static native int num(long ref);

    private static native int height(long ref);

    private static native int width(long ref);

    private static native int channels(long ref);

    private static native int offset(long ref, int n, int c, int h, int w);

    private static native int get_data(long ref, float [] a, int n);

    private static native int put_data(long ref, float [] a, int n);

    private static native int get_diff(long ref, float [] a, int n);

    private static native int put_diff(long ref, float [] a, int n);

    private static native int clearBlob(long ref);
}
