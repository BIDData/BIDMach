package edu.berkeley.bvlc;
import jcuda.*;

public final class NET {

    static {
        jcuda.LibUtils.loadLibrary("caffe");
    } 

    private NET() {_shptr = 0;}

    public NET(String pfile) {_shptr = netFromParamFile(pfile);}

    public NET(String pfile, String prefilled) {_shptr = netFromPretrained(pfile, prefilled);}

    protected NET(long shptr) {_shptr = shptr;}
    
    public void init(String pfile) {
    	if (_shptr != 0) clearNet(_shptr);
    	_shptr = netFromParamFile(pfile);
    }
    
    public void init(String pfile, String prefilled) {
    	if (_shptr != 0) clearNet(_shptr);
    	_shptr = netFromPretrained(pfile, prefilled);
    }
    
    public int num_inputs() {if (_shptr != 0) return num_inputs(_shptr); else throw new RuntimeException("Net uninitialized");}

    public int num_outputs() {if (_shptr != 0) return num_outputs(_shptr); else throw new RuntimeException("Net uninitialized");}
    
    public int num_layers() {if (_shptr != 0) return num_layers(_shptr); else throw new RuntimeException("Net uninitialized");}

    public int num_blobs() {if (_shptr != 0) return num_blobs(_shptr); else throw new RuntimeException("Net uninitialized");}

    public BLOB blob(int i) {
    	if (_shptr == 0) {
    		throw new RuntimeException("Net uninitialized");
    	} else {
    		int n = num_blobs();
    		if (i < 0 || i >= n) {
    			throw new RuntimeException("Net blob index "+i+" out of range (0, "+(n-1)+")");
    		}
    		return new BLOB(blob(_shptr, i));
    	}
    }
    
    public BLOB input_blob(int i) {
    	if (_shptr == 0) {
    		throw new RuntimeException("Net uninitialized");
    	} else {
    		int n = num_inputs();
    		if (i < 0 || i >= n) {
    			throw new RuntimeException("Net input blob index "+i+" out of range (0, "+(n-1)+")");
    		}
    		return new BLOB(input_blob(_shptr, i));
    	}
    }
       
    public BLOB output_blob(int i) {
    	if (_shptr == 0) {
    		throw new RuntimeException("Net uninitialized");
    	} else {
    		int n = num_outputs();
    		if (i < 0 || i >= n) {
    			throw new RuntimeException("Net output blob index "+i+" out of range (0, "+(n-1)+")");
    		}
    		return new BLOB(output_blob(_shptr, i));
    	}
    }

    public LAYER layer(int i) {
    	if (_shptr == 0) {
    		throw new RuntimeException("Net uninitialized");
    	} else {
    		int n = num_layers();
    		if (i < 0 || i >= n) {
    			throw new RuntimeException("Net layer index "+i+" out of range (0, "+(n-1)+")");
    		}
    		return new LAYER(layer(_shptr, i));
    	}
    }

    public Object [] blob_names() {if (_shptr != 0) return blob_names(_shptr); else throw new RuntimeException("Net uninitialized");}

    public Object [] layer_names() {if (_shptr != 0) return layer_names(_shptr); else throw new RuntimeException("Net uninitialized");}

    public BLOB blob_by_name(String s) {if (_shptr != 0) return new BLOB(blob_by_name(_shptr, s)); else throw new RuntimeException("Net uninitialized");}

    public LAYER layer_by_name(String s) {if (_shptr != 0) return new LAYER(layer_by_name(_shptr, s)); else throw new RuntimeException("Net uninitialized");}

    public void forward() {if (_shptr != 0) forward(_shptr); else throw new RuntimeException("Net uninitialized");}

    public void backward() {if (_shptr != 0) backward(_shptr); else throw new RuntimeException("Net uninitialized");}

    @Override
    protected void finalize() {
        if (_shptr != 0) clearNet(_shptr);
    }

    private long _shptr;
    
    private static native long netFromParamFile(String name);

    private static native long netFromPretrained(String name, String prefilled);
    
    private static native int num_inputs(long ref);

    private static native int num_outputs(long ref);
    
    private static native int num_layers(long ref);

    private static native int num_blobs(long ref);

    private static native int forward(long ref);

    private static native int backward(long ref);

    private static native int clearNet(long ref);

    private static native long layer(long ref, int i);

    private static native long blob(long ref, int i);
    
    private static native long input_blob(long ref, int i);
    
    private static native long output_blob(long ref, int i);

    private static native Object [] blob_names(long ref);

    private static native Object [] layer_names(long ref);

    private static native long blob_by_name(long ref, String s);

    private static native long layer_by_name(long ref, String s);

}
