package edu.berkeley.bvlc;

public final class CAFFE {

    private CAFFE() {}

    static {
        LibUtils.loadLibrary("caffe");
    } 
    
    public static native void set_mode(int mode);

    public static native void init(int logtostderr, int stderrthreshold, int minloglevel, String jlogdir);
    
    public static native void set_phase(int phase);
    
    public static native int get_mode();
    
    public static native int get_phase();
    
    public static native void set_device(int n);

    public static native void DeviceQuery();

}
