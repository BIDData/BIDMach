package edu.berkeley.bvlc;

public final class SGDSOLVER {

    static {
        LibUtils.loadLibrary("caffe");
    } 

    public SGDSOLVER(String pfile) {
        _sptr = fromParams(pfile); 
        _net = new NET(net(_sptr));
    }

    public NET net() {return _net;}

    public void Solve() {if (_sptr != 0) Solve(_sptr);}

    public void SolveResume(String s) {if (_sptr != 0) SolveResume(_sptr, s);}

    @Override
    protected void finalize() {
        if (_sptr != 0) clearSGDSolver(_sptr);
    }

    private final long _sptr;

    private final NET _net;
    
    private static native long fromParams(String name);

    private static native long net(long n);

    private static native void Solve(long n);

    private static native void SolveResume(long n, String s);

    private static native void clearSGDSolver(long ref);

}
