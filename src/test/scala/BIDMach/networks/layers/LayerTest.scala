package BIDMach.networks.layers

import BIDMat.{Mat,DMat,FMat,FFilter,GMat,GDMat,GFilter}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.BIDMachSpec;
import BIDMach.networks.Net;
import org.scalatest._;
import org.scalatest.junit._;
import org.scalatest.prop._;
import org.junit.runner.RunWith

@RunWith(classOf[JUnitRunner])
class LayerTest extends BIDMachSpec {

  val w = 10;
  val h = 20;
  val k = 25;
  val c = 30;  
  val n = 40;
  val useGPU = true;
  val matType = if (useGPU) "GMat" else "FMat";

  override def beforeAll {
	Mat.checkMKL(false)
	if (useGPU) Mat.checkCUDA(true)
  }

  def checkSimilar(a0:DMat, b0:DMat, eps:Float):Unit = {
    val a = DMat(a0);
    val b = DMat(b0);
  	a.dims.length should equal (b.dims.length) ;
  	a.dims.data should equal (b.dims.data);
  	assert_approx_eq_double(a.data, b.data, eps);
  }
  
  def checkSimilar(a:DMat, b:DMat):Unit = checkSimilar(a, b, 1e-4f);
  
  def checkSimilar(a0:FMat, b0:FMat, eps:Float) = {
    val a = FMat(a0);
    val b = FMat(b0);
  	a.dims.length should equal (b.dims.length) ;
  	a.dims.data should equal (b.dims.data);
  	assert_approx_eq(a.data, b.data, eps);
  }
  
  def checkSimilar(a:FMat, b:FMat):Unit = checkSimilar(a, b, 1e-4f);
  
  val inmat1 = rand(w\h\c\n);
  val inmat2 = rand(w\h\c\n);
  val inmat3 = rand(h\k\c\n);
  val inmat4 = rand(k\h\c\n);
  val inmat5 = rand(h\w\c\n);

  val inderiv1 = rand(w\h\c\n);
  val inderiv2 = rand(w\h\c\n);
  val inderiv3 = rand(h\k\c\n);
  val inderiv4 = rand(k\h\c\n);
  val inderiv5 = rand(h\w\c\n);

  val dout = rand(w\h\c\n);
  val dout1 = rand(w\k\c\n);
  
  val input1 = Layer.input;
  val input2 = Layer.input;
  val input3 = Layer.input;
  val input4 = Layer.input;
  val input5 = Layer.input;
  
  if (useGPU) { 
    input1.output = GMat(inmat1);
    input2.output = GMat(inmat2);
    input3.output = GMat(inmat3);
    input4.output = GMat(inmat4);
    input5.output = GMat(inmat5);

    input1.deriv = GMat(inderiv1);
    input2.deriv = GMat(inderiv2);
    input3.deriv = GMat(inderiv3);
    input4.deriv = GMat(inderiv4);
    input5.deriv = GMat(inderiv5);
  } else { 
    input1.output = inmat1;
    input2.output = inmat2;
    input3.output = inmat3;
    input4.output = inmat4;
    input5.output = inmat5;

    input1.deriv = inderiv1.copy;
    input2.deriv = inderiv2.copy;
    input3.deriv = inderiv3.copy;  
    input4.deriv = inderiv4.copy;
    input5.deriv = inderiv5.copy;
  }

  "A Layer" should "support forward AddLayer computation" in {
    
    val net = new Net;
    val aa = new AddLayer(net){inputs(0) = input1; inputs(1) = input2;}
    net.layers = Array(input1, input2, aa);
    net.forward;
    
    val cc = zeros(w\h\c\n);
    for (i <- 0 until cc.length) {
      cc.data(i) = inmat1.data(i) + inmat2.data(i);
    }	
    val out = aa.output.asInstanceOf[FMat]
	out.mytype should equal (matType);
    checkSimilar(cc, out); 
  }

  it should "support backward AddLayer computation" in {
    
    val net = new Net;
    val aa = new AddLayer(net){inputs(0) = input1; inputs(1) = input2;}
    net.layers = Array(input1, input2, aa);
    net.forward;
    
    aa.deriv should not be (null);
    
    aa.deriv <-- dout;
    input1.deriv <-- inderiv1;
    input2.deriv <-- inderiv2;
    
    net.backward(0,0);
    
    val input1d = input1.deriv.asInstanceOf[FMat];
    val input2d = input2.deriv.asInstanceOf[FMat];
	input1d.mytype should equal (matType);
	input2d.mytype should equal (matType);
    checkSimilar(dout + inderiv1, input1d);
    checkSimilar(dout + inderiv2, input2d);
  }  

  it should "support forward SubLayer computation" in {
    
    val net = new Net;
    val aa = new SubLayer(net){inputs(0) = input1; inputs(1) = input2;}
    net.layers = Array(input1, input2, aa);
    net.forward;
    
    val cc = zeros(w\h\c\n);
    for (i <- 0 until cc.length) {
      cc.data(i) = inmat1.data(i) - inmat2.data(i);
    }	
    val out = aa.output.asInstanceOf[FMat]
	out.mytype should equal (matType);
    checkSimilar(cc, out);
  }

  it should "support backward SubLayer computation" in {
    
    val net = new Net;
    val aa = new SubLayer(net){inputs(0) = input1; inputs(1) = input2;}
    net.layers = Array(input1, input2, aa);
    net.forward;
    
    aa.deriv should not be (null);
    
    aa.deriv <-- dout;
    input1.deriv <-- inderiv1;
    input2.deriv <-- inderiv2;
    
    net.backward(0,0);

    val input1d = input1.deriv.asInstanceOf[FMat];
    val input2d = input2.deriv.asInstanceOf[FMat];
	input1d.mytype should equal (matType);
	input2d.mytype should equal (matType);
    
    checkSimilar(dout + inderiv1, input1d);
    checkSimilar(-dout + inderiv2, input2d);
  }  

  it should "support forward MulLayer computation" in {
    
    val net = new Net;
    val aa = new MulLayer(net){inputs(0) = input1; inputs(1) = input2;}
    net.layers = Array(input1, input2, aa);
    net.forward;
    
    val cc = zeros(w\h\c\n);
    for (i <- 0 until cc.length) {
      cc.data(i) = inmat1.data(i) * inmat2.data(i);
    }	
    val out = aa.output.asInstanceOf[FMat]
	out.mytype should equal (matType);
    checkSimilar(cc, out);
  }

  it should "support backward MulLayer computation" in {
    
    val net = new Net;
    val aa = new MulLayer(net){inputs(0) = input1; inputs(1) = input2;}
    net.layers = Array(input1, input2, aa);
    net.forward;
    
    aa.deriv should not be (null);
    
    aa.deriv <-- dout;
    input1.deriv <-- inderiv1;
    input2.deriv <-- inderiv2;
    
    net.backward(0,0);
    
    val input1d = input1.deriv.asInstanceOf[FMat];
    val input2d = input2.deriv.asInstanceOf[FMat];
	input1d.mytype should equal (matType);
	input2d.mytype should equal (matType);
    checkSimilar(dout *@ inmat2 + inderiv1, input1d);
    checkSimilar(dout *@ inmat1 + inderiv2, input2d);
  }  

  it should "support forward DivLayer computation" in {
    
    val net = new Net;
    val aa = new DivLayer(net){inputs(0) = input1; inputs(1) = input2;}
    net.layers = Array(input1, input2, aa);
    net.forward;
    
    val cc = zeros(w\h\c\n);
    for (i <- 0 until cc.length) {
      cc.data(i) = inmat1.data(i) / inmat2.data(i);
    }	
    val out = aa.output.asInstanceOf[FMat]
	out.mytype should equal (matType);
    checkSimilar(cc, out);
  }

  it should "support backward DivLayer computation" in {
    
    val net = new Net;
    val aa = new DivLayer(net){inputs(0) = input1; inputs(1) = input2;}
    net.layers = Array(input1, input2, aa);
    net.forward;
    
    aa.deriv should not be (null);
    
    aa.deriv <-- dout;
    input1.deriv <-- inderiv1;
    input2.deriv <-- inderiv2;
    
    net.backward(0,0);
    
    val input1d = input1.deriv.asInstanceOf[FMat];
    val input2d = input2.deriv.asInstanceOf[FMat];
	input1d.mytype should equal (matType);
	input2d.mytype should equal (matType);
    checkSimilar(dout / inmat2 + inderiv1, input1d, 1e-3f);
    checkSimilar(- (dout / inmat2) *@  (inmat1 / inmat2) + inderiv2, input2d, 1e-3f);
  }  

  it should "support forward MatMulLayer *" in {
    
    val net = new Net;
    val aa = new MatMulLayer(net){inputs(0) = input1; inputs(1) = input3;}
    net.layers = Array(input1, input3, aa);
    net.forward;
    
    val cc = zeros(w\k\c\n);
    for (i <- 0 until c) { 
      for (j <- 0 until n) { 
        cc(?,?,i,j) = (inmat1(?,?,i,j).reshapeView(w,h) * inmat3(?,?,i,j).reshapeView(h,k))reshapeView(w,k,1,1)
      }
    }	
    val out = aa.output.asInstanceOf[FMat]
	out.mytype should equal (matType);
    checkSimilar(cc, out);
  }

  it should "support backward MatMulLayer *" in {
    
    val net = new Net;
    val aa = new MatMulLayer(net){inputs(0) = input1; inputs(1) = input3;}
    net.layers = Array(input1, input3, aa);
    net.forward;
    
    aa.deriv should not be (null);
    
    aa.deriv <-- dout1;
    input1.deriv <-- inderiv1;
    input3.deriv <-- inderiv3;
    
    val ind1 = inderiv1.copy;
    val ind3 = inderiv3.copy;
    net.backward(0,0);
    val cc = zeros(w\k\c\n);
    for (i <- 0 until c) { 
      for (j <- 0 until n) { 
        val d1 = dout1(?,?,i,j).reshapeView(w,k);
        val i1 = inmat1(?,?,i,j).reshapeView(w,h);
        val i3 = inmat3(?,?,i,j).reshapeView(h,k);
        ind1(?,?,i,j) = ind1(?,?,i,j) + (d1 * i3.t).reshapeView(w,h,1,1)
        ind3(?,?,i,j) = ind3(?,?,i,j) + (i1.t * d1).reshapeView(h,k,1,1)
      }
    }	
    val input1d = input1.deriv.asInstanceOf[FMat];
    val input3d = input3.deriv.asInstanceOf[FMat];
	input1d.mytype should equal (matType);
	input3d.mytype should equal (matType);
    
    checkSimilar(ind1, input1d);
    checkSimilar(ind3, input3d);
  }  

  it should "support forward MatMulLayer *^" in {
    
    val net = new Net;
    val aa = new MatMulLayer(net){inputs(0) = input1; inputs(1) = input4; opts.transB=true;}
    net.layers = Array(input1, input4, aa);
    net.forward;
    
    val cc = zeros(w\k\c\n);
    for (i <- 0 until c) { 
      for (j <- 0 until n) { 
        cc(?,?,i,j) = (inmat1(?,?,i,j).reshapeView(w,h) * inmat4(?,?,i,j).reshapeView(k,h).t)reshapeView(w,k,1,1)
      }
    }	
    val out = aa.output.asInstanceOf[FMat]
	out.mytype should equal (matType);
    checkSimilar(cc, out);
  }

  it should "support backward MatMulLayer *^" in {
    
    val net = new Net;
    val aa = new MatMulLayer(net){inputs(0) = input1; inputs(1) = input4; opts.transB=true;}
    net.layers = Array(input1, input4, aa);
    net.forward;
    
    aa.deriv should not be (null);
    
    aa.deriv <-- dout1;
    input1.deriv <-- inderiv1;
    input4.deriv <-- inderiv4;
    
    val ind1 = inderiv1.copy;
    val ind4 = inderiv4.copy;
    net.backward(0,0);
    val cc = zeros(w\k\c\n);
    for (i <- 0 until c) { 
      for (j <- 0 until n) { 
        val d1 = dout1(?,?,i,j).reshapeView(w,k);
        val i1 = inmat1(?,?,i,j).reshapeView(w,h);
        val i4 = inmat4(?,?,i,j).reshapeView(k,h);
        ind1(?,?,i,j) = ind1(?,?,i,j) + (d1 * i4).reshapeView(w,h,1,1)
        ind4(?,?,i,j) = ind4(?,?,i,j) + (d1.t * i1).reshapeView(k,h,1,1)
      }
    }	
    
    val input1d = input1.deriv.asInstanceOf[FMat];
    val input4d = input4.deriv.asInstanceOf[FMat];
	input1d.mytype should equal (matType);
	input4d.mytype should equal (matType);

    checkSimilar(ind1, input1d);
    checkSimilar(ind4, input4d);
  }  

  it should "support forward MatMulLayer ^*" in {
    
    val net = new Net;
    val aa = new MatMulLayer(net){inputs(0) = input5; inputs(1) = input3; opts.transA=true;}
    net.layers = Array(input5, input3, aa);
    net.forward;
    
    val cc = zeros(w\k\c\n);
    for (i <- 0 until c) { 
      for (j <- 0 until n) { 
        cc(?,?,i,j) = (inmat5(?,?,i,j).reshapeView(h,w).t * inmat3(?,?,i,j).reshapeView(h,k)).reshapeView(w,k,1,1)
      }
    }	
    val out = aa.output.asInstanceOf[FMat]
	out.mytype should equal (matType);
    checkSimilar(cc, out);
  }

  it should "support backward MatMulLayer ^*" in {
    
    val net = new Net;
    val aa = new MatMulLayer(net){inputs(0) = input5; inputs(1) = input3; opts.transA=true;}
    net.layers = Array(input5, input3, aa);
    net.forward;
    
    aa.deriv should not be (null);
    
    aa.deriv <-- dout1;
    input5.deriv <-- inderiv5;
    input3.deriv <-- inderiv3;
    
    val ind5 = inderiv5.copy;
    val ind3 = inderiv3.copy;
    net.backward(0,0);
    val cc = zeros(w\k\c\n);
    for (i <- 0 until c) { 
      for (j <- 0 until n) { 
        val d1 = dout1(?,?,i,j).reshapeView(w,k);
        val i5 = inmat5(?,?,i,j).reshapeView(h,w);
        val i3 = inmat3(?,?,i,j).reshapeView(h,k);
        ind5(?,?,i,j) = ind5(?,?,i,j) + (i3 * d1.t).reshapeView(h,w,1,1)
        ind3(?,?,i,j) = ind3(?,?,i,j) + (i5 * d1).reshapeView(h,k,1,1)
      }
    }	
    
    val input5d = input5.deriv.asInstanceOf[FMat];
    val input3d = input3.deriv.asInstanceOf[FMat];
	input5d.mytype should equal (matType);
	input3d.mytype should equal (matType);

    checkSimilar(ind5, input5d);
    checkSimilar(ind3, input3d);
  }  

}
