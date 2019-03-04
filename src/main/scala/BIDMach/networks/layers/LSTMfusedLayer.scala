package BIDMach.networks.layers

import BIDMat.{Mat,ND,SBMat,CMat,CSMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach.networks._
import BIDMach._
import scala.util.hashing.MurmurHash3;
import scala.collection.mutable.HashMap;
import edu.berkeley.bid.CUMACH;

/**
 * LSTM unit 
 */

@SerialVersionUID(100L)
class LSTMfusedLayer(override val net:Net, override val opts:LSTMfusedNodeOpts = new LSTMfusedNode) extends Layer(net, opts) {
	override val _inputs = new Array[LayerTerm](5);
	override val _outputs = new Array[Mat](2);
	override val _derivs = new Array[Mat](2);
  
  override def toString = {
    "LSTMcoa@"+Integer.toHexString(hashCode % 0x10000).toString
  }
  
  override def forward = {
	  inplaceNoConnectGetOutput();
	  (inputData(0), inputData(1), inputData(2), inputData(3), inputData(4), outputs(0), outputs(1)) match {
	    case (i0:GMat, i1:GMat, i2:GMat, i3:GMat, i4:GMat, out0:GMat, out1:GMat) => {
	      CUMACH.LSTMfwd(i0.pdata, i1.pdata, i2.pdata, i3.pdata, i4.pdata, out0.pdata, out1.pdata, i0.length);
	    }
	    case (i0:FMat, i1:FMat, i2:FMat, i3:FMat, i4:FMat, out0:FMat, out1:FMat) => {
	      LSTMfusedLayer.LSTMforward(i0, i1, i2, i3, i4, out0, out1);
	    }
        case _ => throw new RuntimeException("LSTMFusedLayer forward matrix type not matched");
	  }
	}
  
  override def backward = {
  	inplaceNoConnectGetInputDerivs();
  		
	  (inputData(0), inputData(1), inputData(2), inputData(3), inputData(4), deriv(0), deriv(1), inputDeriv(0), inputDeriv(1), inputDeriv(2), inputDeriv(3), inputDeriv(4)) match {
	    case (inC:GMat, lin1:GMat, lin2:GMat, lin3:GMat, lin4:GMat, doutC:GMat, doutH:GMat, dinC:GMat, dlin1:GMat, dlin2:GMat, dlin3:GMat, dlin4:GMat) => {
	      CUMACH.LSTMbwd(inC.pdata, lin1.pdata, lin2.pdata, lin3.pdata, lin4.pdata, doutC.pdata, doutH.pdata, dinC.pdata, dlin1.pdata, dlin2.pdata, dlin3.pdata, dlin4.pdata, inC.length);	      
	    }
	    case (inC:FMat, lin1:FMat, lin2:FMat, lin3:FMat, lin4:FMat, doutC:FMat, doutH:FMat, dinC:FMat, dlin1:FMat, dlin2:FMat, dlin3:FMat, dlin4:FMat) => {
	      LSTMfusedLayer.LSTMbackward(inC, lin1, lin2, lin3, lin4, doutC, doutH, dinC, dlin1, dlin2, dlin3, dlin4);	      
	    }
        case _ => throw new RuntimeException("LSTMFusedLayer backward matrix type not matched");
	  }
	  inplaceNoConnectReleaseDeriv();
	}
  
}

trait LSTMfusedNodeOpts extends NodeOpts {
    
   def copyOpts(opts:LSTMfusedNodeOpts):LSTMfusedNodeOpts = {
  		super.copyOpts(opts);
  		opts;
    }
}

@SerialVersionUID(100L)
class LSTMfusedNode extends Node with LSTMfusedNodeOpts {	
  
	  override val inputs:Array[NodeTerm] = Array(null, null, null, null, null);
}

@SerialVersionUID(100L)
object LSTMfusedLayer {  
  
  def apply(net:Net) = new LSTMfusedLayer(net, new LSTMfusedNode);
  
  def apply(net:Net, opts:LSTMfusedNodeOpts) = new LSTMfusedLayer(net, opts);
  
  @inline def sigmoid(a:Float):Float = {
  		if (a > 20.0f) {
  			return 1.0f;
  		} else if (a < -80.0f) {
  			return 0.0f;
  		} else {
  			return 1.0f/(1.0f + math.exp(-a).toFloat);
  		}
  }
  
  @inline def tanh(a:Float):Float = {
    math.tanh(a).toFloat
  }
  
  @inline def deriv_sigmoid(a:Float, d:Float):Float = {
  	d * (a - a * a);
  }
  
  @inline def deriv_tanh(a:Float, d:Float):Float = {
  	d * (1.0f - a * a);
  }


  
  def LSTMforward(incMat:FMat, lin1Mat:FMat, lin2Mat:FMat, lin3Mat:FMat, lin4Mat:FMat, outCMat:FMat, outHMat:FMat) {
    val n = incMat.length;
    val incArr = incMat.data;
    val lin1Arr = lin1Mat.data;
    val lin2Arr = lin2Mat.data;
    val lin3Arr = lin3Mat.data;
    val lin4Arr = lin4Mat.data;
    val outCArr = outCMat.data;
    val outHArr = outHMat.data;
    var i = 0;
    while (i < n) {
      val in_c = incArr(i);
      val lin1 = lin1Arr(i);
      val lin2 = lin2Arr(i);
      val lin3 = lin3Arr(i);
      val lin4 = lin4Arr(i);
      
      val in_gate = sigmoid(lin1);
      val out_gate = sigmoid(lin2);
      val forget_gate = sigmoid(lin3);
      val in_sat = tanh(lin4);

      val in_prod = in_gate * in_sat;
      val f_prod = forget_gate * in_c;
      val out_c = in_prod + f_prod;

      val out_tanh = tanh(out_c);
      val out_h = out_gate * out_tanh;

      outCArr(i) = out_c;
      outHArr(i)= out_h;  

      i += 1;
    }
  }
  
   def LSTMbackward(incMat:FMat, lin1Mat:FMat, lin2Mat:FMat, lin3Mat:FMat, lin4Mat:FMat, doutCMat:FMat, doutHMat:FMat,
       dincMat:FMat, dlin1Mat:FMat, dlin2Mat:FMat, dlin3Mat:FMat, dlin4Mat:FMat) {
    val n = incMat.length;
    val incArr = incMat.data;
    val lin1Arr = lin1Mat.data;
    val lin2Arr = lin2Mat.data;
    val lin3Arr = lin3Mat.data;
    val lin4Arr = lin4Mat.data;
    val doutCArr = doutCMat.data;
    val doutHArr = doutHMat.data;
    val dincArr = dincMat.data;
    val dlin1Arr = dlin1Mat.data;
    val dlin2Arr = dlin2Mat.data;
    val dlin3Arr = dlin3Mat.data;
    val dlin4Arr = dlin4Mat.data;
    var i = 0;
    while (i < n) {
      val in_c = incArr(i);
      val lin1 = lin1Arr(i);
      val lin2 = lin2Arr(i);
      val lin3 = lin3Arr(i);
      val lin4 = lin4Arr(i);
      
      val in_gate = sigmoid(lin1);
      val out_gate = sigmoid(lin2);
      val forget_gate = sigmoid(lin3);
      val in_sat = tanh(lin4);

      val in_prod = in_gate * in_sat;
      val f_prod = forget_gate * in_c;
      val out_c = in_prod + f_prod;

      val out_tanh = tanh(out_c);

      val dout_h = doutHArr(i);
      var dout_c = doutCArr(i);

    //    out_h = out_gate * out_tanh;
      val dout_gate = dout_h * out_tanh;
      val dout_tanh = dout_h * out_gate;

    //    out_tanh = tanh(out_c);
      dout_c += deriv_tanh(out_tanh, dout_tanh);

    //    out_c = in_prod + f_prod;
      val din_prod = dout_c;
      val df_prod = dout_c;

    //    f_prod = forget_gate * in_c;
      val dforget_gate = df_prod * in_c;
      val din_c = df_prod * forget_gate;

    //    in_prod = in_gate * in_sat;
      val din_gate = din_prod * in_sat;
      val din_sat = din_prod * in_gate;

    //    in_gate = forward_sigmoid(lin1);
    //    out_gate = forward_sigmoid(lin2);
    //    forget_gate = forward_sigmoid(lin3);
    //    in_sat = tanh(lin4);

      val dlin4 = deriv_tanh(in_sat, din_sat);
      val dlin3 = deriv_sigmoid(forget_gate, dforget_gate);
      val dlin2 = deriv_sigmoid(out_gate, dout_gate);
      val dlin1 = deriv_sigmoid(in_gate, din_gate);

      dlin4Arr(i) += dlin4;
      dlin3Arr(i) += dlin3;
      dlin2Arr(i) += dlin2;
      dlin1Arr(i) += dlin1;
      dincArr(i) += din_c;

      i += 1;
    }
  }
}
    
