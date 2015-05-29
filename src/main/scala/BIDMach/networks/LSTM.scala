package BIDMach.networks

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach._
import scala.util.hashing.MurmurHash3;
import scala.collection.mutable.HashMap;

/**
 * LSTM. 
 */

class LSTMLayer(override val net:Net, override val spec:LSTMLayer.Spec = new LSTMLayer.Spec) extends ModelLayer(net, spec) {
	override val inputs = new Array[Layer](3);
	override val outputs = new Array[Mat](2);
	override val derivs = new Array[Mat](2);
	
	var internal_layers:Array[Layer] = null;
	
	override def forward = {
	  internal_layers.map(_.forward);
	}
	
	override def backward(ipass:Int, pos:Long) = {
	  internal_layers.reverse.map(_.backward(ipass, pos));
	}

	def construct = {
		internal_layers = new Array[Layer](spec.lspecs.length);
	  for (i <- 0 until internal_layers.length) {
	  	internal_layers(i) = spec.lspecs(i).create(net);
	  	spec.lspecs(i).myLayer = internal_layers(i)
	  }
	  for (i <- 0 until internal_layers.length) {
	  	for (j <- 0 until spec.lspecs(i).inputs.length) {
    		if (spec.lspecs(i).inputs(j) != null) internal_layers(i).inputs(j) = spec.lspecs(i).inputs(j).myLayer;
    	}
	  }
	}
}

object LSTMLayer {
	class Spec extends ModelLayer.Spec {
	  
	  var prefix = "";	
	  var i:Layer.Spec = null;
	  var prev_h:Layer.Spec = null;
	  var prev_c:Layer.Spec = null;
	  
	  var next_h:Layer.Spec = null;
	  var next_c:Layer.Spec = null;
	  
	  var lspecs:Array[Layer.Spec] = null;
	  
	  def constructSpec = {
	  	val il1 = new LinLayer.Spec{inputs(0) = i; modelName = prefix + "LSTM_il1"};
	  	val ph1 = new LinLayer.Spec{inputs(0) = prev_h; modelName = prefix + "LSTM_ph1"};
	  	val sum1 = new AddLayer.Spec{inputs(0) = il1; inputs(1) = ph1};
	  	val in_gate = new SigmoidLayer.Spec{inputs(0) = sum1};
	  	
	  	val il2 = new LinLayer.Spec{inputs(0) = i; modelName = prefix + "LSTM_il2"};
	  	val ph2 = new LinLayer.Spec{inputs(0) = prev_h; modelName = prefix + "LSTM_ph12"};
	  	val sum2 = new AddLayer.Spec{inputs(0) = il2; inputs(1) = ph2};
	  	val out_gate = new SigmoidLayer.Spec{inputs(0) = sum2};
	  	
	  	val il3 = new LinLayer.Spec{inputs(0) = i; modelName = prefix + "LSTM_il3"};
	  	val ph3 = new LinLayer.Spec{inputs(0) = prev_h; modelName = prefix + "LSTM_ph13"};
	  	val sum3 = new AddLayer.Spec{inputs(0) = il3; inputs(1) = ph3};
	  	val forget_gate = new SigmoidLayer.Spec{inputs(0) = sum3};
	  	
	    val il4 = new LinLayer.Spec{inputs(0) = i; modelName = prefix + "LSTM_il4"};
	  	val ph4 = new LinLayer.Spec{inputs(0) = prev_h; modelName = prefix + "LSTM_ph14"};
	  	val sum4 = new AddLayer.Spec{inputs(0) = il4; inputs(1) = ph4};
	  	val in_gate2 = new TanhLayer.Spec{inputs(0) = sum4};
	  	
	  	val in_prod = new MulLayer.Spec{inputs(0) = in_gate; inputs(1) = in_gate2};
	  	val f_prod = new MulLayer.Spec{inputs(0) = forget_gate; inputs(1) = prev_c};
	  	next_c = new AddLayer.Spec{inputs(0) = in_prod; inputs(1) = f_prod};
	  	
	  	val next_tanh = new TanhLayer.Spec{inputs(0) = next_c;};
	  	next_h = new MulLayer.Spec{inputs(0) = out_gate; inputs(1) = next_tanh};
	  	
	  	lspecs = Array(il1, ph1, sum1, in_gate, il2, ph2, sum2, out_gate, il3, ph3, sum3, forget_gate, 
	  			           il4, sum4, in_gate2, in_prod, f_prod, next_c, next_tanh, next_h);
	  	lspecs.map(_.parent = this);
	  }
	  

	  override def clone:Spec = {
		  copyTo(new Spec).asInstanceOf[Spec];
	  }

	  override def create(net:Net):LSTMLayer = {
		  apply(net, this);
	  }
	}
  
  def apply(net:Net) = new LSTMLayer(net, new Spec);
  
  def apply(net:Net, spec:Spec) = {
    val x = new LSTMLayer(net, spec);
    x.construct;
    x;
  }
}





