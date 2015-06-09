package BIDMach.networks

import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach._
import scala.util.hashing.MurmurHash3;
import scala.collection.mutable.HashMap;

/*
 * LSTM next Word prediction model, which comprises a rectangular array of LSTM compound layers.
 */
class LSTMnextWord(override val opts:LSTMnextWord.Opts = new LSTMnextWord.Options) extends Net(opts) {
  
  var dummyword:Mat = null;
	
	override def createLayers = {
	  val net = new LSTMnextWord(opts);
	  val lopts = opts.lopts;
	  val height = opts.height;
	  val width = opts.width;
	  
    net.layers = new Array[Layer]((height+4)*width);
    lopts.constructNet;
    val lopts1 = new LinLayer.Options{modelName = "inWordMap"; outdim = opts.dim};
    val lopts2 = new LinLayer.Options{modelName = "outWordMap"; outdim = opts.nvocab};
    val sopts = new SoftmaxLayer.Options;
    for (j <- 0 until width) {
    	net.layers(j) = InputLayer(net);
    	net.layers(j + width) = LinLayer(net, lopts1);
    }
    for (i <- 2 until height + 2) {
      for (j <- 0 until width) {
        val layer = LSTMLayer(net, lopts);
        layer.setinput(2, net.layers(j + (i - 1) * width));
        if (j > 0) {
          layer.setinput(0, net.layers(j - 1 + i * width));
          layer.setinout(1, net.layers(j - 1 + i * width), 1);
        }
        net.layers(j + i * width) = layer;
      }
      net.layers(i * width).setinput(0, net.layers(i * width + width - 1));
      net.layers(i * width).setinout(1, net.layers(i * width + width - 1), 1);
    }
    for (j <- 0 until width) {
    	val linlayer = LinLayer(net, lopts2); 
    	linlayer.setinput(0, net.layers(j + (height + 1) * width));
    	net.layers(j + (height + 2) * width) = linlayer;
    	
    	val smlayer = SoftmaxLayer(net, sopts);
    	smlayer.setinput(0, linlayer);
    	net.layers(j + (height + 3) * width) = smlayer;
    }
  }
  
  override def assignInputs(gmats:Array[Mat], ipass:Int, pos:Long) {
    if (batchSize % opts.width != 0) throw new RuntimeException("LSTMwordPredict error: batch size must be a multiple of network width")
    val nr = batchSize / opts.width;
    val in = gmats(0).view(nr, opts.width).t;
    for (i <- 0 until opts.width) {
      layers(i).output = in.colslice(i,i+1);
    }	
  }
  
  override def assignTargets(gmats:Array[Mat], ipass:Int, pos:Long) {
  	val nr = batchSize / opts.width;
  	if (dummyword.asInstanceOf[AnyRef] == null) dummyword = gmats(0).zeros(1,1);
  	val in0 = gmats(0);
  	val inshift = in0(0,1->(in0.ncols)) \ dummyword;
    val in = inshift.view(nr, opts.width).t;
    for (i <- 0 until opts.width) {
      val incol = in.colslice(i,i+1);
      layers(i + opts.width * (opts.height + 3)).target = 
      		if (targmap.asInstanceOf[AnyRef] != null) targmap * incol; else incol;
    }
  }
}

object LSTMnextWord {
  class Opts extends Net.Opts {
    var width = 1;
    var height = 1;
    var nvocab = 100000;
    var lopts:LSTMLayer.Options = null;   
  }
  
  class Options extends Opts {}
}
/**
 * LSTM unit 
 */

class LSTMLayer(override val net:Net, override val opts:LSTMLayer.Options = new LSTMLayer.Options) extends CompoundLayer(net, opts) {
	override val _inputs = new Array[Layer](3);
	override val _inputNums = Array(0,0,0);
	override val _outputs = new Array[Mat](2);
	override val _derivs = new Array[Mat](2);
}

object LSTMLayer {
	class Options extends CompoundLayer.Options {	
	  	  
	  def constructNet = {
	    val prev_c = new CopyLayer.Options;
	    val prev_h = new CopyLayer.Options;
	    val i = new CopyLayer.Options;
	    
	  	val il1 = new LinLayer.Options{inputs(0) = i;      modelName = prefix + "LSTM_il1"};
	  	val ph1 = new LinLayer.Options{inputs(0) = prev_h; modelName = prefix + "LSTM_ph1"};
	  	val sum1 = new AddLayer.Options{inputs(0) = il1;   inputs(1) = ph1};
	  	val in_gate = new SigmoidLayer.Options{inputs(0) = sum1};
	  	
	  	val il2 = new LinLayer.Options{inputs(0) = i;      modelName = prefix + "LSTM_il2"};
	  	val ph2 = new LinLayer.Options{inputs(0) = prev_h; modelName = prefix + "LSTM_ph12"};
	  	val sum2 = new AddLayer.Options{inputs(0) = il2;   inputs(1) = ph2};
	  	val out_gate = new SigmoidLayer.Options{inputs(0) = sum2};
	  	
	  	val il3 = new LinLayer.Options{inputs(0) = i;      modelName = prefix + "LSTM_il3"};
	  	val ph3 = new LinLayer.Options{inputs(0) = prev_h; modelName = prefix + "LSTM_ph13"};
	  	val sum3 = new AddLayer.Options{inputs(0) = il3;   inputs(1) = ph3};
	  	val forget_gate = new SigmoidLayer.Options{inputs(0) = sum3};
	  	
	    val il4 = new LinLayer.Options{inputs(0) = i;      modelName = prefix + "LSTM_il4"};
	  	val ph4 = new LinLayer.Options{inputs(0) = prev_h; modelName = prefix + "LSTM_ph14"};
	  	val sum4 = new AddLayer.Options{inputs(0) = il4;   inputs(1) = ph4};
	  	val in_gate2 = new TanhLayer.Options{inputs(0) = sum4};
	  	
	  	val in_prod = new MulLayer.Options{inputs(0) = in_gate;    inputs(1) = in_gate2};
	  	val f_prod = new MulLayer.Options{inputs(0) = forget_gate; inputs(1) = prev_c};
	  	val next_c = new AddLayer.Options{inputs(0) = in_prod;     inputs(1) = f_prod};
	  	
	  	val next_tanh = new TanhLayer.Options{inputs(0) = next_c;};
	  	val next_h = new MulLayer.Options{inputs(0) = out_gate;    inputs(1) = next_tanh};
	  	
	  	lopts = Array(prev_h, prev_c, i,                                         // First 3 layers should be inputs
	  	              il1, ph1, sum1, in_gate,                                   // Otherwise the ordering should support forward-backward inference
	  	              il2, ph2, sum2, out_gate, 
	  	              il3, ph3, sum3, forget_gate, 
	  			          il4, ph4, sum4, in_gate2, 
	  			          in_prod, f_prod, next_c, 
	  			          next_tanh, next_h);
	  	
	  	lopts.map(_.parent = this);
	  	outputNumbers = Array(lopts.length-1, lopts.length-3);                   // Specifies the output layer numbers (next_h and next_c)
	  }
	  

	  override def clone:Options = {
		  copyTo(new Options).asInstanceOf[Options];
	  }

	  override def create(net:Net):LSTMLayer = {
		  apply(net, this);
	  }
	}
  
  def apply(net:Net) = new LSTMLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = {
    val x = new LSTMLayer(net, opts);
    x.construct;
    x;
  }
  

}





