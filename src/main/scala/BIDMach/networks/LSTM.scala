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

/**
 * LSTM unit 
 */

class LSTMLayer(override val net:Net, override val opts:LSTMLayer.Options = new LSTMLayer.Options) extends CompoundLayer(net, opts) {
	override val _inputs = new Array[Layer](3);
	override val _inputTerminals = Array(0,0,0);
	override val _outputs = new Array[Mat](2);
	override val _derivs = new Array[Mat](2);
}

object LSTMLayer {
	class Options extends CompoundLayer.Options {	
    
    var dim = 0;
    var kind = 0;
    
    def constructNet = {
      kind match {
        case 0 => constructNet0
        case 1 => constructNet1
        case 2 => constructNet2
        case 3 => constructNet3
        case _ => throw new RuntimeException("LSTMLayer type %d not recognized" format kind);
      }
    }
    
    // Basic LSTM topology with 8 linear layers
	  	  
	  def constructNet0 = {
			val prev_h = new CopyLayer.Options;
	    val prev_c = new CopyLayer.Options;
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
    
    // LSTM with all layers grouped into 1 - not very stable to train
    
    def constructNet1 = {
      val prev_h = new CopyLayer.Options;
      val prev_c = new CopyLayer.Options;
      val i = new CopyLayer.Options;
      
      val prev_hi = new StackLayer.Options{inputs(0) = prev_h; inputs(1) = i};       
      val il = new LinLayer.Options{inputs(0) = prev_hi; modelName = prefix + "LSTM_all"; outdim = 4*dim};      
      val sp = new SplitVertLayer.Options{inputs(0) = il;  nparts = 4;}

      val in_gate = new SigmoidLayer.Options{inputs(0) = sp;     inputTerminals(0) = 0};
      val out_gate = new SigmoidLayer.Options{inputs(0) = sp;    inputTerminals(0) = 1};
      val forget_gate = new SigmoidLayer.Options{inputs(0) = sp; inputTerminals(0) = 2};
      val in_gate2 = new TanhLayer.Options{inputs(0) = sp;       inputTerminals(0) = 3};
      
      val in_prod = new MulLayer.Options{inputs(0) = in_gate;    inputs(1) = in_gate2};
      val f_prod = new MulLayer.Options{inputs(0) = forget_gate; inputs(1) = prev_c};
      val next_c = new AddLayer.Options{inputs(0) = in_prod;     inputs(1) = f_prod};
      
      val next_tanh = new TanhLayer.Options{inputs(0) = next_c;};
      val next_h = new MulLayer.Options{inputs(0) = out_gate;    inputs(1) = next_tanh};
      
      lopts = Array(prev_h, prev_c, i,                                         // First 3 layers should be inputs
                    prev_hi, il, sp, 
                    in_gate,                                                   // Otherwise the ordering should support forward-backward inference
                    out_gate, 
                    forget_gate, 
                    in_gate2, 
                    in_prod, f_prod, next_c, 
                    next_tanh, next_h);
      
      lopts.map(_.parent = this);
      outputNumbers = Array(lopts.length-1, lopts.length-3);                   // Specifies the output layer numbers (next_h and next_c)
    }
    
    // LSTM with 4 linear layers, with h and x stacked as inputs
    
    def constructNet2 = {
      val prev_h = new CopyLayer.Options;
      val prev_c = new CopyLayer.Options;
      val i = new CopyLayer.Options;
      
      val prev_hi = new StackLayer.Options{inputs(0) = prev_h; inputs(1) = i};       

      val lin1 = new LinLayer.Options{inputs(0) = prev_hi; modelName = prefix + "LSTM_lin1";  outdim = dim}
      val in_gate = new SigmoidLayer.Options{inputs(0) = lin1};
      
      val lin2 = new LinLayer.Options{inputs(0) = prev_hi; modelName = prefix + "LSTM_lin2";  outdim = dim}
      val out_gate = new SigmoidLayer.Options{inputs(0) = lin2};
      
      val lin3 = new LinLayer.Options{inputs(0) = prev_hi; modelName = prefix + "LSTM_lin3";  outdim = dim}
      val forget_gate = new SigmoidLayer.Options{inputs(0) = lin3};
      
      val lin4 = new LinLayer.Options{inputs(0) = prev_hi; modelName = prefix + "LSTM_lin4";  outdim = dim}
      val in_gate2 = new TanhLayer.Options{inputs(0) = lin4};
      
      val in_prod = new MulLayer.Options{inputs(0) = in_gate;    inputs(1) = in_gate2};
      val f_prod = new MulLayer.Options{inputs(0) = forget_gate; inputs(1) = prev_c};
      val next_c = new AddLayer.Options{inputs(0) = in_prod;     inputs(1) = f_prod};
      
      val next_tanh = new TanhLayer.Options{inputs(0) = next_c;};
      val next_h = new MulLayer.Options{inputs(0) = out_gate;    inputs(1) = next_tanh};
      
      lopts = Array(prev_h, prev_c, i,                                         // First 3 layers should be inputs
                    prev_hi,  
                    lin1, in_gate,                                                   // Otherwise the ordering should support forward-backward inference
                    lin2, out_gate, 
                    lin3, forget_gate, 
                    lin4, in_gate2, 
                    in_prod, f_prod, next_c, 
                    next_tanh, next_h);
      
      lopts.map(_.parent = this);
      outputNumbers = Array(lopts.length-1, lopts.length-3);                   // Specifies the output layer numbers (next_h and next_c)
    }
    
    // LSTM with two sets of layers, paired outputs. More stable to train than the single linlayer network
    
    def constructNet3 = {
      val prev_h = new CopyLayer.Options;
      val prev_c = new CopyLayer.Options;
      val i = new CopyLayer.Options;
          
      val prev_hi = new StackLayer.Options{inputs(0) = prev_h; inputs(1) = i};       
      val il1 = new LinLayer.Options{inputs(0) = prev_hi; modelName = prefix + "LSTM_lin1"; outdim = 2*dim};      
      val sp1 = new SplitVertLayer.Options{inputs(0) = il1;  nparts = 2;}
      val il2 = new LinLayer.Options{inputs(0) = prev_hi; modelName = prefix + "LSTM_lin2"; outdim = 2*dim};      
      val sp2 = new SplitVertLayer.Options{inputs(0) = il2;  nparts = 2;}

      val in_gate = new SigmoidLayer.Options{inputs(0) = sp1;     inputTerminals(0) = 0};
      val out_gate = new SigmoidLayer.Options{inputs(0) = sp1;    inputTerminals(0) = 1};
      val forget_gate = new SigmoidLayer.Options{inputs(0) = sp2; inputTerminals(0) = 0};
      val in_gate2 = new TanhLayer.Options{inputs(0) = sp2;       inputTerminals(0) = 1};
      
      val in_prod = new MulLayer.Options{inputs(0) = in_gate;    inputs(1) = in_gate2};
      val f_prod = new MulLayer.Options{inputs(0) = forget_gate; inputs(1) = prev_c};
      val next_c = new AddLayer.Options{inputs(0) = in_prod;     inputs(1) = f_prod};
      
      val next_tanh = new TanhLayer.Options{inputs(0) = next_c;};
      val next_h = new MulLayer.Options{inputs(0) = out_gate;    inputs(1) = next_tanh};
      
      lopts = Array(prev_h, prev_c, i,                                         // First 3 layers should be inputs
                    prev_hi, il1, sp1, il2, sp2,
                    in_gate,                                                   // Otherwise the ordering should support forward-backward inference
                    out_gate, 
                    forget_gate, 
                    in_gate2, 
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
