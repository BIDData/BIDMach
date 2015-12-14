package BIDMach.networks

import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach.networks.layers._
import BIDMach._
import scala.util.hashing.MurmurHash3;
import scala.collection.mutable.HashMap;

/**
 * LSTM unit 
 */

class LSTMLayer(override val net:Net, override val opts:LSTMNodeOpts = new LSTMNode) extends CompoundLayer(net, opts) {
	override val _inputs = new Array[Layer](3);
	override val _inputTerminals = Array(0,0,0);
	override val _outputs = new Array[Mat](2);
	override val _derivs = new Array[Mat](2);
}

trait LSTMNodeOpts extends CompoundNodeOpts {
    var dim = 0;
    var kind = 0;
    var hasBias = false;  
}

class LSTMNode extends CompoundNode with LSTMNodeOpts {	
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
			val prev_h = new CopyNode;
	    val prev_c = new CopyNode;
	    val i = new CopyNode;
	    
	  	val il1 = new LinNode{inputs(0) = i;      modelName = prefix + "LSTM_input_in_gate"; hasBias = this.hasBias};
	  	val ph1 = new LinNode{inputs(0) = prev_h; modelName = prefix + "LSTM_prev_h_in_gate"; hasBias = this.hasBias};
	  	val sum1 = new AddNode{inputs(0) = il1;   inputs(1) = ph1};
	  	val in_gate = new SigmoidNode{inputs(0) = sum1};
	  	
	  	val il2 = new LinNode{inputs(0) = i;      modelName = prefix + "LSTM_input_out_gate"; hasBias = this.hasBias};
	  	val ph2 = new LinNode{inputs(0) = prev_h; modelName = prefix + "LSTM_prev_h_out_gate"; hasBias = this.hasBias};
	  	val sum2 = new AddNode{inputs(0) = il2;   inputs(1) = ph2};
	  	val out_gate = new SigmoidNode{inputs(0) = sum2};
	  	
	  	val il3 = new LinNode{inputs(0) = i;      modelName = prefix + "LSTM_input_forget_gate"; hasBias = this.hasBias};
	  	val ph3 = new LinNode{inputs(0) = prev_h; modelName = prefix + "LSTM_prev_h_forget_gate"; hasBias = this.hasBias};
	  	val sum3 = new AddNode{inputs(0) = il3;   inputs(1) = ph3};
	  	val forget_gate = new SigmoidNode{inputs(0) = sum3};
	  	
	    val il4 = new LinNode{inputs(0) = i;      modelName = prefix + "LSTM_input_tanh"; hasBias = this.hasBias};
	  	val ph4 = new LinNode{inputs(0) = prev_h; modelName = prefix + "LSTM_prev_h_tanh"; hasBias = this.hasBias};
	  	val sum4 = new AddNode{inputs(0) = il4;   inputs(1) = ph4};
	  	val in_gate2 = new TanhNode{inputs(0) = sum4};
	  	
	  	val in_prod = new MulNode{inputs(0) = in_gate;    inputs(1) = in_gate2};
	  	val f_prod = new MulNode{inputs(0) = forget_gate; inputs(1) = prev_c};
	  	val next_c = new AddNode{inputs(0) = in_prod;     inputs(1) = f_prod};
	  	
	  	val next_tanh = new TanhNode{inputs(0) = next_c;};
	  	val next_h = new MulNode{inputs(0) = out_gate;    inputs(1) = next_tanh};
	  	
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
      val prev_h = new CopyNode;
      val prev_c = new CopyNode;
      val i = new CopyNode;
      
      val prev_hi = new StackNode{inputs(0) = prev_h; inputs(1) = i};       
      val il = new LinNode{inputs(0) = prev_hi; modelName = prefix + "LSTM_all"; outdim = 4*dim; hasBias = this.hasBias};      
      val sp = new SplitVertNode{inputs(0) = il;  nparts = 4;}

      val in_gate = new SigmoidNode{inputs(0) = sp;     inputTerminals(0) = 0};
      val out_gate = new SigmoidNode{inputs(0) = sp;    inputTerminals(0) = 1};
      val forget_gate = new SigmoidNode{inputs(0) = sp; inputTerminals(0) = 2};
      val in_gate2 = new TanhNode{inputs(0) = sp;       inputTerminals(0) = 3};
      
      val in_prod = new MulNode{inputs(0) = in_gate;    inputs(1) = in_gate2};
      val f_prod = new MulNode{inputs(0) = forget_gate; inputs(1) = prev_c};
      val next_c = new AddNode{inputs(0) = in_prod;     inputs(1) = f_prod};
      
      val next_tanh = new TanhNode{inputs(0) = next_c;};
      val next_h = new MulNode{inputs(0) = out_gate;    inputs(1) = next_tanh};
      
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
      val prev_h = new CopyNode;
      val prev_c = new CopyNode;
      val i = new CopyNode;
      
      val prev_hi = new StackNode{inputs(0) = prev_h; inputs(1) = i};       

      val lin1 = new LinNode{inputs(0) = prev_hi; modelName = prefix + "LSTM_in_gate";  outdim = dim; hasBias = this.hasBias}
      val in_gate = new SigmoidNode{inputs(0) = lin1};
      
      val lin2 = new LinNode{inputs(0) = prev_hi; modelName = prefix + "LSTM_out_gate";  outdim = dim; hasBias = this.hasBias}
      val out_gate = new SigmoidNode{inputs(0) = lin2};
      
      val lin3 = new LinNode{inputs(0) = prev_hi; modelName = prefix + "LSTM_forget_gate";  outdim = dim; hasBias = this.hasBias}
      val forget_gate = new SigmoidNode{inputs(0) = lin3};
      
      val lin4 = new LinNode{inputs(0) = prev_hi; modelName = prefix + "LSTM_tanh";  outdim = dim; hasBias = this.hasBias}
      val in_gate2 = new TanhNode{inputs(0) = lin4};
      
      val in_prod = new MulNode{inputs(0) = in_gate;    inputs(1) = in_gate2};
      val f_prod = new MulNode{inputs(0) = forget_gate; inputs(1) = prev_c};
      val next_c = new AddNode{inputs(0) = in_prod;     inputs(1) = f_prod};
      
      val next_tanh = new TanhNode{inputs(0) = next_c;};
      val next_h = new MulNode{inputs(0) = out_gate;    inputs(1) = next_tanh};
      
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
      val prev_h = new CopyNode;
      val prev_c = new CopyNode;
      val i = new CopyNode;
          
      val prev_hi = new StackNode{inputs(0) = prev_h; inputs(1) = i};       
      val il1 = new LinNode{inputs(0) = prev_hi; modelName = prefix + "LSTM_in_out"; outdim = 2*dim; hasBias = this.hasBias};      
      val sp1 = new SplitVertNode{inputs(0) = il1;  nparts = 2;}
      val il2 = new LinNode{inputs(0) = prev_hi; modelName = prefix + "LSTM_forget_tanh"; outdim = 2*dim; hasBias = this.hasBias};      
      val sp2 = new SplitVertNode{inputs(0) = il2;  nparts = 2;}

      val in_gate = new SigmoidNode{inputs(0) = sp1;     inputTerminals(0) = 0};
      val out_gate = new SigmoidNode{inputs(0) = sp1;    inputTerminals(0) = 1};
      val forget_gate = new SigmoidNode{inputs(0) = sp2; inputTerminals(0) = 0};
      val in_gate2 = new TanhNode{inputs(0) = sp2;       inputTerminals(0) = 1};
      
      val in_prod = new MulNode{inputs(0) = in_gate;    inputs(1) = in_gate2};
      val f_prod = new MulNode{inputs(0) = forget_gate; inputs(1) = prev_c};
      val next_c = new AddNode{inputs(0) = in_prod;     inputs(1) = f_prod};
      
      val next_tanh = new TanhNode{inputs(0) = next_c;};
      val next_h = new MulNode{inputs(0) = out_gate;    inputs(1) = next_tanh};
      
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
	  
	  override def clone:LSTMNode = {
		  copyTo(new LSTMNode).asInstanceOf[LSTMNode];
	  }

	  override def create(net:Net):LSTMLayer = {
		  LSTMLayer(net, this);
	  }
	}

  
object LSTMLayer {    
  
  def apply(net:Net) = new LSTMLayer(net, new LSTMNode);
  
  def apply(net:Net, opts:LSTMNode) = {
    val x = new LSTMLayer(net, opts);
    x.construct;
    x;
  }
}
