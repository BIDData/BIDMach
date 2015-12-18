package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
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

/**
 * LSTM unit 
 */

class LSTMLayer(override val net:Net, override val opts:LSTMNode = new LSTMNode) extends CompoundLayer(net, opts) {
	override val _inputs = new Array[LayerTerm](3);
	override val _outputs = new Array[Mat](2);
	override val _derivs = new Array[Mat](2);
  
  override def toString = {
    "LSTM@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait LSTMNodeOpts extends CompoundNodeOpts {
    var dim = 0;
    var kind = 0;
    var hasBias = false;  
    var scoreType = 0;
    var outDim = 0;
}

class LSTMNode extends CompoundNode with LSTMNodeOpts {	
  
	  override val inputs:Array[NodeTerm] = Array(null, null, null);
//    override val inputTerminals:Array[Int] = Array(0,0,0);
    
    def constructGraph = {
      kind match {
        case 0 => constructGraph0
        case 1 => constructGraph1
        case 2 => constructGraph2
        case 3 => constructGraph3
        case 4 => constructGraph4
        case _ => throw new RuntimeException("LSTMLayer type %d not recognized" format kind);
      }
    }
  
    // Basic LSTM topology with 8 linear layers
	  	  
	  def constructGraph0 = {
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
	  	val g = new TanhNode{inputs(0) = sum4};
	  	
	  	val in_prod = new MulNode{inputs(0) = in_gate;    inputs(1) = g};
	  	val f_prod = new MulNode{inputs(0) = forget_gate; inputs(1) = prev_c};
	  	val next_c = new AddNode{inputs(0) = in_prod;     inputs(1) = f_prod};
	  	
	  	val next_tanh = new TanhNode{inputs(0) = next_c;};
	  	val next_h = new MulNode{inputs(0) = out_gate;    inputs(1) = next_tanh};
	  	
	  	lopts = Array(prev_h, prev_c, i,                                         // First 3 layers should be inputs
	  	              il1, ph1, sum1, in_gate,                                   // Otherwise the ordering should support forward-backward inference
	  	              il2, ph2, sum2, out_gate, 
	  	              il3, ph3, sum3, forget_gate, 
	  			          il4, ph4, sum4, g, 
	  			          in_prod, f_prod, next_c, 
	  			          next_tanh, next_h);
	  	
	  	lopts.map(_.parent = this);
	  	outputNumbers = Array(lopts.length-1, lopts.length-3);                   // Specifies the output layer numbers (next_h and next_c)
	  }
    
    // LSTM with all linear weights grouped into single matrix, to more fully use GPU.
    // Observed not very stable to train
    // 
    //  | Wi_in     Wh_in     |
    //  | Wi_out    Wh_out    | | i |
    //  | Wi_forget Wh_forget | | h |
    //  | Wi_g      Wh_g      |
    
    def constructGraph1 = {
      val prev_h = new CopyNode;
      val prev_c = new CopyNode;
      val i = new CopyNode;
      
      val prev_hi = new StackNode{inputs(0) = prev_h; inputs(1) = i};       
      val il = new LinNode{inputs(0) = prev_hi; modelName = prefix + "LSTM_all"; outdim = 4*dim; hasBias = this.hasBias};      
      val sp = new SplitVertNode{inputs(0) = il;  nparts = 4;}

      val in_gate = new SigmoidNode{inputs(0) = sp(0)};
      val out_gate = new SigmoidNode{inputs(0) = sp(1)};
      val forget_gate = new SigmoidNode{inputs(0) = sp(2)};
      val g = new TanhNode{inputs(0) = sp(3)};
      
      val in_prod = new MulNode{inputs(0) = in_gate;    inputs(1) = g};
      val f_prod = new MulNode{inputs(0) = forget_gate; inputs(1) = prev_c};
      val next_c = new AddNode{inputs(0) = in_prod;     inputs(1) = f_prod};
      
      val next_tanh = new TanhNode{inputs(0) = next_c;};
      val next_h = new MulNode{inputs(0) = out_gate;    inputs(1) = next_tanh};
      
      lopts = Array(prev_h, prev_c, i,                                         // First 3 layers should be inputs
                    prev_hi, il, sp, 
                    in_gate,                                                   // Otherwise the ordering should support forward-backward inference
                    out_gate, 
                    forget_gate, 
                    g, 
                    in_prod, f_prod, next_c, 
                    next_tanh, next_h);
      
      lopts.map(_.parent = this);
      outputNumbers = Array(lopts.length-1, lopts.length-3);                   // Specifies the output layer numbers (next_h and next_c)
    }
    
    // LSTM with 4 linear layers, with h and i stacked as inputs
    //  | Wi_in Wh_in | | i |   +  | Wi_out Wh_out | | i |   +  | Wi_forget Wh_forget | | i |   +  | Wi_g Wh_g | | i |
    //                  | h |                        | h |                              | h |                    | h |
    
    def constructGraph2 = {
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
      val g = new TanhNode{inputs(0) = lin4};
      
      val in_prod = new MulNode{inputs(0) = in_gate;    inputs(1) = g};
      val f_prod = new MulNode{inputs(0) = forget_gate; inputs(1) = prev_c};
      val next_c = new AddNode{inputs(0) = in_prod;     inputs(1) = f_prod};
      
      val next_tanh = new TanhNode{inputs(0) = next_c;};
      val next_h = new MulNode{inputs(0) = out_gate;    inputs(1) = next_tanh};
      
      lopts = Array(prev_h, prev_c, i,                                         // First 3 layers should be inputs
                    prev_hi,  
                    lin1, in_gate,                                                   // Otherwise the ordering should support forward-backward inference
                    lin2, out_gate, 
                    lin3, forget_gate, 
                    lin4, g, 
                    in_prod, f_prod, next_c, 
                    next_tanh, next_h);
      
      lopts.map(_.parent = this);
      outputNumbers = Array(lopts.length-1, lopts.length-3);                   // Specifies the output layer numbers (next_h and next_c)
    }
    
    // LSTM with two sets of linear weights, paired outputs. More stable to train than the single linlayer network
    //
    //  | Wi_in  Wh_in  | | i |   +  | Wi_forget Wh_forget | | i |
    //  | Wi_out Wh_out | | h |      | Wi_g      Wh_g      | | h |
    
    def constructGraph3 = {
      val prev_h = new CopyNode;
      val prev_c = new CopyNode;
      val i = new CopyNode;
          
      val prev_hi = new StackNode{inputs(0) = prev_h; inputs(1) = i};       
      val il1 = new LinNode{inputs(0) = prev_hi; modelName = prefix + "LSTM_in_out"; outdim = 2*dim; hasBias = this.hasBias};      
      val sp1 = new SplitVertNode{inputs(0) = il1;  nparts = 2;}
      val il2 = new LinNode{inputs(0) = prev_hi; modelName = prefix + "LSTM_forget_tanh"; outdim = 2*dim; hasBias = this.hasBias};      
      val sp2 = new SplitVertNode{inputs(0) = il2;  nparts = 2;}

      val in_gate = new SigmoidNode{inputs(0) = sp1(0)};
      val out_gate = new SigmoidNode{inputs(0) = sp1(1)};
      val forget_gate = new SigmoidNode{inputs(0) = sp2(0)};
      val g = new TanhNode{inputs(0) = sp2(1)};
      
      val in_prod = new MulNode{inputs(0) = in_gate;    inputs(1) = g};
      val f_prod = new MulNode{inputs(0) = forget_gate; inputs(1) = prev_c};
      val next_c = new AddNode{inputs(0) = in_prod;     inputs(1) = f_prod};
      
      val next_tanh = new TanhNode{inputs(0) = next_c;};
      val next_h = new MulNode{inputs(0) = out_gate;    inputs(1) = next_tanh};
      
      lopts = Array(prev_h, prev_c, i,                                         // First 3 layers should be inputs
                    prev_hi, il1, sp1, il2, sp2,
                    in_gate,                                                   // Otherwise the ordering should support forward-backward inference
                    out_gate, 
                    forget_gate, 
                    g, 
                    in_prod, f_prod, next_c, 
                    next_tanh, next_h);
      
      lopts.map(_.parent = this);
      outputNumbers = Array(lopts.length-1, lopts.length-3);                   // Specifies the output layer numbers (next_h and next_c)
    }
    
    // LSTM with 4 linear layers, with h and i stacked as inputs
    
    def constructGraph4 = {
    	import BIDMach.networks.layers.Node._
    		
    	val in_h = copy;
    	val in_c = copy; 
    	val in_i = copy;
    	val h_on_i = in_h on in_i;

    	val lin1 = linear(h_on_i)(prefix+"LSTM_in_gate", outdim=dim, hasBias=hasBias);
    	val lin2 = linear(h_on_i)(prefix+"LSTM_out_gate", outdim=dim, hasBias=hasBias);   
    	val lin3 = linear(h_on_i)(prefix+"LSTM_forget_gate", outdim=dim, hasBias=hasBias);
    	val lin4 = linear(h_on_i)(prefix+"LSTM_tanh_gate", outdim=dim, hasBias=hasBias);
    	
    	val in_gate = σ(lin1);
    	val out_gate = σ(lin2);
    	val forget_gate = σ(lin3);
    	val in_sat = tanh(lin4);
    	
    	val in_prod = in_gate ∘ in_sat;
    	val f_prod = forget_gate ∘ in_c;
    	val out_c = in_prod + f_prod;
    	
    	val out_tanh = tanh(out_c);
    	val out_h = out_gate ∘ out_tanh;

    	grid = in_h    \   lin1   \  in_gate      \  in_prod  \  out_tanh  on
             in_c    \   lin2   \  out_gate     \  f_prod   \  out_h     on
             in_i    \   lin3   \  forget_gate  \  out_c    \  null      on
             h_on_i  \   lin4   \  in_sat       \  null     \  null;
    	
    	lopts = grid.data.filter(_ != null);
    	lopts.map(_.parent = this);
    	outputNumbers = Array(lopts.indexOf(out_h), lopts.indexOf(out_c));
    	
    }
	  
	  override def clone:LSTMNode = {
		  copyTo(new LSTMNode).asInstanceOf[LSTMNode];
	  }

	  override def create(net:Net):LSTMLayer = {
		  LSTMLayer(net, this);
	  }
    
    override def toString = {
    "LSTM@"+Integer.toHexString(hashCode % 0x10000).toString
    }
    
    def h = apply(0);
    
    def c = apply(1);
	}


  
object LSTMNode {    
  
  def apply() = {
    val n = new LSTMNode;
    n.constructGraph;
    n
  }
  
  def apply(opts:LSTMNodeOpts) = {
    val n = new LSTMNode;
    opts.copyOpts(n);
    n.constructGraph;
    n
  }
  
  class GridOpts extends LSTMNodeOpts {var netType = 0};
  
  def grid(nrows:Int, ncols:Int, opts:GridOpts) = {
    import BIDMach.networks.layers.Node._
    val nlin = 2;
    val nsoft = opts.netType match {
      case 0 => 0;
      case 1 => 1;
      case 2 => 2;
    }
    val gr = NodeMat(nrows + nlin + nsoft, ncols);
    
    for (k <- 0 until ncols) {
    	gr(0, k) = input
    }
    
    for (k <- 0 until ncols) {
    	gr(1, k) = linear(gr(0, k))(opts.modelName+"Grid_input", outdim=opts.dim, hasBias = opts.hasBias)
    }
    
    for (k <- 0 until ncols) {
      for (j <- nlin until nrows + nlin) {
    	  val below = gr(j-1, k); 
        if (k > 0) {
        	val left = gr(j, k-1).asInstanceOf[LSTMNode]
        	gr(j, k) = lstm(h=left.h, c=left.c, i=below)(opts);
        } else {
          gr(j, k) = lstm(h=null, c=null, i=below)(opts);
        }
      }
    }
    
    opts.netType match {
      case 0 => {}
      case 1 => {
        for (k <- 0 until ncols) {
        	gr(nrows + nlin, k) = linear(gr(nrows + nlin - 1, k))(name=opts.modelName+"Grid_output", outdim=opts.outDim, hasBias = opts.hasBias)
          gr(nrows + nlin + 1, k) = softmaxout(gr(nrows + nlin, k))(opts.scoreType);
        }
      }
      case 2 => {
        for (k <- 0 until ncols) {
          gr(nrows + nlin, k) = negsamp(gr(nrows + nlin - 1, k))(name=opts.modelName+"Grid_output", outdim=opts.outDim, hasBias=opts.hasBias, scoreType=opts.scoreType)
        }
      }
    }
    gr
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
