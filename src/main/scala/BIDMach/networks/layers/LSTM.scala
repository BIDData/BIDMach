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

/**
 * LSTM unit 
 */

class LSTMLayer(override val net:Net, override val opts:LSTMNode = new LSTMNode) extends CompoundLayer(net, opts) {
	override val _inputs = new Array[LayerTerm](3);
	override val _outputs = new Array[ND](2);
	override val _derivs = new Array[ND](2);
  
  override def toString = {
    "LSTM@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait LSTMNodeOpts extends CompoundNodeOpts {
    var dim = 0;
    var kind = 1;
    var hasBias = false;  
    var scoreType = 0;
    var outdim = 0;
    
   def copyOpts(opts:LSTMNodeOpts):LSTMNodeOpts = {
  		super.copyOpts(opts);
  		opts.dim = dim;
  		opts.kind = kind;
  		opts.hasBias = hasBias;
  		opts.scoreType = scoreType;
  		opts.outdim = outdim;
  		opts;
    }
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
        case 5 => constructGraph5
        case _ => throw new RuntimeException("LSTMLayer type %d not recognized" format kind);
      }
    }
  
    // Basic LSTM topology with 8 linear layers
	  	  
	  def constructGraph0 = {
    	import BIDMach.networks.layers.Node._
    	val odim = dim;
    		
    	val in_h = copy;
    	val in_c = copy; 
    	val in_i = copy;

    	val lin1 = linear(in_h)(prefix+"LSTM_h_in_gate",     outdim=odim, hasBias=hasBias);
    	val lin2 = linear(in_h)(prefix+"LSTM_h_out_gate",    outdim=odim, hasBias=hasBias);   
    	val lin3 = linear(in_h)(prefix+"LSTM_h_forget_gate", outdim=odim, hasBias=hasBias);
    	val lin4 = linear(in_h)(prefix+"LSTM_h_tanh_gate",   outdim=odim, hasBias=hasBias);
    	
    	val lin5 = linear(in_i)(prefix+"LSTM_i_in_gate",     outdim=odim, hasBias=hasBias);
    	val lin6 = linear(in_i)(prefix+"LSTM_i_out_gate",    outdim=odim, hasBias=hasBias);   
    	val lin7 = linear(in_i)(prefix+"LSTM_i_forget_gate", outdim=odim, hasBias=hasBias);
    	val lin8 = linear(in_i)(prefix+"LSTM_i_tanh_gate",   outdim=odim, hasBias=hasBias);
    	
    	val sum1 = lin1 + lin5;
    	val sum2 = lin2 + lin6;
    	val sum3 = lin3 + lin7;
    	val sum4 = lin4 + lin8;
    	 	
    	val in_gate = σ(sum1);
    	val out_gate = σ(sum2);
    	val forget_gate = σ(sum3);
    	val in_sat = tanh(sum4);
    	
    	val in_prod = in_gate ∘ in_sat;
    	val f_prod = forget_gate ∘ in_c;
    	val out_c = in_prod + f_prod;
    	
    	val out_tanh = tanh(out_c);
    	val out_h = out_gate ∘ out_tanh;

    	grid = (in_h on in_c on in_i on null) \  (lin1   \  lin5    \   sum1   \   in_gate      \  in_prod  \  out_tanh  on
                                                lin2   \  lin6    \   sum2   \   out_gate     \  f_prod   \  out_h     on
                                                lin3   \  lin7    \   sum3   \   forget_gate  \  out_c    \  null      on
                                                lin4   \  lin8    \   sum4   \   in_sat       \  null     \  null);
    	
    	val lopts = grid.data;
    	lopts.map((x:Node) => if (x != null) x.parent = this);
    	outputNumbers = Array(lopts.indexOf(out_h), lopts.indexOf(out_c));
	  }
	     
	  // LSTM with 4 linear layers, with h and i stacked as inputs
    
    def constructGraph1 = {
    	import BIDMach.networks.layers.Node._
    	val odim = dim;
    		
    	val in_h = copy;
    	val in_c = copy; 
    	val in_i = copy;
    	val h_over_i = in_h over in_i;

    	val lin1 = linear(h_over_i)(prefix+"LSTM_in_gate",     outdim=odim, hasBias=hasBias);
    	val lin2 = linear(h_over_i)(prefix+"LSTM_out_gate",    outdim=odim, hasBias=hasBias);   
    	val lin3 = linear(h_over_i)(prefix+"LSTM_forget_gate", outdim=odim, hasBias=hasBias);
    	val lin4 = linear(h_over_i)(prefix+"LSTM_tanh_gate",   outdim=odim, hasBias=hasBias);
    	
    	val in_gate = σ(lin1);
    	val out_gate = σ(lin2);
    	val forget_gate = σ(lin3);
    	val in_sat = tanh(lin4);
    	
    	val in_prod = in_gate ∘ in_sat;
    	val f_prod = forget_gate ∘ in_c;
    	val out_c = in_prod + f_prod;
    	
    	val out_tanh = tanh(out_c);
    	val out_h = out_gate ∘ out_tanh;

    	grid = in_h      \   lin1   \  in_gate      \  in_prod  \  out_tanh  on
             in_c      \   lin2   \  out_gate     \  f_prod   \  out_h     on
             in_i      \   lin3   \  forget_gate  \  out_c    \  null      on
             h_over_i  \   lin4   \  in_sat       \  null     \  null;
    	
    	val lopts = grid.data;
    	lopts.map((x:Node) => if (x != null) x.parent = this);
    	outputNumbers = Array(lopts.indexOf(out_h), lopts.indexOf(out_c));
    	
    }
    
    // LSTM with 1 linear layer, with h and i stacked as inputs, and all 4 output stacked
    
    def constructGraph2 = {
      import BIDMach.networks.layers.Node._   
      val odim = dim;
      val in_h = copy;
      val in_c = copy;
      val in_i = copy;          
      val h_over_i = in_h over in_i;
      
      val lin = linear(h_over_i)(prefix+"LSTM_all", outdim=4*odim, hasBias=hasBias);
      val sp = splitvert(lin, 4);
      
      val in_gate = σ(sp(0));
    	val out_gate = σ(sp(1));
    	val forget_gate = σ(sp(2));
    	val in_sat = tanh(sp(3));
    	
    	val in_prod = in_gate ∘ in_sat;
    	val f_prod = forget_gate ∘ in_c;
    	val out_c = in_prod + f_prod;
    	
    	val out_tanh = tanh(out_c);
    	val out_h = out_gate ∘ out_tanh;     
      
      grid = in_h      \   lin    \  in_gate      \  in_prod  \  out_tanh  on
             in_c      \   sp     \  out_gate     \  f_prod   \  out_h     on
             in_i      \   null   \  forget_gate  \  out_c    \  null      on
             h_over_i  \   null   \  in_sat       \  null     \  null;
      
      val lopts = grid.data;      
      lopts.map((x:Node) => if (x != null) x.parent = this);
      outputNumbers = Array(lopts.indexOf(out_h), lopts.indexOf(out_c));                   // Specifies the output layer numbers (next_h and next_c)
    }
    
    // LSTM with two sets of layers, paired outputs. More stable to train than the single linlayer network
    
    def constructGraph3 = {
      import BIDMach.networks.layers.Node._
      val odim = dim;
      val in_h = copy;
      val in_c = copy;
      val in_i = copy;          
      val h_over_i = in_h over in_i;
      
      val lin1 = linear(h_over_i)(prefix+"LSTM_in_out",      outdim=2*odim, hasBias=hasBias);
      val sp1 = splitvert(lin1, 2);
      val lin2 = linear(h_over_i)(prefix+"LSTM_forget_tanh", outdim=2*odim, hasBias=hasBias);
      val sp2 = splitvert(lin2, 2);
      
      val in_gate = σ(sp1(0));
    	val out_gate = σ(sp1(1));
    	val forget_gate = σ(sp2(0));
    	val in_sat = tanh(sp2(1));
    	
    	val in_prod = in_gate ∘ in_sat;
    	val f_prod = forget_gate ∘ in_c;
    	val out_c = in_prod + f_prod;
    	
    	val out_tanh = tanh(out_c);
    	val out_h = out_gate ∘ out_tanh;     
      
      grid = in_h      \   lin1   \  in_gate      \  in_prod  \  out_tanh  on
             in_c      \   sp1    \  out_gate     \  f_prod   \  out_h     on
             in_i      \   lin2   \  forget_gate  \  out_c    \  null      on
             h_over_i  \   sp2    \  in_sat       \  null     \  null;
      
      val lopts = grid.data;      
      lopts.map((x:Node) => if (x != null) x.parent = this);
      outputNumbers = Array(lopts.indexOf(out_h), lopts.indexOf(out_c));                   // Specifies the output layer numbers (next_h and next_c)
    }
    
       // LSTM with 2 linear layers from h and i respectively
    
    def constructGraph4 = {
      import BIDMach.networks.layers.Node._  
      val odim = dim;
      val in_h = copy;
      val in_c = copy;
      val in_i = copy;          
      
      val linh = linear(in_h)(prefix+"LSTM_h", outdim=4*odim, hasBias=hasBias);
      val sph = splitvert(linh, 4);
      val lini = linear(in_i)(prefix+"LSTM_i", outdim=4*odim, hasBias=hasBias);
      val spi = splitvert(lini, 4);
      
      val lin1 = sph(0) + spi(0);
      val lin2 = sph(1) + spi(1);
      val lin3 = sph(2) + spi(2);
      val lin4 = sph(3) + spi(3);
      
      val in_gate = σ(lin1);
    	val out_gate = σ(lin2);
    	val forget_gate = σ(lin3);
    	val in_sat = tanh(lin4);
    	
    	val in_prod = in_gate ∘ in_sat;
    	val f_prod = forget_gate ∘ in_c;
    	val out_c = in_prod + f_prod;
    	
    	val out_tanh = tanh(out_c);
    	val out_h = out_gate ∘ out_tanh;     
      
      grid = (in_h on in_c on in_i on null)  \  (linh   \  lin1   \  in_gate      \  in_prod  \  out_tanh  on
                                                 sph    \  lin2   \  out_gate     \  f_prod   \  out_h     on
                                                 lini   \  lin3   \  forget_gate  \  out_c    \  null      on
                                                 spi    \  lin4   \  in_sat       \  null     \  null);
      
      val lopts = grid.data;      
      lopts.map((x:Node) => if (x != null) x.parent = this);
      outputNumbers = Array(lopts.indexOf(out_h), lopts.indexOf(out_c));                   // Specifies the output layer numbers (next_h and next_c)
    }
   
    // LSTM using a fused inner kernel
    
    def constructGraph5 = {
    	import BIDMach.networks.layers.Node._
    	val odim = dim;
    		
    	val in_h = copy;
    	val in_c = copy; 
    	val in_i = copy;
    	val h_over_i = in_h over in_i;

    	val lin1 = linear(h_over_i)(prefix+"LSTM_in_gate",     outdim=odim, hasBias=hasBias);
    	val lin2 = linear(h_over_i)(prefix+"LSTM_out_gate",    outdim=odim, hasBias=hasBias);   
    	val lin3 = linear(h_over_i)(prefix+"LSTM_forget_gate", outdim=odim, hasBias=hasBias);
    	val lin4 = linear(h_over_i)(prefix+"LSTM_tanh_gate",   outdim=odim, hasBias=hasBias);
    	
    	val lstm_gate = lstm_fused(in_c, lin1, lin2, lin3, lin4);    	
    	val out_h = copy(new NodeTerm(lstm_gate, 1));

    	grid = in_h      \   lin1   \  lstm_gate  on
             in_c      \   lin2   \  out_h      on
             in_i      \   lin3   \  null       on
             h_over_i  \   lin4   \ null       ;
    	
    	val lopts = grid.data;
    	lopts.map((x:Node) => if (x != null) x.parent = this);
    	outputNumbers = Array(lopts.indexOf(out_h), lopts.indexOf(lstm_gate));
    	
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
  
  final val gridTypeNoOutput = 0;
  final val gridTypeSoftmaxOutput = 1;
  final val gridTypeNegsampOutput = 2;
  
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
  
  class GridOpts extends LSTMNodeOpts {var netType = 0; var bylevel = true};
  
  def grid(nrows:Int, ncols:Int, opts:GridOpts):NodeMat = {
    import BIDMach.networks.layers.Node._
    val nlin = 2;
    val odim = opts.outdim;
    val idim = opts.dim;
    val nsoft = opts.netType match {
      case `gridTypeNoOutput` => 0;
      case `gridTypeNegsampOutput` => 1;
      case `gridTypeSoftmaxOutput` => 2;
    }
    val gr = NodeMat(nrows + nlin + nsoft, ncols);
    
    for (k <- 0 until ncols) {
    	gr(0, k) = input
    }
    
    val modelName = opts.modelName;
    
    for (k <- 0 until ncols) {
    	gr(1, k) = linear(gr(0, k))((modelName format 0) +"_bottom", outdim=idim, hasBias = opts.hasBias)
    }
    
    for (k <- 0 until ncols) {
      for (j <- nlin until nrows + nlin) {
        val modelName = if (opts.bylevel) (opts.modelName format j-nlin) else (opts.modelName format 0)
    	  val below = gr(j-1, k); 
        if (k > 0) {
        	val left = gr(j, k-1).asInstanceOf[LSTMNode]
        	gr(j, k) = lstm(h=left.h, c=left.c, i=below, m=modelName)(opts);
        } else {
          gr(j, k) = lstm(h=null, c=null, i=below, m=modelName)(opts);
        }
      }
    }
    
    opts.netType match {
      case `gridTypeNoOutput` => {}
      case `gridTypeSoftmaxOutput` => {
        for (k <- 0 until ncols) {
        	gr(nrows + nlin, k) = linear(gr(nrows + nlin - 1, k))(name=opts.modelName+"_top", outdim=odim, hasBias = opts.hasBias)
          gr(nrows + nlin + 1, k) = softmaxout(gr(nrows + nlin, k))(opts.scoreType);
        }
      }
      case `gridTypeNegsampOutput` => {
        for (k <- 0 until ncols) {
          gr(nrows + nlin, k) = negsamp(gr(nrows + nlin - 1, k))(name=opts.modelName+"_top", outdim=odim, hasBias=opts.hasBias, scoreType=opts.scoreType)
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
  
  def grid(net:Net, nrows:Int, ncols:Int, opts:LSTMNode.GridOpts):LayerMat = {
    val nodeGrid = LSTMNode.grid(nrows, ncols, opts);
    LayerMat(nodeGrid, net);
  }
}
