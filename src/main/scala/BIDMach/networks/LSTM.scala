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
  var allButFirst:Mat = null;
  var leftedge:Layer = null;
  var lopts:LSTMLayer.Options = null;
  var height = 0;
  var width = 0;
  val preamble_size = 3;
  // define some getters/setters on the grid
  def getlayer(i:Int, j:Int):Layer = layers(j + i * width + preamble_size);
  def setlayer(i:Int, j:Int, ll:Layer) = {layers(j + i * width + preamble_size) = ll};
	
	override def createLayers = {
	  lopts = opts.lopts;
	  height = opts.height;
	  width = opts.width;  
    layers = new Array[Layer]((height+2) * width + preamble_size);  
    lopts.constructNet;
    val leftedge = InputLayer(this);                     // dummy layer, left edge of zeros    
    
    // the preamble (bottom) layers
    layers(0) = InputLayer(this);
    val lopts1 = new LinLayer.Options{modelName = "inWordMap"; outdim = opts.dim * opts.width};
    layers(1) = LinLayer(this, lopts1).setinput(0, layers(0));
    val spopts = new SplitLayer.Options{nparts = opts.width};
    layers(2) = SplitLayer(this, spopts).setinput(0, layers(1));
    
    // the main grid
    for (i <- 0 until height) {
      for (j <- 0 until width) {
        val layer = LSTMLayer(this, lopts);
        layer.setinout(2, layers(2), j);                 // input 2 is an output from the split layer
        if (j > 0) {
          layer.setinput(0, getlayer(j-1, i));           // input 0 is layer to the left, output 0
          layer.setinout(1, getlayer(j-1, i), 1);        // input 1 is layer to the left, output 1
        }
        setlayer(j, i, layer);
      }
      getlayer(0, i).setinput(0, leftedge);              // set left edge inputs
      getlayer(0, i).setinput(1, leftedge);
    }
    
    // the top layers
    val lopts2 = new LinLayer.Options{modelName = "outWordMap"; outdim = opts.nvocab};
    val sopts = new SoftmaxLayer.Options;
    for (j <- 0 until width) {
    	val linlayer = LinLayer(this, lopts2).setinput(0, getlayer(j, height - 1));
    	setlayer(j, height, linlayer);    	
    	val smlayer = SoftmaxLayer(this, sopts).setinput(0, linlayer);
    	setlayer(j, height+1, smlayer);
    }
  }
  
  override def assignInputs(gmats:Array[Mat], ipass:Int, pos:Long) {
    if (batchSize % opts.width != 0) throw new RuntimeException("LSTMwordPredict error: batch size must be a multiple of network width %d %d" format (batchSize, opts.width))
    val nr = batchSize / opts.width;
    val in = gmats(0).view(nr, opts.width).t.view(batchSize,1);
    layers(0).output = oneHot(in, opts.nvocab);
    if (leftedge.output.asInstanceOf[AnyRef] == null) {
      leftedge.output = in.izeros(opts.dim, batchSize);
    }
  }
  
  override def assignTargets(gmats:Array[Mat], ipass:Int, pos:Long) {
  	val nr = batchSize / opts.width;
  	val in0 = gmats(0);
  	if (dummyword.asInstanceOf[AnyRef] == null) dummyword = in0.izeros(1,1);
  	if (allButFirst.asInstanceOf[AnyRef] == null) allButFirst = convertMat(1->(in0.ncols));
  	val inshift = in0(0, allButFirst) \ dummyword;
    val in = inshift.view(nr, opts.width).t;
    for (j <- 0 until opts.width) {
    	val incol = in.colslice(j,j+1);
    	getlayer(j, height+1).target = 
    			if (targmap.asInstanceOf[AnyRef] != null) targmap * incol; else incol;
    }
  }
}

object LSTMnextWord {
  trait Opts extends Net.Opts {
    var width = 1;
    var height = 1;
    var nvocab = 100000;
    var lopts:LSTMLayer.Options = null;   
  }
  
  class Options extends Opts {}
  
   def mkNetModel(fopts:Model.Opts) = {
    new LSTMnextWord(fopts.asInstanceOf[LSTMnextWord.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
    new ADAGrad(nopts.asInstanceOf[ADAGrad.Opts])
  } 
  
  def mkRegularizer(nopts:Mixin.Opts):Array[Mixin] = {
    Array(new L1Regularizer(nopts.asInstanceOf[L1Regularizer.Opts]))
  }
    
  class LearnOptions extends Learner.Options with LSTMnextWord.Opts with MatDS.Opts with ADAGrad.Opts with L1Regularizer.Opts

  def learner(mat0:Mat) = {
    val opts = new LearnOptions;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1);
    opts.lopts = new LSTMLayer.Options;
  	val nn = new Learner(
  	    new MatDS(Array(mat0), opts), 
  	    new LSTMnextWord(opts), 
  	    Array(new L1Regularizer(opts)),
  	    new ADAGrad(opts), 
  	    opts)
    (nn, opts)
  }
  
  def learnerX(mat0:Mat) = {
    val opts = new LearnOptions;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1);
    opts.lopts = new LSTMLayer.Options;
  	val nn = new Learner(
  	    new MatDS(Array(mat0), opts), 
  	    new LSTMnextWord(opts), 
  	    null,
  	    null, 
  	    opts)
    (nn, opts)
  }
  
  class FDSopts extends Learner.Options with LSTMnextWord.Opts with FilesDS.Opts with ADAGrad.Opts with L1Regularizer.Opts
   
  def learner(fn1:String):(Learner, FDSopts) = learner(List(FilesDS.simpleEnum(fn1,1,0)));

  def learner(fnames:List[(Int)=>String]):(Learner, FDSopts) = {   
    val opts = new FDSopts;
    opts.fnames = fnames
    opts.batchSize = 100000;
    opts.eltsPerSample = 500;
    opts.lopts = new LSTMLayer.Options;
    implicit val threads = threadPool(4);
    val ds = new FilesDS(opts)
  	val nn = new Learner(
  			ds, 
  	    new LSTMnextWord(opts), 
  	    Array(new L1Regularizer(opts)),
  	    new ADAGrad(opts), 
  	    opts)
    (nn, opts)
  } 
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





