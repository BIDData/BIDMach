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
 * LSTM next Word prediction model, which comprises a rectangular grid of LSTM compound layers.
 */
class NextWord(override val opts:NextWord.Opts = new NextWord.Options) extends Net(opts) {
  
  var shiftedInds:Mat = null;
  var leftedge:Layer = null;
  var height = 0;
  var width = 0;
  val preamble_size = 3;
  // define some getters/setters on the grid
  def getlayer(j:Int, i:Int):Layer = layers(j + i * width + preamble_size);
  def setlayer(j:Int, i:Int, ll:Layer) = {layers(j + i * width + preamble_size) = ll};
	
	override def createLayers = {
	  height = opts.height;
	  width = opts.width; 
    layers = if (opts.allout) {
    	new Array[Layer]((height+2) * width + preamble_size);  
    } else {
      new Array[Layer]((height) * width + preamble_size + 2);
    }
    leftedge = InputLayer(this);                     // dummy layer, left edge of zeros    
    
    // the preamble (bottom) layers
    layers(0) = InputLayer(this);
    val lopts1 = new LinLayer.Options{modelName = "inWordMap"; outdim = opts.dim; aopts = opts.aopts};
    layers(1) = LinLayer(this, lopts1).setinput(0, layers(0));
    val spopts = new SplitHorizLayer.Options{nparts = opts.width};
    layers(2) = SplitHorizLayer(this, spopts).setinput(0, layers(1));
    
    // the main grid
    for (i <- 0 until height) {
    	val lopts = new LSTMLayer.Options;
    	lopts.dim = opts.dim;
    	lopts.aopts = opts.aopts;
    	lopts.kind = opts.kind;
    	lopts.prefix = if (opts.bylevel) "level_%d" format i; else ""
    	lopts.constructNet;
      for (j <- 0 until width) {
        val layer = LSTMLayer(this, lopts);
        if (i > 0) {
          layer.setinput(2, getlayer(j, i-1));           // in most layers, input 2 (i) is from layer below
        } else {
        	layer.setinout(2, layers(2), j);               // on bottom layer, input 2 is j^th output from the split layer
        }
        if (j > 0) {
          layer.setinput(0, getlayer(j-1, i));           // input 0 (prev_h) is layer to the left, output 0 (h)
          layer.setinout(1, getlayer(j-1, i), 1);        // input 1 (prev_c) is layer to the left, output 1 (c)
        } else {
          layer.setinput(0, leftedge);                   // in first column, just use dummy (zeros) input
          layer.setinput(1, leftedge);
        }
        setlayer(j, i, layer);
      }
    }
    
    // the top layers
    val lopts2 = new LinLayer.Options{modelName = "outWordMap"; outdim = opts.nvocab; aopts = opts.aopts};
    val sopts = new SoftmaxOutputLayer.Options;
    if (opts.allout) {
    	output_layers = new Array[Layer](width);
    	for (j <- 0 until width) {
    		val linlayer = LinLayer(this, lopts2).setinput(0, getlayer(j, height - 1));
    		setlayer(j, height, linlayer);    	
    		val smlayer = SoftmaxOutputLayer(this, sopts).setinput(0, linlayer);
    		setlayer(j, height+1, smlayer);
    		output_layers(j) = smlayer;
    	}
    } else {
      val linlayer = LinLayer(this, lopts2).setinput(0, getlayer(width-1, height - 1));
      layers(width*height+preamble_size) = linlayer;
      val smlayer = SoftmaxOutputLayer(this, sopts).setinput(0, linlayer);   
      layers(width*height+preamble_size+1) = smlayer;    
      output_layers = Array(smlayer);
    }
  }
  
  override def assignInputs(gmats:Array[Mat], ipass:Int, pos:Long) {
    if (batchSize % opts.width != 0) throw new RuntimeException("LSTMwordPredict error: batch size must be a multiple of network width %d %d" format (batchSize, opts.width))
    val nr = batchSize / opts.width;
    val in = gmats(0).view(opts.width, nr).t.view(1, batchSize);
    layers(0).output = oneHot(in, opts.nvocab);
    if (leftedge.output.asInstanceOf[AnyRef] == null) {
      leftedge.output = convertMat(zeros(opts.dim, nr));
    }
  }
  
  override def assignTargets(gmats:Array[Mat], ipass:Int, pos:Long) {
  	val nr = batchSize / opts.width;
  	val in0 = gmats(0);
  	if (shiftedInds.asInstanceOf[AnyRef] == null) shiftedInds = convertMat(irow(1->in0.ncols) \ (in0.ncols-1));
  	val inshift = in0(0, shiftedInds);
    val in = inshift.view(opts.width, nr).t;
    if (opts.allout) {
    	for (j <- 0 until opts.width) {
    		val incol = in.colslice(j,j+1).t;
    		getlayer(j, height+1).target = if (targmap.asInstanceOf[AnyRef] != null) targmap * incol; else incol;
    	}
    } else {
      val incol = in.colslice(opts.width-1, opts.width).t;
      layers(height*width + preamble_size + 1).target = if (targmap.asInstanceOf[AnyRef] != null) targmap * incol; else incol;
    }
  }
}

object NextWord {
  trait Opts extends Net.Opts {
    var width = 1;
    var height = 1;
    var nvocab = 100000;
    var kind = 0;
    var allout = true;
    var bylevel = true;
  }
  
  class Options extends Opts {}
  
   def mkNetModel(fopts:Model.Opts) = {
    new NextWord(fopts.asInstanceOf[NextWord.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
    new ADAGrad(nopts.asInstanceOf[ADAGrad.Opts])
  } 
  
  def mkRegularizer(nopts:Mixin.Opts):Array[Mixin] = {
    Array(new L1Regularizer(nopts.asInstanceOf[L1Regularizer.Opts]))
  }
    
  class LearnOptions extends Learner.Options with NextWord.Opts with MatDS.Opts with ADAGrad.Opts with L1Regularizer.Opts

  def learner(mat0:Mat) = {
    val opts = new LearnOptions;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1);
  	val nn = new Learner(
  	    new MatDS(Array(mat0), opts), 
  	    new NextWord(opts), 
  	    Array(new L1Regularizer(opts)),
  	    new ADAGrad(opts), 
  	    opts)
    (nn, opts)
  }
  
  def learnerX(mat0:Mat) = {
    val opts = new LearnOptions;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1);
  	val nn = new Learner(
  	    new MatDS(Array(mat0), opts), 
  	    new NextWord(opts), 
  	    null,
  	    null, 
  	    opts)
    (nn, opts)
  }
  
  class FDSopts extends Learner.Options with NextWord.Opts with FilesDS.Opts with ADAGrad.Opts with L1Regularizer.Opts
   
  def learner(fn1:String):(Learner, FDSopts) = learner(List(FilesDS.simpleEnum(fn1,1,0)));

  def learner(fnames:List[(Int)=>String]):(Learner, FDSopts) = {   
    val opts = new FDSopts;
    opts.fnames = fnames
    opts.batchSize = 100000;
    opts.eltsPerSample = 500;
    implicit val threads = threadPool(4);
    val ds = new FilesDS(opts)
  	val nn = new Learner(
  			ds, 
  	    new NextWord(opts), 
  	    Array(new L1Regularizer(opts)),
  	    new ADAGrad(opts), 
  	    opts)
    (nn, opts)
  } 
}