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
class Translator(override val opts:Translator.Opts = new Translator.Options) extends Net(opts) {
  
  var shiftedInds:Mat = null;
  var leftedge:Layer = null;
  var height = 0;
  var width = 0;
  val preamble_size = 6;
  // define some getters/setters on the grid
  def getlayer(j:Int, i:Int):Layer = layers(j + i * width + preamble_size);
  def setlayer(j:Int, i:Int, ll:Layer) = {layers(j + i * width + preamble_size) = ll};
	
	override def createLayers = {
	  height = opts.height;
	  width = opts.width; 
    layers =  new Array[Layer]((height+1) * width + preamble_size);
    leftedge = InputLayer(this);                     // dummy layer, left edge of zeros    
    
    // the preamble (bottom) layers
    layers(0) = InputLayer(this);
    layers(1) = InputLayer(this);
    
    val lopts1 = new LinLayer.Options{modelName = "srcWordMap"; outdim = opts.dim; aopts = opts.aopts};
    layers(2) = LinLayer(this, lopts1).setinput(0, layers(0));
    val lopts2 = new LinLayer.Options{modelName = "dstWordMap"; outdim = opts.dim; aopts = opts.aopts};
    layers(3) = LinLayer(this, lopts2).setinput(0, layers(1));
    
    val spopts = new SplitHorizLayer.Options{nparts = opts.width};
    layers(4) = SplitHorizLayer(this, spopts).setinput(0, layers(2));
    val spopts2 = new SplitHorizLayer.Options{nparts = opts.width};
    layers(5) = SplitHorizLayer(this, spopts2).setinput(0, layers(3));
    
    // the main grid
    for (i <- 0 until height) {
    	val loptsSrc = new LSTMLayer.Options;
      val loptsDst = new LSTMLayer.Options;
    	loptsSrc.dim = opts.dim;
      loptsDst.dim = opts.dim;
    	loptsSrc.aopts = opts.aopts;
      loptsDst.aopts = opts.aopts;
    	loptsSrc.kind = opts.kind;
      loptsDst.kind = opts.kind;
    	loptsSrc.prefix = if (opts.bylevel) "SrcLevel_%d" format i; else "Src";
    	loptsDst.prefix = if (opts.bylevel) "DstLevel_%d" format i; else "Dst";
    	loptsSrc.constructNet;
      loptsDst.constructNet;
      for (j <- 0 until 2*width) {
        val layer = LSTMLayer(this, if (j < width) loptsSrc else loptsDst);
        if (i > 0) {
          layer.setinput(2, getlayer(j, i-1));           // in most layers, input 2 (i) is from layer below
        } else {
          if (j < width) {
        	  layer.setinout(2, layers(4), j);             // on bottom layer, input 2 is j^th output from the src split layer
          } else {
            layer.setinout(2, layers(5), j-width);       // on bottom layer, input 2 is j^th output from the dst split layer         
          }
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
    val lopts3 = new LinLayer.Options{modelName = "outWordMap"; outdim = opts.nvocab; aopts = opts.aopts};
    val sopts = new SoftmaxOutputLayer.Options;
    output_layers = new Array[Layer](width);
    for (j <- 0 until width) {
    	val linlayer = LinLayer(this, lopts3).setinput(0, getlayer(j+width, height - 1));
    	setlayer(j, height, linlayer);    	
    	val smlayer = SoftmaxOutputLayer(this, sopts).setinput(0, linlayer);
    	setlayer(j + width, height, smlayer);
    	output_layers(j) = smlayer;
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
    for (j <- 0 until opts.width) {
    	val incol = in.colslice(j,j+1).t;
    	getlayer(j, height+1).target = if (targmap.asInstanceOf[AnyRef] != null) targmap * incol; else incol;
    }
  }
}

object Translator {
  trait Opts extends Net.Opts {
    var width = 1;
    var height = 1;
    var nvocab = 100000;
    var kind = 0;
    var bylevel = true;
  }
  
  class Options extends Opts {}
  
   def mkNetModel(fopts:Model.Opts) = {
    new Translator(fopts.asInstanceOf[Translator.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
    new ADAGrad(nopts.asInstanceOf[ADAGrad.Opts])
  } 
  
  def mkRegularizer(nopts:Mixin.Opts):Array[Mixin] = {
    Array(new L1Regularizer(nopts.asInstanceOf[L1Regularizer.Opts]))
  }
    
  class LearnOptions extends Learner.Options with Translator.Opts with MatDS.Opts with ADAGrad.Opts with L1Regularizer.Opts

  def learner(mat0:Mat) = {
    val opts = new LearnOptions;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1);
  	val nn = new Learner(
  	    new MatDS(Array(mat0), opts), 
  	    new Translator(opts), 
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
  	    new Translator(opts), 
  	    null,
  	    null, 
  	    opts)
    (nn, opts)
  }
  
  class FDSopts extends Learner.Options with Translator.Opts with FilesDS.Opts with ADAGrad.Opts with L1Regularizer.Opts
   
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
  	    new Translator(opts), 
  	    Array(new L1Regularizer(opts)),
  	    new ADAGrad(opts), 
  	    opts)
    (nn, opts)
  } 
}

