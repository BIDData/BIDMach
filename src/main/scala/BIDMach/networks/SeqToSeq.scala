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
class SeqToSeq(override val opts:SeqToSeq.Opts = new SeqToSeq.Options) extends Net(opts) {
  
  var EOS:Mat = null;
  var leftedge:Layer = null;
  var height = 0;
  var width = 0;
  var srcn = 0;
  var dstn = 0;
  val preamble_rows = 2;
  // define some getters/setters on the grid
  def getlayer(r:Int, c:Int):Layer = layers(r + c * height);
  def setlayer(r:Int, c:Int, ll:Layer) = {layers(r + c * height) = ll};
	
	override def createLayers = {
	  height = opts.height + preamble_rows + 1;
	  width = opts.width; 
    layers =  new Array[Layer](height * width);
    leftedge = InputLayer(this);                     // dummy layer, left edge of zeros    
    
    // the preamble (bottom) layers
    val lopts1 = new LinLayer.Options{modelName = "srcWordMap"; outdim = opts.dim; aopts = opts.aopts};
    val lopts2 = new LinLayer.Options{modelName = "dstWordMap"; outdim = opts.dim; aopts = opts.aopts};
    for (j <- 0 until 2*width) {
    	setlayer(0, j, InputLayer(this));
      if (j < width) {
    	  setlayer(1, j, LinLayer(this, lopts1).setinput(0, getlayer(j, 0)));
      } else {
        setlayer(1, j, LinLayer(this, lopts2).setinput(0, getlayer(j, 0)));
      }
    }

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
    	  layer.setinput(2, getlayer(i-1+preamble_rows, j));             // input 2 (i) is from layer below
        if (j > 0) {
          layer.setinput(0, getlayer(i+preamble_rows, j-1));           // input 0 (prev_h) is layer to the left, output 0 (h)
          layer.setinout(1, getlayer(i+preamble_rows, j-1), 1);        // input 1 (prev_c) is layer to the left, output 1 (c)
        } else {
          layer.setinput(0, leftedge);                   // in first column, just use dummy (zeros) input
          layer.setinput(1, leftedge);
        }
        setlayer(i+preamble_rows, j, layer);
      }
    }
    
    // the top layers
    val lopts3 = new LinLayer.Options{modelName = "outWordMap"; outdim = opts.nvocab; aopts = opts.aopts};
    val sopts = new SoftmaxOutputLayer.Options;
    output_layers = new Array[Layer](width);
    for (j <- 0 until width) {
    	val linlayer = LinLayer(this, lopts3).setinput(0, getlayer(height-2, j+width));
    	setlayer(height-1, j, linlayer);    	
    	val smlayer = SoftmaxOutputLayer(this, sopts).setinput(0, linlayer);
    	setlayer(height-1, j+width, smlayer);
    	output_layers(j) = smlayer;
    }
  }
  
  override def assignInputs(gmats:Array[Mat], ipass:Int, pos:Long) {
    val src = gmats(0);
    val dst = gmats(1);
    srcn = src.nnz/src.ncols;
    if (srcn*src.ncols != src.nnz) throw new RuntimeException("SeqToSeq src batch not fixed length");
    dstn = dst.nnz/dst.ncols;
    if (dstn*dst.ncols != dst.nnz) throw new RuntimeException("SeqToSeq dst batch not fixed length");
    val srcdata = int(src.contents.view(srcn, batchSize).t);   // IMat with columns corresponding to word positions, with batchSize rows. 
    val dstdata = int(dst.contents.view(dstn, batchSize).t);
    val srcmat = oneHot(srcdata.contents, opts.nvocab);
    val dstmat = oneHot(dstdata.contents, opts.nvocab);
    for (i <- 0 until srcn) {
      val cols = srcmat.colslice(i*batchSize, (i+1)*batchSize);
      layers(width + i - srcn).output = cols;
    }
    for (i <- 0 until dstn) {
      val cols = dstmat.colslice(i*batchSize, (i+1)*batchSize);
      layers(width + i).output = cols;
    }   
    if (leftedge.output.asInstanceOf[AnyRef] == null) {
      leftedge.output = convertMat(zeros(opts.dim, batchSize));
    }
  }
  
  override def assignTargets(gmats:Array[Mat], ipass:Int, pos:Long) {
	  val dst = gmats(1);
    val dstdata = int(dst.contents.view(dstn, batchSize).t);
    for (j <- 0 until dstn-1) {
    	val incol = dstdata.colslice(j+1,j+2).t;
    	getlayer(height-1,width+j).target = incol;
    }
    if (EOS.asInstanceOf[AnyRef] == null) {
      EOS = convertMat(iones(1, batchSize) * opts.EOSsym);
    }
    getlayer(height-1, width+dstn-1).target = EOS;
  }
  
  override def dobatch(gmats:Array[Mat], ipass:Int, pos:Long):Unit = {
    if (batchSize < 0) batchSize = gmats(0).ncols;
    if (batchSize == gmats(0).ncols) {                                    // discard odd-sized minibatches
      assignInputs(gmats, ipass, pos);
      assignTargets(gmats, ipass, pos);
      if (mask.asInstanceOf[AnyRef] != null) {
        modelmats(0) ~ modelmats(0) ∘ mask;
      }
      val minlayer = (width - srcn) * height;
      val maxlayer = (width + dstn) * height;
      var i = minlayer;
      while (i < maxlayer) {
        if (opts.debug > 0) {
          println("dobatch forward %d %s" format (i, layers(i).getClass))
        }
        layers(i).forward;
        i += 1;
      }
      var j = 0;
      while (j < output_layers.length) {
        output_layers(j).deriv.set(1);
        j += 1;
      }
      if (opts.aopts == null) {
        for (j <- 0 until updatemats.length) updatemats(j).clear;
      }
      while (i > minlayer) {
        i -= 1;
        if (opts.debug > 0) {
          println("dobatch backward %d %s" format (i, layers(i).getClass))
        }
        layers(i).backward(ipass, pos);
      }
    }
  }
  
  override def evalbatch(mats:Array[Mat], ipass:Int, pos:Long):FMat = {  
    if (batchSize < 0) batchSize = gmats(0).ncols;
    if (batchSize == gmats(0).ncols) { 
      assignInputs(gmats, ipass, pos);
      assignTargets(gmats, ipass, pos);
      if (mask.asInstanceOf[AnyRef] != null) {
        modelmats(0) ~ modelmats(0) ∘ mask;
      }
      val minlayer = (width - srcn) * height;
      val maxlayer = (width + dstn) * height;
      var i = minlayer;
      while (i < maxlayer) {
        if (opts.debug > 0) {
          println("evalbatch forward %d %s" format (i, layers(i).getClass))
        }
        layers(i).forward;
        i += 1;
      }
      if (putBack >= 0) {
        output_layers(dstn-1).output.colslice(0, gmats(0).ncols, gmats(1));
      }
      val scores = zeros(output_layers.length, 1);
      var j = 0;
      while (j < dstn) {
        scores(j) = output_layers(j).score.v;
        j += 1;
      }
      scores;
    } else {
      zeros(output_layers.length, 1);
    }
  }
}

object SeqToSeq {
  trait Opts extends Net.Opts {
    var width = 1;
    var height = 1;
    var nvocab = 100000;
    var kind = 0;
    var bylevel = true;
    var EOSsym = 1;
  }
  
  class Options extends Opts {}
  
   def mkNetModel(fopts:Model.Opts) = {
    new SeqToSeq(fopts.asInstanceOf[SeqToSeq.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
    new ADAGrad(nopts.asInstanceOf[ADAGrad.Opts])
  } 
  
  def mkRegularizer(nopts:Mixin.Opts):Array[Mixin] = {
    Array(new L1Regularizer(nopts.asInstanceOf[L1Regularizer.Opts]))
  }
    
  class LearnOptions extends Learner.Options with SeqToSeq.Opts with MatDS.Opts with ADAGrad.Opts with L1Regularizer.Opts

  def learner(mat0:Mat) = {
    val opts = new LearnOptions;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1);
  	val nn = new Learner(
  	    new MatDS(Array(mat0), opts), 
  	    new SeqToSeq(opts), 
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
  	    new SeqToSeq(opts), 
  	    null,
  	    null, 
  	    opts)
    (nn, opts)
  }
  
  class FDSopts extends Learner.Options with SeqToSeq.Opts with FilesDS.Opts with ADAGrad.Opts with L1Regularizer.Opts
   
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
  	    new SeqToSeq(opts), 
  	    Array(new L1Regularizer(opts)),
  	    new ADAGrad(opts), 
  	    opts)
    (nn, opts)
  } 
}

