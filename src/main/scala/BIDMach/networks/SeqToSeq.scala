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
  
  var PADsym:Mat = null;
  var leftedge:Layer = null;
  var height = 0;
  var fullheight = 0;
  var inwidth = 0;
  var outwidth = 0;
  var width = 0;
  var srcn = 0;
  var dstxn = 0;
  var dstyn = 0;
  val preamble_rows = 2;
  // define some getters/setters on the grid
  def lindex(r:Int, c:Int) = if (c < inwidth) (r + c * fullheight) else (width * fullheight + r + (c - inwidth) * (fullheight + 2));
  def getlayer(r:Int, c:Int):Layer = layers(lindex(r,c));
  def setlayer(r:Int, c:Int, ll:Layer) = {layers(lindex(r,c)) = ll};
	
	override def createLayers = {
    height = opts.height;
	  fullheight = height + preamble_rows;
	  inwidth = opts.inwidth; 
    outwidth = opts.outwidth;
    width = inwidth + outwidth;
    layers =  new Array[Layer](fullheight * width + outwidth * 2);
    leftedge = InputLayer(this);                     // dummy layer, left edge of zeros    
    
    // the preamble (bottom) layers
    val lopts1 = new LinLayer.Options{modelName = "srcWordMap"; outdim = opts.dim; aopts = opts.aopts};
    val lopts2 = new LinLayer.Options{modelName = "dstWordMap"; outdim = opts.dim; aopts = opts.aopts};
    for (j <- 0 until width) {
    	setlayer(0, j, InputLayer(this));
      if (j < inwidth) {
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
      for (j <- 0 until width) {
    	  val layer = LSTMLayer(this, if (j < inwidth) loptsSrc else loptsDst);
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
    output_layers = new Array[Layer](outwidth);
    for (j <- 0 until outwidth) {
    	val linlayer = LinLayer(this, lopts3).setinput(0, getlayer(height-1, j+inwidth));
    	setlayer(height, j, linlayer);    	
    	val smlayer = SoftmaxOutputLayer(this, sopts).setinput(0, linlayer);
    	setlayer(height+1, j, smlayer);
    	output_layers(j) = smlayer;
    }
  }
  
  override def assignInputs(gmats:Array[Mat], ipass:Int, pos:Long) {
    val src = gmats(0);
    val dstx = gmats(1);
    srcn = src.nnz/src.ncols;
    if (srcn*src.ncols != src.nnz) throw new RuntimeException("SeqToSeq src batch not fixed length");
    dstxn = dstx.nnz/dstx.ncols;
    if (dstxn*dstx.ncols != dstx.nnz) throw new RuntimeException("SeqToSeq dstx batch not fixed length");
    val srcdata = int(src.contents.view(srcn, batchSize).t);   // IMat with columns corresponding to word positions, with batchSize rows. 
    val dstxdata = int(dstx.contents.view(dstxn, batchSize).t);
    val srcmat = oneHot(srcdata.contents, opts.nvocab);
    val dstxmat = oneHot(dstxdata.contents, opts.nvocab);
    for (i <- 0 until srcn) {
      val cols = srcmat.colslice(i*batchSize, (i+1)*batchSize);
      layers(inwidth + i - srcn).output = cols;
    }
    for (i <- 0 until dstxn) {
      val cols = dstxmat.colslice(i*batchSize, (i+1)*batchSize);
      layers(inwidth + i).output = cols;
    }   
    if (leftedge.output.asInstanceOf[AnyRef] == null) {
      leftedge.output = convertMat(zeros(opts.dim, batchSize));
    }
  }
  
  override def assignTargets(gmats:Array[Mat], ipass:Int, pos:Long) {
	  val dsty = if (gmats.length > 2) gmats(2) else gmats(1);
	  dstyn = dsty.nnz/dsty.ncols;
    if (dstyn*dsty.ncols != dsty.nnz) throw new RuntimeException("SeqToSeq dsty batch not fixed length");
    val dstydata = int(dsty.contents.view(dstyn, batchSize).t);
    val dstylim = if (gmats.length > 2) dstyn else dstyn - 1;
    for (j <- 0 until dstylim) {
    	val incol = if (gmats.length > 2) dstydata.colslice(j,j+1).t else dstydata.colslice(j+1,j+2).t ;
    	getlayer(fullheight+1,j).target = incol;
    }
    if (PADsym.asInstanceOf[AnyRef] == null) {
      PADsym = convertMat(iones(1, batchSize) * opts.PADsymbol);
    }
    if (dstylim < dstxn) {
    	getlayer(fullheight+1, dstylim).target = PADsym;
    }
  }
  
  override def dobatch(gmats:Array[Mat], ipass:Int, pos:Long):Unit = {
    if (batchSize < 0) batchSize = gmats(0).ncols;
    if (batchSize == gmats(0).ncols) {                                    // discard odd-sized minibatches
      assignInputs(gmats, ipass, pos);
      assignTargets(gmats, ipass, pos);
      if (mask.asInstanceOf[AnyRef] != null) {
        modelmats(0) ~ modelmats(0) ∘ mask;
      }
      val minlayer = lindex(0, inwidth - srcn);
      val maxlayer = lindex(0, inwidth + dstxn); 
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
      val minlayer = lindex(0, inwidth - srcn);
      val maxlayer = lindex(0, inwidth + dstxn); 
      var i = minlayer;
      while (i < maxlayer) {
        if (opts.debug > 0) {
          println("evalbatch forward %d %s" format (i, layers(i).getClass))
        }
        layers(i).forward;
        i += 1;
      }
      if (putBack >= 0) {
        output_layers(dstxn-1).output.colslice(0, gmats(0).ncols, gmats(1));
      }
      val scores = zeros(output_layers.length, 1);
      var j = 0;
      while (j < dstxn) {
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
    var inwidth = 1;
    var outwidth = 1;
    var height = 1;
    var nvocab = 100000;
    var kind = 0;
    var bylevel = true;
    var PADsymbol = 1;
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
    opts.batchSize = 128;
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
  
  
  def learnerX(fnames:List[(Int)=>String]):(Learner, FDSopts) = {   
    val opts = new FDSopts;
    opts.fnames = fnames
    opts.batchSize = 128;
    opts.eltsPerSample = 500;
    implicit val threads = threadPool(4);
    val ds = new FilesDS(opts)
    val nn = new Learner(
        ds, 
        new SeqToSeq(opts), 
        null,
        null, 
        opts)
    (nn, opts)
  }
  
}

