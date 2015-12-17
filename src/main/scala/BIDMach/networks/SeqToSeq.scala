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

/*
 * LSTM next Word prediction model, which comprises a rectangular grid of LSTM compound layers.
 */
class SeqToSeq(override val opts:SeqToSeq.Opts = new SeqToSeq.Options) extends Net(opts) {
  
  var PADrow:Mat = null;
  var OOVelem:Mat = null;
  var leftedge:Layer = null;
  var leftStart:Mat = null;
  var dstxdata:Mat = null;
  var dstxdata0:Mat = null;
  var srcGrid:LayerMat = null;
  var dstGrid:LayerMat = null;
  var srcNodeGrid:NodeMat = null;
  var dstNodeGrid:NodeMat = null;
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
  var heightDiff = 2;
  def lindex(r:Int, c:Int) = if (c < inwidth) (r + c * fullheight) else (inwidth * fullheight + r + (c - inwidth) * (fullheight + heightDiff));
  def getlayer(r:Int, c:Int):Layer = layers(lindex(r,c));
  def setlayer(r:Int, c:Int, ll:Layer) = {layers(lindex(r,c)) = ll};
	
	override def createLayers = {
    height = opts.height;
    heightDiff = if (opts.netType == 0) 2 else 1;
	  fullheight = height + preamble_rows;
	  inwidth = opts.inwidth; 
    outwidth = opts.outwidth;
    width = inwidth + outwidth;
    layers =  new Array[Layer](fullheight * width + outwidth * heightDiff);
    leftedge = InputLayer(this);                     // dummy layer, left edge of zeros   
    
    var gridopts = new LSTMNode.GridOpts;
    gridopts.dim = opts.dim;
    gridopts.aopts = opts.aopts;
    gridopts.hasBias = opts.hasBias;
    gridopts.kind = opts.kind;
    gridopts.outDim = opts.nvocab;
    
    srcNodeGrid = LSTMNode.grid(height, inwidth, gridopts);
    dstNodeGrid = LSTMNode.grid(height, outwidth, gridopts);
    
 //   srcGrid = srcNodeGrid.map((x:Node) => LSTMLayer(this, x.asInstanceOf[LSTMNode]));
  //  dstGrid = dstNodeGrid.map((x:Node) => LSTMLayer(this, x.asInstanceOf[LSTMNode]));
    
//    srcGrid(?,inwidth-1).data.zip(dstGrid(?,0).data).map{case (from:Layer, to:Layer) => {to.input = from; to.setInput(1, from(1));}}
    
    // the preamble (bottom) layers
    val lopts1 = new LinNode{modelName = "srcWordMap"; outdim = opts.dim; aopts = opts.aopts; hasBias=opts.hasBias};
    val lopts2 = new LinNode{modelName = "dstWordMap"; outdim = opts.dim; aopts = opts.aopts; hasBias=opts.hasBias};
    for (j <- 0 until width) {
    	setlayer(0, j, InputLayer(this));
      if (j < inwidth) {
    	  setlayer(1, j, LinLayer(this, lopts1).setInput(0, getlayer(0, j)));
      } else {
        setlayer(1, j, LinLayer(this, lopts2).setInput(0, getlayer(0, j)));
      }
    }

    // the main grid
    for (i <- 0 until height) {
    	val loptsSrc = new LSTMNode{dim = opts.dim; aopts = opts.aopts; kind = opts.kind; hasBias = opts.hasBias};
      val loptsDst = new LSTMNode{dim = opts.dim; aopts = opts.aopts; kind = opts.kind; hasBias = opts.hasBias};
    	loptsSrc.prefix = if (opts.bylevel) "SrcLevel_%d" format i; else "Src";
    	loptsDst.prefix = if (opts.bylevel) "DstLevel_%d" format i; else "Dst";
    	loptsSrc.constructGraph;
      loptsDst.constructGraph;
      for (j <- 0 until width) {
    	  val layer = LSTMLayer(this, if (j < inwidth) loptsSrc else loptsDst);
    	  layer.setInput(2, getlayer(i-1+preamble_rows, j));             // input 2 (i) is from layer below
        if (j > 0) {
          layer.setInput(0, getlayer(i+preamble_rows, j-1));           // input 0 (prev_h) is layer to the left, output 0 (h)
          layer.setInput(1, getlayer(i+preamble_rows, j-1)(1));        // input 1 (prev_c) is layer to the left, output 1 (c)
        } else {
          layer.setInput(0, leftedge);                                 // in first column, just use dummy (zeros) input
          layer.setInput(1, leftedge);
        }
        setlayer(i+preamble_rows, j, layer);
      }
    }
    
    // the top layers
    output_layers = new Array[Layer](outwidth);
    if (opts.netType == 0) {
    	val lopts3 = new LinNode{modelName = "outWordMap"; outdim = opts.nvocab; aopts = opts.aopts; hasBias = opts.hasBias};
    	val sopts = new SoftmaxOutputNode{scoreType = opts.scoreType};
    	for (j <- 0 until outwidth) {
    		val linlayer = LinLayer(this, lopts3).setInput(0, getlayer(fullheight-1, j+inwidth));
    		setlayer(fullheight, inwidth + j, linlayer);    	
    		val smlayer = SoftmaxOutputLayer(this, sopts).setInput(0, linlayer);
    		setlayer(fullheight+1, inwidth + j, smlayer);
    		output_layers(j) = smlayer;
    	}
    } else {
      val nsopts = new NegsampOutputNode{modelName = "outWordMap"; outdim = opts.nvocab; aopts = opts.aopts; hasBias = opts.hasBias; 
                                                  scoreType = opts.scoreType; nsamps = opts.nsamps; expt = opts.expt};
      for (j <- 0 until outwidth) {
        val nslayer = NegsampOutputLayer(this, nsopts).setInput(0, getlayer(fullheight-1, j+inwidth));
        setlayer(fullheight, inwidth + j, nslayer);      
        output_layers(j) = nslayer;
      }    
    }
  }
  
  def mapOOV(in:Mat) = {
    if (OOVelem.asInstanceOf[AnyRef] == null) {
      OOVelem = convertMat(iones(1,1) * opts.OOVsym);
    }
    in ~ in + ((in > opts.nvocab) ∘ (OOVelem - in))
  }
  
  override def assignInputs(gmats:Array[Mat], ipass:Int, pos:Long) {
    val src = gmats(0);
    val dstx = gmats(1);
    srcn = src.nnz/src.ncols;
    if (srcn*src.ncols != src.nnz) throw new RuntimeException("SeqToSeq src batch not fixed length");
    val srcdata = int(src.contents.view(srcn, batchSize).t);   // IMat with columns corresponding to word positions, with batchSize rows.
    
    val dstxn0 = dstx.nnz/dstx.ncols;
    if (dstxn0*dstx.ncols != dstx.nnz) throw new RuntimeException("SeqToSeq dstx batch not fixed length"); 
    val dstxdata0 = int(dstx.contents.view(dstxn0, batchSize).t);
    dstxn = dstxn0 + (if (opts.addStart) 1 else 0);
    if (opts.addStart && (leftStart.asInstanceOf[AnyRef] == null)) {
      leftStart = convertMat(izeros(batchSize, 1));
    }
    val dstxdata = if (opts.addStart) (leftStart \ dstxdata0) else dstxdata0;
    
    mapOOV(srcdata);
    mapOOV(dstxdata);
    val srcmat = oneHot(srcdata.contents, opts.nvocab);
    val dstxmat = oneHot(dstxdata.contents, opts.nvocab);
    srcn = math.min(srcn, opts.inwidth);
    if (srcn < inwidth) initPrevCol;
    for (i <- 0 until srcn) {
      val cols = srcmat.colslice(i*batchSize, (i+1)*batchSize);
      getlayer(0, inwidth + i - srcn).output = cols;
    }
    dstxn = math.min(dstxn, opts.outwidth);
    for (i <- 0 until dstxn) {
      val cols = dstxmat.colslice(i*batchSize, (i+1)*batchSize);
      getlayer(0, inwidth + i).output = cols;
    }   
    if (leftedge.output.asInstanceOf[AnyRef] == null) {
      leftedge.output = convertMat(zeros(opts.dim, batchSize));
    }
  }
  
  def initPrevCol = {
	  for (i <- 0 until height) {
		  val leftlayer = getlayer(i+preamble_rows, inwidth-srcn-1);
		  if (leftlayer.output.asInstanceOf[AnyRef] == null) {
			  leftlayer.output = convertMat(zeros(opts.dim, batchSize));
		  }
		  if (leftlayer.outputs(1).asInstanceOf[AnyRef] == null) {
			  leftlayer.setOutput(1, convertMat(zeros(opts.dim, batchSize)));
		  }       
	  }
  }
  
  override def assignTargets(gmats:Array[Mat], ipass:Int, pos:Long) {
	  val dsty = if (gmats.length > 2) gmats(2) else gmats(1);
	  val dstyn0 = dsty.nnz/dsty.ncols;
    if (dstyn0*dsty.ncols != dsty.nnz) throw new RuntimeException("SeqToSeq dsty batch not fixed length");
    val dstydata = int(dsty.contents.view(dstyn0, batchSize).t);
    mapOOV(dstydata);
    val dstyn1 = math.min(dstyn0 - (if (opts.addStart) 0 else 1), opts.outwidth);
    for (j <- 0 until dstyn1) {
    	val incol = if (opts.addStart) dstydata.colslice(j,j+1).t else dstydata.colslice(j+1,j+2).t
    	output_layers(j).target = incol;
    }
    if (PADrow.asInstanceOf[AnyRef] == null) {
      PADrow = convertMat(iones(1, batchSize) * opts.PADsym);
    }
    dstyn = math.min(dstyn1 + 1, opts.outwidth);
    if (dstyn1 < opts.outwidth) {
    	output_layers(dstyn1).target = PADrow;
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
      for (j <- 0 until output_layers.length) {
    	  output_layers(j) match {
    	  case _:OutputLayer => {}
    	  case _ => {
    		  if (output_layers(j).deriv.asInstanceOf[AnyRef] != null) output_layers(j).deriv.set(1);
    	  }
    	  }
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
      var score = 0f
      var j = 0;
      while (j < dstxn-1) {
        score += output_layers(j).score.v;
        j += 1;
      }
      row(score/(dstxn-1))
    } else {
      zeros(1, 1);
    }
  }
}

object SeqToSeq {
  trait Opts extends Net.Opts {
    var inwidth = 1;     // Max src sentence length
    var outwidth = 1;    // Max dst sentence lenth
    var height = 1;      // Number of LSTM layers vertically
    var nvocab = 100000; // Vocabulary size
    var kind = 0;        // LSTM type, see below
    var bylevel = true;  // Use different models for each level
    var netType = 0;     // Network type, 0 = softmax output, 1 = Neg Sampling output
    var PADsym = 1;      // Padding symbol
    var OOVsym = 2;      // OOV symbol
    var STARTsym = 1;    // Start symbol
    var addStart = true; // Add the start symbol to dst sentences
    var scoreType = 0;   // Score type, 0 = LL, 1 = accuracy, 2 = LL of full Softmax, 3 = accuracy of full Softmax
    var nsamps = 100;    // Number of negative samples
    var expt = 0.8f;     // Negative sampling exponent (tail boost)
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
    
  class LearnOptions extends Learner.Options with SeqToSeq.Opts with MatSource.Opts with ADAGrad.Opts with L1Regularizer.Opts

  def learner(mat0:Mat, mat1:Mat, regularize:Boolean = false) = {
    val opts = new LearnOptions;
    opts.batchSize = 128;
  	val nn = new Learner(
  	    new MatSource(Array(mat0, mat1), opts), 
  	    new SeqToSeq(opts), 
  	    if (regularize) Array(new L1Regularizer(opts)) else null,
  	    new ADAGrad(opts), 
  	    null,
  	    opts)
    (nn, opts)
  }
  
  def learnerX(mat0:Mat, mat1:Mat) = {
    val opts = new LearnOptions;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1);
  	val nn = new Learner(
  	    new MatSource(Array(mat0, mat1), opts), 
  	    new SeqToSeq(opts), 
  	    null,
  	    null, 
  	    null,
  	    opts)
    (nn, opts)
  }
  
  class FDSopts extends Learner.Options with SeqToSeq.Opts with FileSource.Opts with ADAGrad.Opts with L1Regularizer.Opts
   
  def learner(fn1:String, fn2:String, regularize:Boolean, adagrad:Boolean):(Learner, FDSopts) = learner(List(FileSource.simpleEnum(fn1,1,0), FileSource.simpleEnum(fn2,1,0)), regularize, adagrad);
  
  def learner(fn1:String, fn2:String):(Learner, FDSopts) = learner(List(FileSource.simpleEnum(fn1,1,0), FileSource.simpleEnum(fn2,1,0)), false, true);
  
  def learnerX(fn1:String, fn2:String):(Learner, FDSopts) = learnerX(List(FileSource.simpleEnum(fn1,1,0), FileSource.simpleEnum(fn2,1,0)));
  
  def learner(fnames:List[(Int)=>String]):(Learner, FDSopts) = learner(fnames, false, true);
  
  def learner(fnames:List[(Int)=>String], regularize:Boolean, adagrad:Boolean):(Learner, FDSopts) = {   
    val opts = new FDSopts;
    opts.fnames = fnames
    opts.batchSize = 128;
    opts.eltsPerSample = 500;
    implicit val threads = threadPool(4);
    val ds = new FileSource(opts)
  	val nn = new Learner(
  			ds, 
  	    new SeqToSeq(opts), 
  	    if (regularize) Array(new L1Regularizer(opts)) else null,
  	    if (adagrad) new ADAGrad(opts) else new Grad(opts),
  	    null,
  	    opts)
    (nn, opts)
  }  
  
  def learnerX(fnames:List[(Int)=>String]):(Learner, FDSopts) = {   
    val opts = new FDSopts;
    opts.fnames = fnames
    opts.batchSize = 128;
    opts.eltsPerSample = 500;
    implicit val threads = threadPool(4);
    val ds = new FileSource(opts)
    val nn = new Learner(
        ds, 
        new SeqToSeq(opts), 
        null,
        null, 
        null,
        opts)
    (nn, opts)
  }
  
    def load(fname:String):SeqToSeq = {
      val mm = new SeqToSeq;
      mm.loadMetaData(fname);
      mm.load(fname);
      mm
    }
  
}

