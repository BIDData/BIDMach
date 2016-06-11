package BIDMach.networks

import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.datasinks._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach.networks.layers._
import BIDMach._

/*
 * LSTM next Word prediction model, which comprises a rectangular grid of LSTM compound layers.
 */
class SeqToSeq(override val opts:SeqToSeq.Opts = new SeqToSeq.Options) extends Net(opts) {
  
  var PADrow:Mat = null
  var OOVelem:Mat = null
  var leftEdge:Layer = null
  var leftStart:Mat = null
  var dstxdata:Mat = null
  var dstxdata0:Mat = null
  var srcGrid:LayerMat = null
  var dstGrid:LayerMat = null
  var srcGridOpts:LSTMNode.GridOpts = null
  var dstGridOpts:LSTMNode.GridOpts = null
  var height = 0
  var inwidth = 0
  var outwidth = 0
  var width = 0
  var srcn = 0
  var dstxn = 0
  var dstyn = 0
  val preamble_rows = 2
  
  override def createLayers = {
    height = opts.height
    inwidth = opts.inwidth; 
    outwidth = opts.outwidth
    leftEdge = InputLayer(this);                     // dummy layer, left edge of zeros   
    
    srcGridOpts = new LSTMNode.GridOpts
    srcGridOpts.copyFrom(opts)
    srcGridOpts.modelName = "src_level%d"
    srcGridOpts.netType = LSTMNode.gridTypeNoOutput
    srcGrid = LSTMLayer.grid(this, height, inwidth, srcGridOpts)
    layers = srcGrid.data.filter(_ != null)
    for (i <- 0 until height) srcGrid(i+preamble_rows, 0).setInputs(leftEdge, leftEdge)
    
    if (! opts.embed) {
      dstGridOpts = new LSTMNode.GridOpts
      dstGridOpts.copyFrom(opts)
      dstGridOpts.modelName = "dst_level%d"
      dstGridOpts.netType = LSTMNode.gridTypeSoftmaxOutput
      dstGridOpts.outdim = opts.nvocab
      dstGrid = LSTMLayer.grid(this, height, outwidth, dstGridOpts)

      srcGrid link dstGrid
      layers = layers ++ dstGrid.data.filter(_ != null)
      output_layers = new Array[Layer](outwidth)
      for (i <- 0 until outwidth) output_layers(i) = dstGrid(dstGrid.nrows-1, i)
    }
  }
  
  def mapOOV(in:Mat) = {
    if (OOVelem.asInstanceOf[AnyRef] == null) {
      OOVelem = convertMat(iones(1,1) * opts.OOVsym)
    }
    in ~ in + ((in >= opts.nvocab) ∘ (OOVelem - in))
  }
  
  override def assignInputs(gmats:Array[Mat], ipass:Int, pos:Long) = {
    val src = gmats(0)
    srcn = src.nnz/src.ncols
    if (srcn*src.ncols != src.nnz) throw new RuntimeException("SeqToSeq src batch not fixed length")
    val srcdata = int(src.contents.view(srcn, batchSize).t);   // IMat with columns corresponding to word positions, with batchSize rows.
    mapOOV(srcdata)
    val srcmat = oneHot(srcdata.contents, opts.nvocab)
    srcn = math.min(srcn, opts.inwidth)
    if (srcn < inwidth) initPrevCol
    for (i <- 0 until srcn) {
      val cols = srcmat.colslice(i*batchSize, (i+1)*batchSize)
      srcGrid(0, inwidth + i - srcn).output = cols
    }
    if (leftEdge.output.asInstanceOf[AnyRef] == null) {
      leftEdge.output = convertMat(zeros(opts.dim \ batchSize))
    }
    
    if (! opts.embed) {
      val dstx = gmats(1)
      val dstxn0 = dstx.nnz/dstx.ncols
      if (dstxn0*dstx.ncols != dstx.nnz) throw new RuntimeException("SeqToSeq dstx batch not fixed length"); 
      val dstxdata0 = int(dstx.contents.view(dstxn0, batchSize).t)
      dstxn = dstxn0 + (if (opts.addStart) 1 else 0)
      if (opts.addStart && (leftStart.asInstanceOf[AnyRef] == null)) {
        leftStart = convertMat(izeros(batchSize, 1))
      }
      val dstxdata = if (opts.addStart) (leftStart \ dstxdata0) else dstxdata0
      mapOOV(dstxdata)
      val dstxmat = oneHot(dstxdata.contents, opts.nvocab)

      dstxn = math.min(dstxn, opts.outwidth)
      for (i <- 0 until dstxn) {
        val cols = dstxmat.colslice(i*batchSize, (i+1)*batchSize)
        dstGrid(0, i).output = cols
      }   
    }
  }
  
  def initPrevCol = {
    for (i <- 0 until height) {
      val leftlayer = srcGrid(i+preamble_rows, inwidth-srcn-1)
      if (leftlayer.output.asInstanceOf[AnyRef] == null) {
        leftlayer.output = convertMat(zeros(opts.dim \ batchSize))
      }
      leftlayer.output.clear
      if (leftlayer.outputs(1).asInstanceOf[AnyRef] == null) {
        leftlayer.setOutput(1, convertMat(zeros(opts.dim \ batchSize)))
      }   
      leftlayer.outputs(1).clear
    }
  }
  
  override def assignTargets(gmats:Array[Mat], ipass:Int, pos:Long) {
    val dsty = if (gmats.length > 2) gmats(2) else gmats(1)
    val dstyn0 = dsty.nnz/dsty.ncols
    if (dstyn0*dsty.ncols != dsty.nnz) throw new RuntimeException("SeqToSeq dsty batch not fixed length")
    val dstydata = int(dsty.contents.view(dstyn0, batchSize).t)
    mapOOV(dstydata)
    val dstyn1 = math.min(dstyn0 - (if (opts.addStart) 0 else 1), opts.outwidth)
    for (j <- 0 until dstyn1) {
      val incol = if (opts.addStart) dstydata.colslice(j,j+1).t else dstydata.colslice(j+1,j+2).t
      output_layers(j).target = incol
    }
    if (PADrow.asInstanceOf[AnyRef] == null) {
      PADrow = convertMat(iones(1, batchSize) * opts.PADsym)
    }
    dstyn = math.min(dstyn1 + 1, opts.outwidth)
    if (dstyn1 < opts.outwidth) {
      output_layers(dstyn1).target = PADrow
    }
  }
  
  override def dobatch(gmats:Array[Mat], ipass:Int, pos:Long):Unit = {
    if (batchSize < 0) batchSize = gmats(0).ncols
    if (batchSize == gmats(0).ncols) {                                    // discard odd-sized minibatches
      assignInputs(gmats, ipass, pos)
      assignTargets(gmats, ipass, pos)
      if (mask.asInstanceOf[AnyRef] != null) {
        modelmats(0) ~ modelmats(0) ∘ mask
      }
      val mincol = inwidth - srcn
      val maxcol = dstxn
      srcGrid.forward(mincol, inwidth-1, opts.debug)
      dstGrid.forward(0, maxcol-1, opts.debug)
 
      output_layers.map((layer:Layer) => layer match {
        case _:OutputLayer => {}
        case _ => {if (layer.deriv.asInstanceOf[AnyRef] != null) layer.deriv.set(1);}
      })
      if (opts.aopts == null) updatemats.map(_.clear)

      dstGrid.backward(0, maxcol-1, opts.debug, ipass, pos)
      srcGrid.backward(mincol, inwidth-1, opts.debug, ipass, pos)
    }
  }
  
  override def evalbatch(mats:Array[Mat], ipass:Int, pos:Long):FMat = {  
    if (batchSize < 0) batchSize = gmats(0).ncols
    if (batchSize == gmats(0).ncols) { 
      assignInputs(gmats, ipass, pos)
      if (mask.asInstanceOf[AnyRef] != null) {
        modelmats(0) ~ modelmats(0) ∘ mask
      }
      val mincol = inwidth - srcn; 
      srcGrid.forward(mincol, inwidth-1, opts.debug)
      if (! opts.embed) {
        val maxcol = dstxn
        assignTargets(gmats, ipass, pos)
        dstGrid.forward(0, maxcol-1, opts.debug);     
        if (putBack >= 0) {
          output_layers(dstxn-1).output.colslice(0, gmats(0).ncols, gmats(1))
        }
        var score = 0f
        var j = 0
        while (j < dstxn-1) {
          score += output_layers(j).score.v
          j += 1
        }
        row(score/(dstxn-1))
      } else {
        if (ogmats != null) {
          var embedding = srcGrid(height+preamble_rows-1, srcGrid.ncols-1).output.asMat
          for (j <- 1 until opts.nembed) {
            embedding = embedding on srcGrid(height-j+preamble_rows-1, srcGrid.ncols-1).output.asMat
          }
          ogmats(0) = embedding
        }
        zeros(1,1)
      }
    } else {
      zeros(1, 1)
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
    var embed = false;   // Whether to compute an embedding (vs. a model)
    var nembed = 1;      // number of layers (counted from the top) to use for the embedding
    
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
    val opts = new LearnOptions
    opts.batchSize = 128
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
    val opts = new LearnOptions
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
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
   
  def learner(fn1:String, fn2:String, regularize:Boolean, adagrad:Boolean):(Learner, FDSopts) = learner(List(FileSource.simpleEnum(fn1,1,0), FileSource.simpleEnum(fn2,1,0)), regularize, adagrad)
  
  def learner(fn1:String, fn2:String):(Learner, FDSopts) = learner(List(FileSource.simpleEnum(fn1,1,0), FileSource.simpleEnum(fn2,1,0)), false, true)
  
  def learnerX(fn1:String, fn2:String):(Learner, FDSopts) = learnerX(List(FileSource.simpleEnum(fn1,1,0), FileSource.simpleEnum(fn2,1,0)))
  
  def learner(fnames:List[(Int)=>String]):(Learner, FDSopts) = learner(fnames, false, true)
  
  def learner(fnames:List[(Int)=>String], regularize:Boolean, adagrad:Boolean):(Learner, FDSopts) = {   
    val opts = new FDSopts
    opts.fnames = fnames
    opts.batchSize = 128
    opts.eltsPerSample = 500
    implicit val threads = threadPool(4)
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
    val opts = new FDSopts
    opts.fnames = fnames
    opts.batchSize = 128
    opts.eltsPerSample = 500
    implicit val threads = threadPool(4)
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
  
  class FEopts extends Learner.Options with SeqToSeq.Opts with FileSource.Opts with FileSink.Opts
  
  def embed(model:SeqToSeq, ifname:String, ofname:String):(Learner, FEopts) = {   
    val opts = new FEopts
    opts.copyFrom(model.opts)
    opts.fnames = List(FileSource.simpleEnum(ifname,1,0))
    opts.ofnames = List(FileSource.simpleEnum(ofname,1,0))
    opts.embed = true
    val newmod = new SeqToSeq(opts)
    newmod.refresh = false
    model.copyTo(newmod)
    implicit val threads = threadPool(4)
    val ds = new FileSource(opts)
    val nn = new Learner(
        new FileSource(opts), 
        newmod, 
        null,
        null,
        new FileSink(opts),
        opts)
    (nn, opts)
  }
  
  def load(fname:String):SeqToSeq = {
    val mm = new SeqToSeq
    mm.loadMetaData(fname)
    mm.load(fname)
    mm
  }

}

