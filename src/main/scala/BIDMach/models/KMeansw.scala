package BIDMach.models

// Minibatch k-means with soft size constraint. 
// Includes a size weight matrix w. Size of a cluster is the sum of the w values for that cluster. 
// Size weight is controlled by wsize.

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._
import BIDMach.models._

/**
 * KMeans
 * {{{
 * val (nn, opts) = KMeansw.learner(a,w)
 * val (nn, opts) = KMeansw.learner(a)
 * a                     // input matrix
 * w                     // optional weight matrix
 * opts.what             // prints the available options
 * opts.dim=200          // customize options
 * nn.train              // train the learner
 * nn.modelmat           // get the final model
 * 
 * val (nn, opts) = KMeansw.learnPar(a,w) // Build a parallel learner
 * val (nn, opts) = KMeansw.learnPar(a) 
 * opts.nthreads=2       // number of threads (defaults to number of GPUs)
 * nn.train              // train the learner
 * nn.modelmat           // get the final model
 * }}}
 */

class KMeansw(override val opts:KMeansw.Opts = new KMeansw.Options) extends Model(opts) {

  var mm:Mat = null
  var mcounts:Mat = null
  var mweights:Mat = null
  
  var um:Mat = null
  var umcounts:Mat = null
  var umweights:Mat = null
  
  
  def init() = {
    useGPU = opts.useGPU && Mat.hasCUDA > 0
    val data0 = mats(0)
    val nc = data0.ncols
    if (opts.dim > nc)
      throw new RuntimeException("KMeansw need batchsize >= dim")
    
    if (refresh) {
    	val rp = randperm(nc);
    	val mmi = full(data0(?,rp(0,0->opts.dim))).t;
    	mm = convertMat(mmi);
    	mcounts = mm.zeros(mm.nrows, 1);
    	mweights = mm.zeros(mm.nrows, 1);
    	setmodelmats(Array(mm, mcounts, mweights));
    }
    for (i <- 0 until 3) modelmats(i) = convertMat(modelmats(i));
    um = modelmats(0).zeros(mm.nrows, mm.ncols)
    umcounts = mm.zeros(mm.nrows, 1)
    umweights = mm.zeros(mm.nrows, 1)
    updatemats = Array(um, umcounts, umweights)
 
  } 

  
  def dobatch(gmats:Array[Mat], ipass:Int, i:Long) = {
    if (gmats.length > 1) {
      mupdate(gmats(0), gmats(1), ipass)
    } else {
      mupdate(gmats(0), null, ipass)
    }
  }
  
  def evalbatch(gmats:Array[Mat], ipass:Int, here:Long):FMat = {
    if (gmats.length > 1) {
      evalfun(gmats(0), gmats(1))
    } else {
      evalfun(gmats(0), null)
    }
  }
  
  def mupdate(sdata:Mat, weights:Mat, ipass:Int):Unit = {
    val vmatch = -2 * mm * sdata + snorm(sdata) + ((mm dotr mm) + (opts.wsize * mweights));
    val bestm = vmatch <= mini(vmatch);
    bestm ~ bestm / sum(bestm);
    um ~ bestm *^ sdata;
    sum(bestm, 2, umcounts);
    if (weights.asInstanceOf[AnyRef] != null) {
      umweights ~ bestm *^ weights
    } else {
      sum(bestm, 1, umweights)
    }
  }
    
  def evalfun(sdata:Mat, weights:Mat):FMat = {  
    val vmatch = -2 * mm * sdata + snorm(sdata) + ((mm dotr mm) + (opts.wsize * mweights))
    val vm = mini(vmatch)
    max(vm, 0f, vm)
    val vv = if (weights.asInstanceOf[AnyRef] != null) {
      mean(sqrt(vm) *@ weights).dv
    } else {
      mean(sqrt(vm)).dv
    }
  	row(-vv, math.exp(vv))
  }
  
  override def updatePass(ipass:Int) = {
    if (ipass > 0) {
      max(umcounts, 1f, umcounts);
      mm ~ um / umcounts;
      mweights <-- umweights;
      um.clear;
      umcounts.clear;
      umweights.clear;
    }
  }
}

object KMeansw  {
  trait Opts extends ClusteringModel.Opts {
    var wsize = 1e-4f
  }
  
  class Options extends Opts {}
  
  def mkKMeansModel(fopts:Model.Opts) = {
  	new KMeansw(fopts.asInstanceOf[KMeansw.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
  	new IncNorm(nopts.asInstanceOf[IncNorm.Opts])
  } 
  
  class FsOpts extends Learner.Options with KMeansw.Opts with FilesDS.Opts with IncNorm.Opts
  
  class MemOpts extends Learner.Options with KMeansw.Opts with MatDS.Opts with IncNorm.Opts
   
  def learner(datamat:Mat, wghts:Mat, d:Int) = {
    val opts = new MemOpts
    opts.dim = d
    opts.batchSize = math.min(100000, datamat.ncols/30 + 1)
    opts.isprob = false
    opts.power = 0.5f
  	val nn = new Learner(
  	    new MatDS(Array(datamat, wghts), opts), 
  	    new KMeansw(opts), 
  	    null,
  	    new IncNorm(opts), opts)
    (nn, opts)
  }
  
  def learner(datamat:Mat, d:Int) = {
    val opts = new MemOpts
    opts.dim = d
    opts.batchSize = math.min(100000, datamat.ncols/30 + 1)
    opts.isprob = false
    opts.power = 0.5f
    val nn = new Learner(
        new MatDS(Array(datamat), opts), 
        new KMeansw(opts), 
        null,
        new IncNorm(opts), opts)
    (nn, opts)
  } 
   
  // This function constructs a predictor from an existing model 
  def predictor(model:Model, mat1:Mat, preds:Mat, d:Int):(Learner, MemOpts) = {
    val nopts = new MemOpts;
    nopts.batchSize = math.min(10000, mat1.ncols/30 + 1)
    nopts.putBack = 1
    val newmod = new KMeansw(nopts);
    newmod.refresh = false
    model.copyTo(newmod)
    val nn = new Learner(
        new MatDS(Array(mat1, preds), nopts), 
        newmod, 
        null,
        null,
        nopts)
    (nn, nopts)
  }
   
  def learnPar(mat0:Mat, d:Int = 256) = {
    class xopts extends ParLearner.Options with KMeansw.Opts with MatDS.Opts with IncNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.batchSize = math.min(100000, mat0.ncols/30/opts.nthreads + 1)
    opts.coolit = 0 // Assume we dont need cooling on a matrix input
    opts.power = 0.5f
  	val nn = new ParLearnerF(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    opts, mkKMeansModel _, 
  	    null, null, 
  	    opts, mkUpdater _,
  	    opts)
    (nn, opts)
  } 
}


