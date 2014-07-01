package BIDMach.models

// Batch k-means with soft size constraint. Size weight is controlled by wsize.

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
 * val (nn, opts) = KMeans.learn(a)
 * opts.what             // prints the available options
 * opts.dim=200          // customize options
 * nn.run                // run the learner
 * nn.modelmat           // get the final model
 * 
 * val (nn, opts) = KMeans.learnPar(a) // Build a parallel learner
 * opts.nthreads=2       // number of threads (defaults to number of GPUs)
 * nn.run                // run the learner
 * nn.modelmat           // get the final model
 * }}}
 */

class KMeanspp(override val opts:KMeanspp.Opts = new KMeanspp.Options) extends Model(opts) {

  var mm:Mat = null
  var mmnorm:Mat = null
  var mweight:Mat = null
  
  var um:Mat = null
  var umcount:Mat = null
  var umweight:Mat = null
  
  
  def init() = {

    useGPU = opts.useGPU && Mat.hasCUDA > 0
    val data0 = mats(0)
    val nc = data0.ncols
    if (opts.dim > nc)
      throw new RuntimeException("KMeanspp need batchsize >= dim")

    val rp = randperm(nc)
    val mmi = full(data0(?,rp(0,0->opts.dim)))
    
    mm = if (useGPU) GMat(mmi) else mmi
    mmnorm = mm dotr mm
    mweight = mm.zeros(mm.nrows,1)
    modelmats = Array(mm, mmnorm, mweight)
    
    um = modelmats(0).zeros(mm.nrows, mm.ncols)
    umcount = mm.zeros(mm.nrows, 1)
    umweight = mm.zeros(mm.nrows, 1)
    updatemats = Array(um, umcount, umweight)
 
  } 

  
  def doblock(gmats:Array[Mat], ipass:Int, i:Long) = {
    mupdate(gmats(0), gmats(1), ipass)
  }
  
  def evalblock(gmats:Array[Mat], ipass:Int):FMat = {
    evalfun(gmats(0), gmats(1))
  }
  
  def mupdate(sdata:Mat, weights:Mat, ipass:Int):Unit = {
    val vmatch = -2 * mm * sdata + mmnorm + snorm(sdata)
    val bestm = vmatch <= mini(vmatch)
    bestm ~ bestm / sum(bestm)
    um ~ um + bestm *^ sdata     
    umcount ~ umcount + sum(bestm, 2)
    mweight ~ mweight + bestm *^ weights
  }
    
  def evalfun(sdata:Mat, weights:Mat):FMat = {  
    val vmatch = -2 * mm * sdata + mmnorm + snorm(sdata) + (opts.wsize * mweight)
    val vm = mini(vmatch)
    max(vm, 0f, vm)
    val vv = mean(sqrt(vm) *@ weights).dv
  	row(-vv, math.exp(vv))
  }
  
  override def updatePass = {
    max(umcount, 1f, umcount)
    mm ~ um / umcount
    mmnorm ~ mm dotr mm
    mweight.clear
    um.clear
    umcount.clear
  }
}

object KMeanspp  {
  trait Opts extends ClusteringModel.Opts {
    var wsize = 1e-4f
  }
  
  class Options extends Opts {}
  
  def mkKMeansModel(fopts:Model.Opts) = {
  	new KMeanspp(fopts.asInstanceOf[KMeanspp.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
  	new Batch(nopts.asInstanceOf[Batch.Opts])
  } 
   
  def learner(mat0:Mat, d:Int = 256) = {
    class xopts extends Learner.Options with KMeanspp.Opts with MatDS.Opts with Batch.Opts
    val opts = new xopts
    opts.dim = d
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    new KMeanspp(opts), 
  	    null,
  	    new Batch(opts), opts)
    (nn, opts)
  }
   
  def learnPar(mat0:Mat, d:Int = 256) = {
    class xopts extends ParLearner.Options with KMeanspp.Opts with MatDS.Opts with Batch.Opts
    val opts = new xopts
    opts.dim = d
    opts.batchSize = math.min(100000, mat0.ncols/30/opts.nthreads + 1)
    opts.coolit = 0 // Assume we dont need cooling on a matrix input
  	val nn = new ParLearnerF(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    opts, mkKMeansModel _, 
  	    null, null, 
  	    opts, mkUpdater _,
  	    opts)
    (nn, opts)
  } 
}


