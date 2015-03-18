package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._

/**
 * KMeans
 * {{{
 * val (nn, opts) = KMeans.learner(a)
 * opts.what             // prints the available options
 * opts.dim=200          // customize options
 * nn.train              // rain the learner
 * nn.modelmat           // get the final model
 * 
 * val (nn, opts) = KMeans.learnPar(a) // Build a parallel learner
 * opts.nthreads=2       // number of threads (defaults to number of GPUs)
 * nn.train              // train the learner
 * nn.modelmat           // get the final model
 * }}}
 */

class KMeans(override val opts:KMeans.Opts = new KMeans.Options) extends ClusteringModel(opts) {

  var mm:Mat = null
  var mmnorm:Mat = null
  var um:Mat = null
  var umcount:Mat = null
  
  override def init() = {
    super.init()
    mm = modelmats(0);
    if (refresh) {
    	mmnorm = mm dotr mm;
    	setmodelmats(Array(mm, mmnorm));
    }
    for (i <- 0 until modelmats.length) modelmats(i) = convertMat(modelmats(i))
    mm = modelmats(0)
    mmnorm = modelmats(1)
    um = updatemats(0)
    umcount = mm.zeros(mm.nrows, 1)
    updatemats = Array(um, umcount)
  }
  
  def mupdate(sdata:Mat, ipass:Int):Unit = {
    val vmatch = -2 * mm * sdata + mmnorm + snorm(sdata)
    val bestm = vmatch <= mini(vmatch)
    bestm ~ bestm / sum(bestm)
    um ~ um + bestm *^ sdata     
    umcount ~ umcount + sum(bestm, 2)
  }
    
  def evalfun(sdata:Mat):FMat = {  
    val vmatch = -2 * mm * sdata + mmnorm + snorm(sdata)  
    val vm = mini(vmatch)
    max(vm, 0f, vm)
    val vv = mean(sqrt(vm)).dv
  	row(-vv, math.exp(vv))
  }
  
  override def evalfun(sdata:Mat, targ:Mat):FMat = {  
    val vmatch = -2 * mm * sdata + mmnorm + snorm(sdata)  
    val (vm, im) = mini2(vmatch)
    if (putBack >= 0) {targ <-- im}
    max(vm, 0f, vm)
    val vv = mean(sqrt(vm)).dv
  	row(-vv, math.exp(vv))
  }
  
  override def updatePass(ipass:Int) = {
    if (ipass > 0) {
      max(umcount, 1f, umcount);
      mm ~ um / umcount;
      um.clear;
      umcount.clear;
    }
    mmnorm ~ mm dotr mm;
  }
}

object KMeans  {
  trait Opts extends ClusteringModel.Opts {
  }
  
  class Options extends Opts {}
  
  def mkKMeansModel(fopts:Model.Opts) = {
  	new KMeans(fopts.asInstanceOf[KMeans.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
  	new Batch(nopts.asInstanceOf[Batch.Opts])
  } 
   
  def learner(mat0:Mat, d:Int = 256) = {
    class xopts extends Learner.Options with KMeans.Opts with MatDS.Opts with Batch.Opts
    val opts = new xopts
    opts.dim = d
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    opts.npasses = 10
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    new KMeans(opts), 
  	    null,
  	    new Batch(opts), opts)
    (nn, opts)
  }
  class fsopts extends Learner.Options with KMeans.Opts with FilesDS.Opts with Batch.Opts
  
  class memopts extends Learner.Options with KMeans.Opts with MatDS.Opts with Batch.Opts
  /**
   * KMeans with a files dataSource
   */
  def learner(fnames:List[(Int)=>String], d:Int) = {
    val opts = new fsopts
    opts.dim = d
    opts.fnames = fnames
    opts.batchSize = 10000;
    implicit val threads = threadPool(4)
  	val nn = new Learner(
  	    new FilesDS(opts), 
  	    new KMeans(opts), 
  	    null,
  	    new Batch(opts), 
  	    opts)
    (nn, opts)
  }
  
  def learner(fnames:String, d:Int):(Learner, fsopts) = learner(List(FilesDS.simpleEnum(fnames,1,0)), d) 
  
    // This function constructs a predictor from an existing model 
  def predictor(model:Model, mat1:Mat, preds:Mat, d:Int):(Learner, memopts) = {
    val nopts = new memopts;
    nopts.batchSize = math.min(10000, mat1.ncols/30 + 1)
    nopts.putBack = 1
    val newmod = new KMeans(nopts);
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
    class xopts extends ParLearner.Options with KMeans.Opts with MatDS.Opts with Batch.Opts
    val opts = new xopts
    opts.dim = d
    opts.batchSize = math.min(100000, mat0.ncols/30/opts.nthreads + 1)
    opts.npasses = 10
    opts.coolit = 0 // Assume we dont need cooling on a matrix input
  	val nn = new ParLearnerF(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    opts, mkKMeansModel _, 
  	    null, null, 
  	    opts, mkUpdater _,
  	    opts)
    (nn, opts)
  }
  
  def learnFParx(
      nstart:Int=FilesDS.encodeDate(2012,3,1,0), 
      nend:Int=FilesDS.encodeDate(2012,12,1,0), 
      d:Int = 256
      ) = {
  	class xopts extends ParLearner.Options with KMeans.Opts with SFilesDS.Opts with Batch.Opts
  	val opts = new xopts
  	opts.dim = d
  	opts.npasses = 10
  	opts.resFile = "/big/twitter/test/results.mat"
  	val nn = new ParLearnerxF(
  	    null, 
  	    (dopts:DataSource.Opts, i:Int) => Experiments.Twitter.twitterWords(nstart, nend, opts.nthreads, i), 
  	    opts, mkKMeansModel _, 
  	    null, null, 
  	    opts, mkUpdater _,
  	    opts
  	)
  	(nn, opts) 
  }
  
  def learnFPar(
      nstart:Int=FilesDS.encodeDate(2012,3,1,0), 
      nend:Int=FilesDS.encodeDate(2012,12,1,0), 
      d:Int = 256
      ) = {	
  	class xopts extends ParLearner.Options with KMeans.Opts with SFilesDS.Opts with Batch.Opts
  	val opts = new xopts
  	opts.dim = d
  	opts.npasses = 4
  	opts.resFile = "/big/twitter/test/results.mat"
  	val nn = new ParLearnerF(
  	    Experiments.Twitter.twitterWords(nstart, nend), 
  	    opts, mkKMeansModel _, 
  	    null, null, 
  	    opts, mkUpdater _,
  	    opts
  	)
  	(nn, opts)
  }
}


