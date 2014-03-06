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

class KMeans(override val opts:KMeans.Opts = new KMeans.Options) extends ClusteringModel(opts) {

  var mm:Mat = null
  var um:Mat = null
  var mone:Mat = null
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    mm = modelmats(0)
    um = updatemats(0)
    mone = mm.ones(1,1)
  }
  
  def mupdate(sdata:Mat, ipass:Int):Unit = {
    val vmatch = mm * sdata 
    val bestm = vmatch >= maxi(vmatch)
    bestm ~ bestm / sum(bestm)
    um ~ um + bestm *^ sdata     
  }
    
  def evalfun(sdata:Mat):FMat = {  
    val vmatch = mm * sdata 
    val maxv = maxi(vmatch)
    val diff = 1f - maxv / max(colnorm(sdata),opts.eps) 
  	val vv = mean(sqrt(diff)).dv
  	row(-vv, math.exp(vv))
  }
}

object KMeans  {
  trait Opts extends ClusteringModel.Opts {
    var eps = 1e-5f 
  }
  
  class Options extends Opts {}
  
  def mkKMeansModel(fopts:Model.Opts) = {
  	new KMeans(fopts.asInstanceOf[KMeans.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
  	new Batch(nopts.asInstanceOf[Batch.Opts])
  } 
   
  def learn(mat0:Mat, d:Int = 256) = {
    class xopts extends Learner.Options with KMeans.Opts with MatDS.Opts with Batch.Opts
    val opts = new xopts
    opts.dim = d
    opts.blockSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    new KMeans(opts), 
  	    null,
  	    new Batch(opts), opts)
    (nn, opts)
  }
   
  def learnPar(mat0:Mat, d:Int = 256) = {
    class xopts extends ParLearner.Options with KMeans.Opts with MatDS.Opts with Batch.Opts
    val opts = new xopts
    opts.dim = d
    opts.blockSize = math.min(100000, mat0.ncols/30/opts.nthreads + 1)
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
  	opts.npasses = 4
  	opts.resFile = "/big/twitter/test/results.mat"
  	val nn = new ParLearnerxF(
  	    null, 
  	    (dopts:DataSource.Opts, i:Int) => SFilesDS.twitterWords(nstart, nend, opts.nthreads, i), 
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
  	    SFilesDS.twitterWords(nstart, nend), 
  	    opts, mkKMeansModel _, 
  	    null, null, 
  	    opts, mkUpdater _,
  	    opts
  	)
  	(nn, opts)
  }
}


