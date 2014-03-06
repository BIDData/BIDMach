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
 * opts.uiter=2          // customize options
 * nn.run                // run the learner
 * nn.modelmat           // get the final model
 * nn.datamat            // get the other factor (requires opts.putBack=1)
 * 
 * val (nn, opts) = KMeans.learnPar(a) // Build a parallel learner
 * opts.nthreads=2       // number of threads (defaults to number of GPUs)
 * nn.run                // run the learner
 * nn.modelmat           // get the final model
 * nn.datamat            // get the other factor
 * }}}
 */

class KMeans(override val opts:KMeans.Opts = new KMeans.Options) extends FactorModel(opts) {

  var mm:Mat = null
  var um:Mat = null
  var alpha:Mat = null 
  var traceMem = false
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    mm = modelmats(0)
    updatemats = new Array[Mat](1)
    updatemats(0) = mm.zeros(mm.ncols, mm.nrows)
    um = updatemats(0)
  }
  
  def uupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    val vmatch = mm * sdata 
    val (maxv, maxp) = maxi2(vmatch)
    um(?,maxp) = um(?,maxp) + sdata    
  }
  
  def mupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    val ip = sqrt(um dot um)
    mm = (um / ip).t
  }
  
  def evalfun(sdata:Mat, user:Mat):FMat = {  
    val ssd = max(sqrt(sum(sdata)), 1f)
    val vmatch = mm * sdata 
    val (maxv, maxp) = maxi2(vmatch)
    val diff = um(?,maxp) - full(sdata) / ssd 
  	val vv = mean(sqrt(diff dot diff)).dv
  	row(vv, math.exp(-vv))
  }
}

object KMeans  {
  trait Opts extends FactorModel.Opts {

  }
  
  class Options extends Opts {}
  
  def mkKMeansModel(fopts:Model.Opts) = {
  	new KMeans(fopts.asInstanceOf[KMeans.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
  	new IncNorm(nopts.asInstanceOf[IncNorm.Opts])
  } 
   
  /**
   * Online Variational Bayes LDA algorithm
   */
  def learn(mat0:Mat, d:Int = 256) = {
    class xopts extends Learner.Options with KMeans.Opts with MatDS.Opts with IncNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = 1
    opts.uiter = 2
    opts.blockSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    new KMeans(opts), 
  	    null,
  	    new IncNorm(opts), opts)
    (nn, opts)
  }
     
  /**
   * Batch Variational Bayes LDA algorithm
   */
  def learnBatch(mat0:Mat, d:Int = 256) = {
    class xopts extends Learner.Options with KMeans.Opts with MatDS.Opts with BatchNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = 1
    opts.uiter = 2
    opts.blockSize = math.min(100000, mat0.ncols/30 + 1)
    val nn = new Learner(
        new MatDS(Array(mat0:Mat), opts), 
        new KMeans(opts), 
        null, 
        new BatchNorm(opts),
        opts)
    (nn, opts)
  }
  
  /**
   * Parallel online LDA algorithm
   */ 
  def learnPar(mat0:Mat, d:Int = 256) = {
    class xopts extends ParLearner.Options with KMeans.Opts with MatDS.Opts with IncNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = -1
    opts.uiter = 5
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
  
  /**
   * Parallel online LDA algorithm with multiple file datasources
   */ 
  def learnFParx(
      nstart:Int=FilesDS.encodeDate(2012,3,1,0), 
      nend:Int=FilesDS.encodeDate(2012,12,1,0), 
      d:Int = 256
      ) = {
  	class xopts extends ParLearner.Options with KMeans.Opts with SFilesDS.Opts with IncNorm.Opts
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
  
  /**
   * Parallel online LDA algorithm with one file datasource
   */
  def learnFPar(
      nstart:Int=FilesDS.encodeDate(2012,3,1,0), 
      nend:Int=FilesDS.encodeDate(2012,12,1,0), 
      d:Int = 256
      ) = {	
  	class xopts extends ParLearner.Options with KMeans.Opts with SFilesDS.Opts with IncNorm.Opts
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


