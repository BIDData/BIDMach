package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._

/**
 * Latent Dirichlet Model using repeated Gibbs sampling. 
 * 
 * Extends Factor Model Options with:
 - dim(256): Model dimension
 - uiter(5): Number of iterations on one block of data
 - alpha(0.001f) Dirichlet prior on document-topic weights
 - beta(0.0001f) Dirichlet prior on word-topic weights
 - nsamps(100) the number of repeated samples to take
 *
 * Other key parameters inherited from the learner, datasource and updater:
 - batchSize: the number of samples processed in a block
 - power(0.3f): the exponent of the moving average model' = a dmodel + (1-a)*model, a = 1/nblocks^power
 - npasses(10): number of complete passes over the dataset
 *     
 * '''Example:'''
 * 
 * a is a sparse word x document matrix
 * {{{
 * val (nn, opts) = LDAgibbs.learn(a)
 * opts.what             // prints the available options
 * opts.uiter=2          // customize options
 * nn.run                // run the learner
 * nn.modelmat           // get the final model
 * nn.datamat            // get the other factor (requires opts.putBack=1)
 *  
 * val (nn, opts) = LDAgibbs.learnPar(a) // Build a parallel learner
 * opts.nthreads = 2     // number of threads (defaults to number of GPUs)
 * nn.run                // run the learner
 * nn.modelmat           // get the final model
 * nn.datamat            // get the other factor
 * }}}
 * 
 */

class LDAgibbs(override val opts:LDAgibbs.Opts = new LDAgibbs.Options) extends FactorModel(opts) {
 
  var mm:Mat = null
  var alpha:Mat = null 
  var traceMem = false
  
  override def init() = {
    super.init
    if (refresh) {
    	mm = modelmats(0);
    	setmodelmats(Array(mm, mm.ones(mm.nrows, 1)));
    }
    updatemats = new Array[Mat](2)
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols)
    updatemats(1) = mm.zeros(mm.nrows, 1)
  }
  
  def uupdate(sdata:Mat, user:Mat, ipass: Int):Unit = {
    
    	if (putBack < 0 || ipass == 0) user.set(1f)
        for (i <- 0 until opts.uiter) yield {
    	val preds = DDS(mm, user, sdata)	
    	if (traceMem) println("uupdate %d %d %d, %d %f %d" format (mm.GUID, user.GUID, sdata.GUID, preds.GUID, GPUmem._1, getGPU))
    	val dc = sdata.contents
    	val pc = preds.contents
    	pc ~ pc / dc
    	
    	val unew = user*0
    	val mnew = updatemats(0)
    	mnew.set(0f)
  
    	LDAgibbs.LDAsample(mm, user, mnew, unew, preds, opts.nsamps)
        
    	if (traceMem) println("uupdate %d %d %d, %d %d %d %d %f %d" format (mm.GUID, user.GUID, sdata.GUID, preds.GUID, dc.GUID, pc.GUID, unew.GUID, GPUmem._1, getGPU))
    	user ~ unew + opts.alpha
    	}
  
  }
  
  def mupdate(sdata:Mat, user:Mat, ipass: Int):Unit = {
	val um = updatemats(0)
	um ~ um + opts.beta 
  	sum(um, 2, updatemats(1))
  }
  
  def evalfun(sdata:Mat, user:Mat, ipass:Int):FMat = {  
  	val preds = DDS(mm, user, sdata)
  	val dc = sdata.contents
  	val pc = preds.contents
  	max(opts.weps, pc, pc)
  	ln(pc, pc)
  	val sdat = sum(sdata,1)
  	val mms = sum(mm,2)
  	val suu = ln(mms ^* user)
  	if (traceMem) println("evalfun %d %d %d, %d %d %d, %d %f" format (sdata.GUID, user.GUID, preds.GUID, pc.GUID, sdat.GUID, mms.GUID, suu.GUID, GPUmem._1))
  	val vv = ((pc ddot dc) - (sdat ddot suu))/sum(sdat,2).dv
  	row(vv, math.exp(-vv))
  }
}

object LDAgibbs  {
  import edu.berkeley.bid.CUMACH
  import jcuda.runtime.JCuda._
  import jcuda.runtime.cudaError._
  import jcuda.runtime._
  
  trait Opts extends FactorModel.Opts {
    var alpha = 0.1f
    var beta = 0.1f
    var nsamps = 100f
  }
  
  class Options extends Opts {}
  
   def LDAsample(A:Mat, B:Mat, AN:Mat, BN:Mat, C:Mat, nsamps:Float):Unit = {
    (A, B, AN, BN, C) match {
     case (a:GMat, b:GMat, an:GMat, bn:GMat, c:GSMat) => doLDAgibbs(a, b, an, bn, c, nsamps):Unit
     case _ => throw new RuntimeException("LDAgibbs: arguments not recognized")
    }
  }

  def doLDAgibbs(A:GMat, B:GMat, AN:GMat, BN:GMat, C:GSMat, nsamps:Float):Unit = {
    if (A.nrows != B.nrows || C.nrows != A.ncols || C.ncols != B.ncols || 
        A.nrows != AN.nrows || A.ncols != AN.ncols || B.nrows != BN.nrows || B.ncols != BN.ncols) {
      throw new RuntimeException("LDAgibbs dimensions mismatch")
    }
    var err = CUMACH.LDAgibbs(A.nrows, C.nnz, A.data, B.data, AN.data, BN.data, C.ir, C.ic, C.data, nsamps)
    if (err != 0) throw new RuntimeException(("GPU %d LDAgibbs kernel error "+cudaGetErrorString(err)) format getGPU)
    Mat.nflops += 12L * C.nnz * A.nrows   // Charge 10 for Poisson RNG
  }
  
  def doLDAgibbsx(A:GMat, B:GMat, C:GSMat, Ms:GIMat, Us:GIMat):Unit = {
    if (A.nrows != B.nrows || C.nrows != A.ncols || C.ncols != B.ncols || C.nnz != Ms.ncols || C.nnz != Us.ncols || Ms.nrows != Us.nrows) {
      throw new RuntimeException("LDAgibbsx dimensions mismatch")
    }


    Mat.nflops += 12L * C.nnz * A.nrows    // Charge 10 for Poisson RNG
  }
  
  def mkGibbsLDAmodel(fopts:Model.Opts) = {
  	new LDAgibbs(fopts.asInstanceOf[LDAgibbs.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
  	new IncNorm(nopts.asInstanceOf[IncNorm.Opts])
  } 
  
  /*
   * This learner uses stochastic updates (like the standard LDA model)
   */
  def learner(mat0:Mat, d:Int = 256) = {
    class xopts extends Learner.Options with LDAgibbs.Opts with MatDS.Opts with IncNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = 1
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts), 
  			new LDAgibbs(opts), 
  			null,
  			new IncNorm(opts), opts)
    (nn, opts)
  }
  
  /*
   * Batch learner
   */
  def learnBatch(mat0:Mat, d:Int = 256) = {
    class xopts extends Learner.Options with LDAgibbs.Opts with MatDS.Opts with BatchNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = 1
    opts.uiter = 2
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    val nn = new Learner(
        new MatDS(Array(mat0:Mat), opts), 
        new LDAgibbs(opts), 
        null, 
        new BatchNorm(opts),
        opts)
    (nn, opts)
  }
  
  /*
   * Parallel learner with matrix source
   */ 
  def learnPar(mat0:Mat, d:Int = 256) = {
    class xopts extends ParLearner.Options with LDAgibbs.Opts with MatDS.Opts with IncNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = -1
    opts.uiter = 5
    opts.batchSize = math.min(100000, mat0.ncols/30/opts.nthreads + 1)
    opts.coolit = 0 // Assume we dont need cooling on a matrix input
    val nn = new ParLearnerF(
        new MatDS(Array(mat0:Mat), opts), 
        opts, mkGibbsLDAmodel _, 
            null, null, 
            opts, mkUpdater _,
            opts)
    (nn, opts)
  }
  
  /*
   * Parallel learner with multiple file datasources
   */
  def learnFParx(
      nstart:Int=FilesDS.encodeDate(2012,3,1,0), 
      nend:Int=FilesDS.encodeDate(2012,12,1,0), 
      d:Int = 256
      ) = {
    class xopts extends ParLearner.Options with LDAgibbs.Opts with SFilesDS.Opts with IncNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.npasses = 4
    opts.resFile = "/big/twitter/test/results.mat"
    val nn = new ParLearnerxF(
        null, 
            (dopts:DataSource.Opts, i:Int) => Experiments.Twitter.twitterWords(nstart, nend, opts.nthreads, i), 
            opts, mkGibbsLDAmodel _, 
            null, null, 
        opts, mkUpdater _,
        opts
    )
    (nn, opts)
  }
  
  /* 
   * Parallel learner with single file datasource
   */ 
  def learnFPar(
      nstart:Int=FilesDS.encodeDate(2012,3,1,0), 
      nend:Int=FilesDS.encodeDate(2012,12,1,0), 
      d:Int = 256
      ) = {   
    class xopts extends ParLearner.Options with LDAgibbs.Opts with SFilesDS.Opts with IncNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.npasses = 4
    opts.resFile = "/big/twitter/test/results.mat"
    val nn = new ParLearnerF(
        Experiments.Twitter.twitterWords(nstart, nend), 
        opts, mkGibbsLDAmodel _, 
        null, null, 
        opts, mkUpdater _,
        opts
    )
    (nn, opts)
  }
  
}


