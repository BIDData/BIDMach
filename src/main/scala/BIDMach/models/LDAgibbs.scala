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
 - alpha(0.1f) Dirichlet prior on document-topic weights
 - beta(0.0001f) Dirichlet prior on word-topic weights
 - nsamps(100) the number of repeated samples to take
 - useBino(true): use poisson (default) or binomial sampling (if true)
 - doDirichlet(true): explicitly sample theta params from a Dirichlet posterior
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
 
    var mm:Mat = null;
    var alpha:Mat = null;
    var beta:Mat = null;	
    var traceMem = false;
    var iupdate = 0;
  
  override def init() = {
      super.init;
      if (refresh) {
      	mm = modelmats(0);
      	setmodelmats(Array(mm, mm.ones(mm.nrows, 1), mm.zeros(mm.nrows, 1)));
      }
      updatemats = new Array[Mat](3);
      updatemats(0) = mm.zeros(mm.nrows, mm.ncols);
      updatemats(1) = mm.zeros(mm.nrows, 1);
      updatemats(2) = mm.zeros(mm.nrows, 1);
      alpha = mm.zeros(mm.nrows, 1) + convertMat(opts.alpha);
      beta = convertMat(opts.beta);
      iupdate = 0;
  }
  
  def uupdate(sdata:Mat, user:Mat, ipass: Int, pos:Long):Unit = {
    if (putBack < 0 || ipass == 0) user.set(1f);
    for (i <- 0 until opts.uiter) {
    	if (opts.doDirichlet) {                         // Here the user mat contains Dirichlet params, replace it with Dirichlet samples
    		val scaleFact = (user == user);             // Make a matrix of ones for the Dirichlet scale params.
    		gamrnd(user, scaleFact, user);              // Replace user mat with gamma random variables with unit scale params.
    		user ~ user / sum(user,1);                  // Dirichlet = normalized colums of gamma variables
    	}		
    	val preds = DDS(mm, user, sdata);	            // Probability normalizer for each word (inner product of beta and theta for that word).
    	if (traceMem) println("uupdate %d %d %d, %d %f %d" format (mm.GUID, user.GUID, sdata.GUID, preds.GUID, GPUmem._1, getGPU));
    	val dc = sdata.contents;
    	val pc = preds.contents;
    	pc ~ pc / dc;                                   // Scale preds mat by 1/word freq. Only needed by Poisson sampler. 
    	val unew = user*0;
    	val mnew = updatemats(0);
    	updatemats(0).clear;

    	LDAgibbs.LDAsample(mm, user, mnew, unew, preds, dc, opts.nsamps, opts.useBino);
    	if (traceMem) println("uupdate %d %d %d, %d %d %d %d %f %d" format (mm.GUID, user.GUID, sdata.GUID, preds.GUID, dc.GUID, pc.GUID, unew.GUID, GPUmem._1, getGPU));
    	user ~ unew + alpha;
    }
  }
  
  def mupdate(sdata:Mat, user:Mat, ipass: Int, pos:Long):Unit = {
  	val um = updatemats(0);
  	um ~ um + beta ;	 
  	sum(um, 2, updatemats(1));
  	user ~ user / sum(user,1);
  	sum(ln(user), 2, updatemats(2));
  	updatemats(2) ~ updatemats(2) * (1.0f/user.ncols);
  	if (opts.doAlpha && iupdate > 10) {
  		alpha <-- psiinv(psi(sum(alpha)) + modelmats(2));
  	}
  	iupdate += 1;
  }
  
  def evalfun(sdata:Mat, user:Mat, ipass:Int, pos:Long):FMat = {  
  	val preds = DDS(mm, user, sdata);
  	val dc = sdata.contents;
  	val pc = preds.contents;
  	max(opts.weps, pc, pc);
  	ln(pc, pc);
  	val sdat = sum(sdata,1);
  	val mms = sum(mm,2);
  	val suu = ln(mms ^* user);
  	if (traceMem) println("evalfun %d %d %d, %d %d %d, %d %f" format (sdata.GUID, user.GUID, preds.GUID, pc.GUID, sdat.GUID, mms.GUID, suu.GUID, GPUmem._1));
  	val vv = ((pc ddot dc) - (sdat ddot suu))/sum(sdat,2).dv;
  	row(vv, math.exp(-vv));
  }
  
  override def wrapUp(ipass:Int):Unit = {
    if (opts.doAlpha) {
    	modelmats(2) <-- alpha;
    }
  }
}

object LDAgibbs  {
  import edu.berkeley.bid.CUMACH
  import jcuda.runtime.JCuda._
  import jcuda.runtime.cudaError._
  import jcuda.runtime._
  
  trait Opts extends FactorModel.Opts {
      var alpha = row(0.3f);
      var beta = row(0.1f);
      var nsamps = 100f;
      var doDirichlet = true; // Explicitly do Dirichlet sampling
      var useBino = true;     // Use binomial or poisson (default) sampling
      var doAlpha = false;    // Perform updates to alpha
  }
  
  class Options extends Opts {}
  
  def LDAsample(A:Mat, B:Mat, AN:Mat, BN:Mat, C:Mat, D:Mat, nsamps:Float, doBino:Boolean):Unit = {
    (A, B, AN, BN, C, D) match {
     case (a:GMat, b:GMat, an:GMat, bn:GMat, c:GSMat, d:GMat) => doLDAgibbs(a, b, an, bn, c, d, nsamps, doBino):Unit
     case _ => throw new RuntimeException("LDAgibbs: arguments not recognized")
    }
  }

  def doLDAgibbs(A:GMat, B:GMat, AN:GMat, BN:GMat, C:GSMat, D:GMat, nsamps:Float, doBino:Boolean):Unit = {
    if (A.nrows != B.nrows || C.nrows != A.ncols || C.ncols != B.ncols || 
        A.nrows != AN.nrows || A.ncols != AN.ncols || B.nrows != BN.nrows || B.ncols != BN.ncols) {
      throw new RuntimeException("LDAgibbs dimensions mismatch")
    }
    var err = if (doBino) {
      CUMACH.LDAgibbsBino(A.nrows, C.nnz, A.data, B.data, AN.data, BN.data, C.ir, C.ic, D.data, C.data, nsamps.toInt)
    } else {
      CUMACH.LDAgibbs(A.nrows, C.nnz, A.data, B.data, AN.data, BN.data, C.ir, C.ic, C.data, nsamps)
    }
    if (err != 0) throw new RuntimeException(("GPU %d LDAgibbs kernel error "+cudaGetErrorString(err)) format getGPU)
    Mat.nflops += (if (doBino) 40L else 12L) * C.nnz * A.nrows   // Charge 10 for Poisson RNG
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
    class xopts extends Learner.Options with LDAgibbs.Opts with MatSource.Opts with IncNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = -1
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatSource(Array(mat0:Mat), opts), 
  			new LDAgibbs(opts), 
  			null,
  			new IncNorm(opts), 
  			null,
  			opts)
    (nn, opts)
  }
  
  /*
   * Batch learner
   */
  def learnBatch(mat0:Mat, d:Int = 256) = {
    class xopts extends Learner.Options with LDAgibbs.Opts with MatSource.Opts with BatchNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = -1
    opts.uiter = 2
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    val nn = new Learner(
        new MatSource(Array(mat0:Mat), opts), 
        new LDAgibbs(opts), 
        null, 
        new BatchNorm(opts),
        null,
        opts)
    (nn, opts)
  }
  
  /*
   * Parallel learner with matrix source
   */ 
  def learnPar(mat0:Mat, d:Int = 256) = {
    class xopts extends ParLearner.Options with LDAgibbs.Opts with MatSource.Opts with IncNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = -1
    opts.uiter = 5
    opts.batchSize = math.min(100000, mat0.ncols/30/opts.nthreads + 1)
    opts.coolit = 0 // Assume we dont need cooling on a matrix input
    val nn = new ParLearnerF(
        new MatSource(Array(mat0:Mat), opts), 
        opts, mkGibbsLDAmodel _, 
            null, null, 
            opts, mkUpdater _,
            null, null,
            opts)
    (nn, opts)
  }
  
  /*
   * Parallel learner with multiple file datasources
   */
  def learnFParx(
      nstart:Int=FileSource.encodeDate(2012,3,1,0), 
      nend:Int=FileSource.encodeDate(2012,12,1,0), 
      d:Int = 256
      ) = {
    class xopts extends ParLearner.Options with LDAgibbs.Opts with SFileSource.Opts with IncNorm.Opts
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
    		null, null,
        opts
    )
    (nn, opts)
  }
  
  /* 
   * Parallel learner with single file datasource
   */ 
  def learnFPar(
      nstart:Int=FileSource.encodeDate(2012,3,1,0), 
      nend:Int=FileSource.encodeDate(2012,12,1,0), 
      d:Int = 256
      ) = {   
    class xopts extends ParLearner.Options with LDAgibbs.Opts with SFileSource.Opts with IncNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.npasses = 4
    opts.resFile = "/big/twitter/test/results.mat"
    val nn = new ParLearnerF(
        Experiments.Twitter.twitterWords(nstart, nend), 
        opts, mkGibbsLDAmodel _, 
        null, null, 
        opts, mkUpdater _,
        null, null,
        opts
    )
    (nn, opts)
  }
  
}


