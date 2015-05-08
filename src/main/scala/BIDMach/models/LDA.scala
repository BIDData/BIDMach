package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._

/**
 * LDA model using online Variational Bayes (Hoffman, Blei and Bach, 2010)
 * 
 * '''Parameters'''
 - dim(256): Model dimension
 - uiter(5): Number of iterations on one block of data
 - alpha(0.001f): Dirichlet document-topic prior
 - beta(0.0001f): Dirichlet word-topic prior
 - exppsi(true):  Apply exp(psi(X)) if true, otherwise just use X
 - LDAeps(1e-9):  A safety floor constant
 *
 * Other key parameters inherited from the learner, datasource and updater:
 - blockSize: the number of samples processed in a block
 - power(0.3f): the exponent of the moving average model' = a dmodel + (1-a)*model, a = 1/nblocks^power
 - npasses(10): number of complete passes over the dataset
 *
 * '''Example:'''
 * 
 * a is a sparse word x document matrix
 * {{{
 * val (nn, opts) = LDA.learner(a)
 * opts.what             // prints the available options
 * opts.uiter=2          // customize options
 * nn.train              // train the model
 * nn.modelmat           // get the final model
 * nn.datamat            // get the other factor (requires opts.putBack=1)
 * 
 * val (nn, opts) = LDA.learnPar(a) // Build a parallel learner
 * opts.nthreads=2       // number of threads (defaults to number of GPUs)
 * nn.train              // train the model
 * nn.modelmat           // get the final model
 * nn.datamat            // get the other factor
 * }}}
 */

class LDA(override val opts:LDA.Opts = new LDA.Options) extends FactorModel(opts) {

  var mm:Mat = null
  var alpha:Mat = null 
  var traceMem = false
  
  /** Sets up the modelmats and updatemats arrays and initializes modelmats(0) randomly unless stated otherwise. */
  override def init() = {
    super.init()
    if (refresh) {
    	mm = modelmats(0);
    	setmodelmats(Array(mm, mm.ones(mm.nrows, 1)));
    }
    updatemats = new Array[Mat](2);
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols);
    updatemats(1) = mm.zeros(mm.nrows, 1);
  }
  
  /**
   * Updates '''user''' according to the variational EM update process in the original (2003) LDA Paper.
   * 
   * This can be a bit tricky to understand. See Equation 2.2 in Huasha Zhao's PhD from UC Berkeley
   * for details on the math and cross-reference it with the 2003 LDA journal paper.
   * 
   * @param sdata The word x document input data. Has dimension (# words x opts.batchSize), where batchSize is
   *   typically much smaller than the total number of documents, so sdata is usually a portion of the full input.
   * @param user An (opts.dim x opts.batchSize) matrix that stores some intermediate/temporary data and gets left-
   *   multiplied by modelmats(0) to form sdata.
   * @param ipass Index of the pass over the data (0 = first pass, 1 = second pass, etc.).
   */
  def uupdate(sdata:Mat, user:Mat, ipass:Int, pos:Long):Unit = {
    if (putBack < 0 || ipass == 0) user.set(1f)
    for (i <- 0 until opts.uiter) {
      val preds = DDS(mm, user, sdata)	
      val dc = sdata.contents
      val pc = preds.contents
      max(opts.weps, pc, pc)
      pc ~ dc / pc
      val unew = user ∘ (mm * preds) + opts.alpha
      if (opts.exppsi) exppsi(unew, unew)
      user <-- unew   
    }	
  }
  
  /**
   * Updates '''modelmats(0)''', the topic x word matrix that is ultimately returned as output for the model.
   * 
   * @param sdata The word x document input data. Has dimension (# words x opts.batchSize), where batchSize is
   *   typically much smaller than the total number of documents, so sdata is usually a portion of the full input.
   * @param user An (opts.dim x opts.batchSize) matrix that stores some intermediate/temporary data and gets left-
   *   multiplied by modelmats(0) to form sdata.
   * @param ipass Index of the pass over the data (0 = first pass, 1 = second pass, etc.).
   */
  def mupdate(sdata:Mat, user:Mat, ipass:Int, pos:Long):Unit = {
    val preds = DDS(mm, user, sdata)
    val dc = sdata.contents
    val pc = preds.contents
    max(opts.weps, pc, pc)
    pc ~ dc / pc
    val ud = user *^ preds
    ud ~ ud ∘ mm
    ud ~ ud + opts.beta
  	updatemats(0) <-- ud  
  	sum(ud, 2, updatemats(1))
  }
  
  /** 
   * Evaluates model log-likelihood on a held-out batch of the input data.
   *  
   * @param sdata The word x document input data. Has dimension (# words x opts.batchSize), where batchSize is
   *   typically much smaller than the total number of documents, so sdata is usually a portion of the full input.
   * @param user An (opts.dim x opts.batchSize) matrix that stores some intermediate/temporary data and gets left-
   *   multiplied by modelmats(0) to form sdata.
   * @param ipass Index of the pass over the data (0 = first pass, 1 = second pass, etc.).
   */
  def evalfun(sdata:Mat, user:Mat, ipass:Int, pos:Long):FMat = {  
  	val preds = DDS(mm, user, sdata);
  	val dc = sdata.contents;
  	val pc = preds.contents;
  	max(opts.weps, pc, pc);
  	ln(pc, pc);
  	val sdat = sum(sdata,1);
  	val mms = sum(mm,2);
  	val suu = ln(mms ^* user);
  	val vv = ((pc ddot dc) - (sdat ddot suu))/sum(sdat,2).dv;
  	row(vv, math.exp(-vv))
  }
}

object LDA  {
  trait Opts extends FactorModel.Opts {
    var LDAeps = 1e-9
    var exppsi = true
    var alpha = 0.001f
    var beta = 0.0001f
  }
  
  class Options extends Opts {}
  
  /** Creates a new LDA model. */
  def mkLDAmodel(fopts:Model.Opts) = {
  	new LDA(fopts.asInstanceOf[LDA.Opts])
  }
  
  /** Creates a new IncNorm updater. */
  def mkUpdater(nopts:Updater.Opts) = {
  	new IncNorm(nopts.asInstanceOf[IncNorm.Opts])
  } 
   
  /** Online Variational Bayes LDA algorithm with a matrix datasource. */
  def learner(mat0:Mat, d:Int) = {
    class xopts extends Learner.Options with LDA.Opts with MatDS.Opts with IncNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    new LDA(opts), 
  	    null,
  	    new IncNorm(opts), 
  	    opts)
    (nn, opts)
  }
  
  class FsOpts extends Learner.Options with LDA.Opts with SFilesDS.Opts with IncNorm.Opts
  
  def learner(fpattern:String, d:Int):(Learner, FsOpts) = learner(List(FilesDS.simpleEnum(fpattern, 0, 1)), d)
  
  /** Online Variational Bayes LDA algorithm with a files dataSource. */
  def learner(fnames:List[(Int)=>String], d:Int):(Learner, FsOpts) = { 
    val opts = new FsOpts
    opts.dim = d
    opts.fnames = fnames
    opts.batchSize = 100000;
    opts.eltsPerSample = 500;
    implicit val threads = threadPool(4)
  	val nn = new Learner(
  	    new SFilesDS(opts), 
  	    new LDA(opts), 
  	    null,
  	    new IncNorm(opts), 
  	    opts)
    (nn, opts)
  }
     
  /** Batch Variational Bayes LDA algorithm with a matrix datasource. */
  def learnBatch(mat0:Mat, d:Int = 256) = {
    class xopts extends Learner.Options with LDA.Opts with MatDS.Opts with BatchNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    val nn = new Learner(
        new MatDS(Array(mat0:Mat), opts), 
        new LDA(opts), 
        null, 
        new BatchNorm(opts),
        opts)
    (nn, opts)
  }
  
  /** Parallel online LDA algorithm with a matrix datasource. */ 
  def learnPar(mat0:Mat, d:Int = 256) = {
    class xopts extends ParLearner.Options with LDA.Opts with MatDS.Opts with IncNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.batchSize = math.min(100000, mat0.ncols/30/opts.nthreads + 1)
    opts.coolit = 0 // Assume we dont need cooling on a matrix input
  	val nn = new ParLearnerF(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    opts, mkLDAmodel _, 
  	    null, null, 
  	    opts, mkUpdater _,
  	    opts)
    (nn, opts)
  }
  
  class SFDSopts extends ParLearner.Options with LDA.Opts with SFilesDS.Opts with IncNorm.Opts
  
  def learnPar(fnames:String, d:Int):(ParLearnerF, SFDSopts) = learnPar(List(FilesDS.simpleEnum(fnames, 0, 1)), d);
  
  /** Parallel online LDA algorithm with one file datasource. */
  def learnPar(fnames:List[(Int) => String], d:Int):(ParLearnerF, SFDSopts) = {
  	val opts = new SFDSopts;
  	opts.dim = d;
  	opts.npasses = 4;
    opts.fnames = fnames;
    opts.batchSize = 100000;
    opts.eltsPerSample = 500;
  	opts.resFile = "../results.mat"
  	implicit val threads = threadPool(4)
  	val nn = new ParLearnerF(
  	    new SFilesDS(opts),
  	    opts, mkLDAmodel _, 
  	    null, null, 
  	    opts, mkUpdater _,
  	    opts
  	)
  	(nn, opts)
  }
}


