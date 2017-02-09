package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.datasinks._
import BIDMach.updaters._
import BIDMach._

/**
 * Click model using online Variational Bayes (Hoffman, Blei and Bach, 2010)
 * This is the same as the online VB LDA model in this directory, except there are two input matrices, clicks and views. 
 * Click count is the target value. The view count is used to scale the LDA prediction for that feature. 
 * 
 * '''Parameters'''
 - dim(256): Model dimension
 - uiter(5): Number of iterations on one block of data
 - alpha(0.001f): Dirichlet document-topic prior
 - beta(0.0001f): Dirichlet word-topic prior
 - exppsi(true):  Apply exp(psi(X)) if true, otherwise just use X
 - Clickeps(1e-9):  A safety floor constant
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
 * val (nn, opts) = Click.learner(a)
 * opts.what             // prints the available options
 * opts.uiter=2          // customize options
 * nn.train              // train the model
 * nn.modelmat           // get the final model
 * nn.datamat            // get the other factor (requires opts.putBack=1)
 * 
 * val (nn, opts) = Click.learnPar(a) // Build a parallel learner
 * opts.nthreads=2       // number of threads (defaults to number of GPUs)
 * nn.train              // train the model
 * nn.modelmat           // get the final model
 * nn.datamat            // get the other factor
 * }}}
 */

class Click(override val opts:Click.Opts = new Click.Options) extends FactorModel(opts) {

  var mm:Mat = null
  var traceMem = false
  
  /** Sets up the modelmats and updatemats arrays and initializes modelmats(0) randomly unless stated otherwise. */
  override def init() = {
    super.init();
    mm = modelmats(0);
    if (refresh) {
    	setmodelmats(Array(mm, mm.ones(mm.nrows, 1)));
    }
    updatemats = new Array[Mat](2);
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols);     // The actual model matrix
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
  def uupdate(views:Mat, clicks:Mat, user:Mat, ipass:Int, pos:Long):Unit = {
    if (putBack < 0 || ipass == 0) user.set(1f);
    for (i <- 0 until opts.uiter) {
      val preds = DDS(mm, user, views);	
//      if (ipass == 0 && pos <= 10000) println("preds "+preds.contents(0->20))
      val dc = clicks.contents - opts.clickOffset;                 // Subtract one assuming click counts incremented to avoid sparse matrix misalignment
      val dv = views.contents;
      val pc = preds.contents;
      pc ~ pc ∘ dv;                                                // scale the click prediction by the number of views
      max(opts.weps, pc, pc)
      pc ~ dc / pc
      val unew = user ∘ (mm * preds) + opts.alpha
      if (opts.exppsi) exppsi(unew, unew)
      user <-- unew   
//      if (ipass == 0 && pos <= 10000) println("user "+ user(0->20))
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
  def mupdate(views:Mat, clicks:Mat, user:Mat, ipass:Int, pos:Long):Unit = {
    val preds = DDS(mm, user, views);
    val dc = clicks.contents -opts.clickOffset;
    val dv = views.contents;
    val pc = preds.contents;
    pc ~ pc ∘ dv; 
    max(opts.weps, pc, pc);
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
  override def evalfun(views:Mat, clicks:Mat, user:Mat, ipass:Int, pos:Long):FMat = {  
  	if (ogmats != null) ogmats(0) = user;
  	val preds = DDS(mm, user, views);
    val dc = clicks.contents - opts.clickOffset;
    val dv = views.contents;    
  	val pc = preds.contents;
    pc ~ pc ∘ dv;
  	max(opts.weps, pc, pc);
  	val spc = sum(pc);
  	ln(pc, pc);
    val vv = ((dc ∙ pc) - sum(gammaln(dc + 1)) - spc).dv / dc.length;
  	row(vv)
  }
  
  override def dobatch(gmats:Array[Mat], ipass:Int, i:Long) = {
    val views = gmats(0);
    val clicks = gmats(1);
    val user = if (gmats.length > 2) gmats(2) else FactorModel.reuseuser(gmats(0), opts.dim, opts.initUval)
    uupdate(views, clicks, user, ipass, i)
    mupdate(views, clicks, user, ipass, i)
  }
  
  override def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat = {
    val views = gmats(0);
    val clicks = gmats(1);
    val user = if (gmats.length > 2) gmats(2) else FactorModel.reuseuser(gmats(0), opts.dim, opts.initUval);
    uupdate(views, clicks, user, ipass, here);
    evalfun(views, clicks, user, ipass, here);
  }
  
  def uupdate(data:Mat, user:Mat, ipass:Int, pos:Long) = {} 
  
  def mupdate(data:Mat, user:Mat, ipass:Int, pos:Long) = {}
  
  def evalfun(data:Mat, user:Mat, ipass:Int, pos:Long):FMat = {zeros(1,1)}
}

object Click  {
  trait Opts extends FactorModel.Opts {
    var LDAeps = 1e-9
    var exppsi = false
    var alpha = 0.001f
    var beta = 0.0001f
    var clickOffset = 1f
  }
  
  class Options extends Opts {}
  
  /** Creates a new Click model. */
  def mkClickmodel(fopts:Model.Opts) = {
  	new Click(fopts.asInstanceOf[Click.Opts])
  }
  
  /** Creates a new IncNorm updater. */
  def mkUpdater(nopts:Updater.Opts) = {
  	new IncNorm(nopts.asInstanceOf[IncNorm.Opts])
  } 
   
  /** Online Variational Bayes Click algorithm with a two matrix datasource. */
  def learner(mat0:Mat, mat1:Mat) = {
    class xopts extends Learner.Options with Click.Opts with MatSource.Opts with IncNorm.Opts
    val opts = new xopts
    opts.dim = 1
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatSource(Array(mat0, mat1), opts), 
  	    new Click(opts), 
  	    null,
  	    new IncNorm(opts), 
  	    null,
  	    opts)
    (nn, opts)
  }
  
  class FsOpts extends Learner.Options with Click.Opts with SFileSource.Opts with IncNorm.Opts
  
  def learner(fpattern:String, d:Int):(Learner, FsOpts) = learner(List(FileSource.simpleEnum(fpattern, 1, 0)), d)
  
  /** Online Variational Bayes Click algorithm with a files dataSource. */
  def learner(fnames:List[(Int)=>String], d:Int):(Learner, FsOpts) = { 
    val opts = new FsOpts
    opts.dim = d
    opts.fnames = fnames
    opts.batchSize = 100000;
    opts.eltsPerSample = 500;
    implicit val threads = threadPool(4)
  	val nn = new Learner(
  	    new SFileSource(opts), 
  	    new Click(opts), 
  	    null,
  	    new IncNorm(opts),
  	    null,
  	    opts)
    (nn, opts)
  }
     
  /** Batch Variational Bayes Click algorithm with a matrix datasource. */
  def learnBatch(mat0:Mat, mat1:Mat) = {
    class xopts extends Learner.Options with Click.Opts with MatSource.Opts with BatchNorm.Opts
    val opts = new xopts
    opts.dim = 1
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    val nn = new Learner(
        new MatSource(Array(mat0, mat1), opts), 
        new Click(opts), 
        null, 
        new BatchNorm(opts),
        null,
        opts)
    (nn, opts)
  }
  
  class PredOptions extends Learner.Options with Click.Opts with MatSource.Opts with MatSink.Opts;
  
    // This function constructs a predictor from an existing model 
  def predictor(model:Model, mat0:Mat, mat1:Mat):(Learner, PredOptions) = {
    val nopts = new PredOptions;
    nopts.batchSize = math.min(10000, mat1.ncols/30 + 1)
    nopts.dim = model.opts.dim;
    val newmod = new Click(nopts);
    newmod.refresh = false
    model.copyTo(newmod)
    val nn = new Learner(
        new MatSource(Array(mat0, mat1), nopts), 
        newmod, 
        null,
        null,
        new MatSink(nopts),
        nopts)
    (nn, nopts)
  }
  
  /** Parallel online Click algorithm with a matrix datasource. */ 
  def learnPar(mat0:Mat, mat1:Mat) = {
    class xopts extends ParLearner.Options with Click.Opts with MatSource.Opts with IncNorm.Opts;
    val opts = new xopts;
    opts.dim = 1;
    opts.batchSize = math.min(100000, mat0.ncols/30/opts.nthreads + 1);
    opts.coolit = 0 // Assume we dont need cooling on a matrix input
  	val nn = new ParLearnerF(
  	    new MatSource(Array(mat0:Mat), opts), 
  	    opts, mkClickmodel _, 
  	    null, null, 
  	    opts, mkUpdater _,
  	    null, null,
  	    opts)
    (nn, opts)
  }
  
  class SFDSopts extends ParLearner.Options with Click.Opts with SFileSource.Opts with IncNorm.Opts
  
  def learnPar(fnames:String, d:Int):(ParLearnerF, SFDSopts) = learnPar(List(FileSource.simpleEnum(fnames, 1, 0)), d);
  
  /** Parallel online Click algorithm with one file datasource. */
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
  	    new SFileSource(opts),
  	    opts, mkClickmodel _, 
  	    null, null, 
  	    opts, mkUpdater _,
  	    null, null,
  	    opts
  	)
  	(nn, opts)
  }
}


