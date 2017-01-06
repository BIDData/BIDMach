package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,FND,IMat,HMat,GMat,GIMat,GSMat,GND,ND,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.datasinks._
import BIDMach.updaters._
import BIDMach._

/**
 * Non-negative Matrix Factorization (NMF) with L2 loss
 * 
 * '''Parameters'''
 - dim(256): Model dimension
 - uiter(5): Number of iterations on one block of data
 - uprior: Prior on the user (data) factor
 - mprior: Prior on the model
 - NMFeps(1e-9):  A safety floor constant
 *
 * Other key parameters inherited from the learner, datasource and updater:
 - batchSize: the number of samples processed in a block
 - power(0.3f): the exponent of the moving average model' = a dmodel + (1-a)*model, a = 1/nblocks^power
 - npasses(2): number of complete passes over the dataset
 *
 * '''Example:'''
 * 
 * a is a sparse word x document matrix
 * {{{
 * val (nn, opts) = NMF.learner(a)
 * opts.what             // prints the available options
 * opts.uiter=2          // customize options
 * nn.train              // train the model
 * nn.modelmat           // get the final model
 * nn.datamat            // get the other factor (requires opts.putBack=1)
 * 
 * val (nn, opts) = NMF.learnPar(a) // Build a parallel learner
 * opts.nthreads=2       // number of threads (defaults to number of GPUs)
 * nn.train              // run the model
 * nn.modelmat           // get the final model
 * nn.datamat            // get the other factor
 * }}}
 */
class NMF(opts:NMF.Opts = new NMF.Options) extends FactorModel(opts) {
  
  var mm:Mat = null
  var mdiag:Mat = null
  var udiag:Mat = null
  
  override def init() = {
  	super.init()
  	mm = modelmats(0).asMat
    setmodelmats(Array(mm, mm.zeros(mm.nrows, mm.ncols)));
  	updatemats = new Array[ND](2)
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols)
    updatemats(1) = mm.zeros(mm.nrows, mm.ncols)
    udiag = mkdiag(opts.uprior*ones(opts.dim,1))
  	mdiag = mkdiag(opts.mprior*ones(opts.dim,1))
    if (useGPU) {
      udiag = GMat(udiag)
      mdiag = GMat(mdiag)
    }
  }
  
  override def uupdate(sdata:Mat, user:Mat, ipass:Int, pos:Long) = {
	if (ipass == 0) user.set(1f)
	val modeldata = mm * sdata
  	val mmu = mm *^ mm + udiag
    for (i <- 0 until opts.uiter) {
    	val quot =  modeldata / (mmu * user)               
    	min(10.0f, max(0.1f, quot, quot), quot)
    	user ~ user ∘ quot
    	max(opts.minuser, user, user)
    }
  }  
  
  override def mupdate(sdata:Mat, user:Mat, ipass:Int, pos:Long):Unit = {
    val uu = user *^ user + mdiag *@ (1.0f*size(user,2)/opts.nusers) 
    updatemats(0) ~ (user *^ sdata) *@ mm
    updatemats(1) ~ uu * mm
    max(updatemats(1), opts.NMFeps, updatemats(1))
  }

  override def mupdate2(sdata:Mat, user:Mat, ipass:Int):Unit = {
    val uu = user *^ user + mdiag *@ (1.0f*size(user,2)/opts.nusers)
    updatemats(0) ~ user *^ sdata
    updatemats(1) ~ uu * mm
  }
  
  override def evalfun(sdata:Mat, user:Mat, ipass:Int, pos:Long):FMat = {
    if (ogmats != null) ogmats(0) = user;
    if (opts.doubleScore) {
      evalfunx(sdata, user)
    } else {
    	val modeldata =  mm * sdata
    	val uu = user *^ user + mdiag *@ (1.0f*size(user,2)/opts.nusers)
    	val mmm = mm *^ mm

    	val ll0 =  sdata.contents ddot sdata.contents
    	val ll1 =  modeldata ddot user
    	val ll2 =  uu ddot mmm
    	val v1  =              (-ll0 + 2*ll1 - ll2)/sdata.nnz
    	val v2 =               -opts.uprior*(user ddot user)/sdata.nnz
    	row(v1,v2)
    }
  }
  
  def evalfunx(sdata0:Mat, user0:Mat):FMat = { 
    val sdata = SDMat(sdata0)
    val user = DMat(user0)
    val mmf = DMat(mm)
    val mdiagf = DMat(mdiag)

	  val modeldata =  mmf * sdata
    val uu = user *^ user + mdiagf *@ (1.0f*size(user,2)/opts.nusers)
    val mmm = mmf *^ mmf

    val ll0 =  sdata.contents ddot sdata.contents
    val ll1 =  modeldata ddot user
    val ll2 =  uu ddot mmm
    val v1  =              (-ll0 + 2*ll1 - ll2)/sdata.nnz
    val v2 =               -opts.uprior*(user ddot user)/sdata.nnz
    row(v1,v2)
  }
}

object NMF  {
  trait Opts extends FactorModel.Opts {
    var NMFeps = 1e-12
    var uprior = 0.01f
    var mprior = 1e-4f
    var nusers = 100000
  }
  
  class Options extends Opts {}
  
  def mkNMFmodel(fopts:Model.Opts) = {
  	new NMF(fopts.asInstanceOf[NMF.Opts])
  } 
   
  def mkUpdater(nopts:Updater.Opts) = {
  	new IncNorm(nopts.asInstanceOf[IncNorm.Opts])
  }
    
  def learner(mat0:Mat, d:Int = 256) = {
    class xopts extends Learner.Options with NMF.Opts with MatSource.Opts with IncNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.uiter = 2
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatSource(Array(mat0:Mat), opts), 
  			new NMF(opts), 
  			null,
  			new IncNorm(opts), 
  			null,
  			opts)
    (nn, opts)
  }
  
  class PredOptions extends Learner.Options with NMF.Opts with MatSource.Opts with MatSink.Opts;
  
    // This function constructs a predictor from an existing model 
  def predictor(model:Model, mat1:Mat):(Learner, PredOptions) = {
    val nopts = new PredOptions;
    nopts.batchSize = math.min(10000, mat1.ncols/30 + 1)
    nopts.dim = model.opts.dim;
    val newmod = new NMF(nopts);
    newmod.refresh = false
    model.copyTo(newmod)
    val nn = new Learner(
        new MatSource(Array(mat1), nopts), 
        newmod, 
        null,
        null,
        new MatSink(nopts),
        nopts)
    (nn, nopts)
  }
     
  def learnBatch(mat0:Mat, d:Int = 256) = {
    class xopts extends Learner.Options with NMF.Opts with MatSource.Opts with BatchNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.uiter = 1
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    val nn = new Learner(
        new MatSource(Array(mat0:Mat), opts), 
        new NMF(opts), 
        null, 
        new BatchNorm(opts),
        null,
        opts)
    (nn, opts)
  }
  
  def learnPar(mat0:Mat, d:Int = 256) = {
    class xopts extends ParLearner.Options with NMF.Opts with MatSource.Opts with IncNorm.Opts
    val opts = new xopts
    opts.dim = d
    opts.npasses = 4
    opts.batchSize = math.min(100000, mat0.ncols/30/opts.nthreads + 1)
    opts.coolit = 0 // Assume we dont need cooling on a matrix input
  	val nn = new ParLearnerF(
  	    new MatSource(Array(mat0:Mat), opts), 
  	    opts, mkNMFmodel _, 
  			null, null, 
  			opts, mkUpdater _,
  			null, null,
  			opts)
    (nn, opts)
  }
 
} 



