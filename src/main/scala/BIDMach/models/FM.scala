package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.CUMAT
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach._


class FM(opts:FM.Opts) extends RegressionModel(opts) {
  
  var mylinks:Mat = null
  
  val linkArray = Array[GLMlink](LinearLink, LogisticLink)
  
  var totflops = 0L
  
  var mv:Mat = null
  var mm1:Mat = null
  var mm2:Mat = null
  var uv:Mat = null
  var um1:Mat = null
  var um2:Mat = null
  
  override def init() = {
    super.init()
    mylinks = if (useGPU) GIMat(opts.links) else opts.links
    mv = modelmats(0)
    if (mask.asInstanceOf[AnyRef] != null) mv ~ mv ∘ mask
    val rmat1 = rand(opts.dim1, mv.ncols) - 0.5f 
    rmat1 ~ rmat1 *@ (sp * (1f/math.sqrt(opts.dim1).toFloat))
    mm1 = if (useGPU) GMat(rmat1) else rmat1
    val rmat2 = rand(opts.dim2, mv.ncols) - 0.5f 
    rmat2 ~ rmat2 *@ (sp * (1f/math.sqrt(opts.dim2).toFloat))
    mm2 = if (useGPU) GMat(rmat2) else rmat2
    modelmats = Array(mv, mm1, mm2)
    uv = updatemats(0)
    um1 = uv.zeros(opts.dim1, uv.ncols)
    um2 = uv.zeros(opts.dim2, uv.ncols)
    updatemats = Array(uv, um1, um2)
    totflops = 0L
    for (i <- 0 until opts.links.length) {
      totflops += linkArray(opts.links(i)).fnflops
    }
  }
    
  def mupdate(in:Mat) = {
    val targs = targets * in
    min(targs, 1f, targs)
    val alltargs = targmap * targs
    mupdate2(in, alltargs)
  }
  
  def mupdate2(in:Mat, targ:Mat) = {
    val vt1 = mm1 * in
    val vt2 = mm2 * in
    val eta = mv * in + (vt1 dot vt1) - (vt2 dot vt2)
    GLM.preds(eta, mylinks, eta, linkArray, totflops)
    eta ~ targ - eta
    uv ~ eta *^ in
    um1 ~ (vt1 ∘ (eta * 2f)) *^ in
    um2 ~ (vt2 ∘ (eta * -2f)) *^ in
  }
  
  def meval(in:Mat):FMat = {
    val targs = targets * in
    min(targs, 1f, targs)
    val alltargs = targmap * targs
    meval2(in, alltargs)
  }
  
  def meval2(in:Mat, targ:Mat):FMat = {
    val vt1 = mm1 * in
    val vt2 = mm2 * in
    val eta = mv * in + (vt1 dot vt1) - (vt2 dot vt2)
    GLM.preds(eta, mylinks, eta, linkArray, totflops)
    GLM.llfun(eta, targ, mylinks, linkArray, totflops)
  }

}

object FM {
  trait Opts extends RegressionModel.Opts {
    var links:IMat = null
    var dim1 = 128
    var dim2 = 128
  }
  
  class Options extends Opts {}
  
  def mkFMModel(fopts:Model.Opts) = {
  	new FM(fopts.asInstanceOf[FM.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
  	new ADAGrad(nopts.asInstanceOf[ADAGrad.Opts])
  }
  
  def mkRegularizer(nopts:Mixin.Opts):Array[Mixin] = {
    Array(new L1Regularizer(nopts.asInstanceOf[Regularizer.Opts]))
  } 
  
  class LearnOptions extends Learner.Options with FM.Opts with MatDS.Opts with ADAGrad.Opts with Regularizer.Opts
     
  def learn(mat0:Mat, d:Int = 0) = { 
    val opts = new LearnOptions
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    new FM(opts), 
  	    mkRegularizer(opts),
  	    new ADAGrad(opts), opts)
    (nn, opts)
  }
  
  def learn(mat0:Mat):(Learner, LearnOptions) = learn(mat0, 0)
  
  def learn(mat0:Mat, targ:Mat, d:Int) = {
    val opts = new LearnOptions
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    if (opts.links == null) opts.links = izeros(targ.nrows,1)
    opts.links.set(d)
    val nn = new Learner(
        new MatDS(Array(mat0, targ), opts), 
        new FM(opts), 
        mkRegularizer(opts),
        new ADAGrad(opts), opts)
    (nn, opts)
  }
  
  def learn(mat0:Mat, targ:Mat):(Learner, LearnOptions) = learn(mat0, targ, 0)
     
  def learnBatch(mat0:Mat, d:Int) = {
    val opts = new LearnOptions
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    opts.links.set(d)
    val nn = new Learner(
        new MatDS(Array(mat0), opts), 
        new FM(opts), 
        mkRegularizer(opts),
        new ADAGrad(opts),
        opts)
    (nn, opts)
  }
  
  class LearnParOptions extends ParLearner.Options with FM.Opts with MatDS.Opts with ADAGrad.Opts with Regularizer.Opts
  
  def learnPar(mat0:Mat, d:Int) = {
    val opts = new LearnParOptions
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    opts.links.set(d)
  	val nn = new ParLearnerF(
  	    new MatDS(Array(mat0), opts), 
  	    opts, mkFMModel _,
  	    opts, mkRegularizer _,
  	    opts, mkUpdater _, 
  	    opts)
    (nn, opts)
  }
  
  def learnPar(mat0:Mat):(ParLearnerF, LearnParOptions) = learnPar(mat0, 0)
  
  def learnPar(mat0:Mat, targ:Mat, d:Int) = {
    val opts = new LearnParOptions
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    if (opts.links == null) opts.links = izeros(targ.nrows,1)
    opts.links.set(d)
    val nn = new ParLearnerF(
        new MatDS(Array(mat0, targ), opts), 
        opts, mkFMModel _,
        opts, mkRegularizer _,
        opts, mkUpdater _, 
        opts)
    (nn, opts)
  }
  
  def learnPar(mat0:Mat, targ:Mat):(ParLearnerF, LearnParOptions) = learnPar(mat0, targ, 0)
  
  class LearnFParOptions extends ParLearner.Options with FM.Opts with SFilesDS.Opts with ADAGrad.Opts with Regularizer.Opts
  
  def learnFParx(
    nstart:Int=FilesDS.encodeDate(2012,3,1,0), 
		nend:Int=FilesDS.encodeDate(2012,12,1,0), 
		d:Int = 0
		) = {
  	val opts = new LearnFParOptions
  	val nn = new ParLearnerxF(
  	    null,
  	    (dopts:DataSource.Opts, i:Int) => SFilesDS.twitterWords(nstart, nend, opts.nthreads, i),
  	    opts, mkFMModel _,
        opts, mkRegularizer _,
  	    opts, mkUpdater _,
  	    opts
  	)
  	(nn, opts)
  }
  
  def learnFPar(
    nstart:Int=FilesDS.encodeDate(2012,3,1,0), 
		nend:Int=FilesDS.encodeDate(2012,12,1,0), 
		d:Int = 0
		) = {	
  	val opts = new LearnFParOptions
  	val nn = new ParLearnerF(
  	    SFilesDS.twitterWords(nstart, nend),
  	    opts, mkFMModel _, 
        opts, mkRegularizer _,
  	    opts, mkUpdater _,
  	    opts
  	)
  	(nn, opts)
  }
}

