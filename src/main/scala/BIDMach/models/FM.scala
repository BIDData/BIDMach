package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.CUMAT
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._


class FM(opts:FM.Opts) extends RegressionModel(opts) {
  
  var mylinks:Mat = null
  
  val linkArray = Array[GLMlink](LinearLink, LogisticLink)
  
  var totflops = 0L
  
  var mv:Mat = null
  var mm:Mat = null
  var uv:Mat = null
  var um:Mat = null
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    mylinks = if (useGPU) GIMat(opts.links) else opts.links
    mv = modelmats(0)
    if (mask.asInstanceOf[AnyRef] != null) mv ~ mv ∘ mask
    mm = mv.zeros(opts.dim, mm.ncols)
    modelmats = Array(mv, mm)
    uv = updatemats(0)
    um = uv.zeros(opts.dim, mm.ncols)
    updatemats = Array(uv, um)
    totflops = 0L
    for (i <- 0 until opts.links.length) {
      totflops += linkArray(opts.links(i)).fnflops
    }
  }
    
  def mupdate(in:Mat):FMat = {
    val targs = targets * in
    min(targs, 1f, targs)
    val alltargs = targmap * targs
    mupdate2(in, alltargs)
  }
  
  def mupdate2(in:Mat, targ:Mat):FMat = {
    val vt = mm * in
    val eta = mv * in + vt dot vt
    GLM.applymeans(eta, mylinks, eta, linkArray, totflops)
    val lls = GLM.llfun(eta, targ, mylinks, linkArray, totflops)
    eta ~ targ - eta
    uv ~ eta *^ in
    um ~ (eta * 2f) ∘ (vt *^ in)
    lls
  }

}

object FM {
  trait Opts extends RegressionModel.Opts {
    var links:IMat = null
  }
  
  class Options extends Opts {}
  
  def mkFMModel(fopts:Model.Opts) = {
  	new FM(fopts.asInstanceOf[FM.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
  	new ADAGrad(nopts.asInstanceOf[ADAGrad.Opts])
  } 
  
  class LearnOptions extends Learner.Options with FM.Opts with MatDS.Opts with ADAGrad.Opts
     
  def learn(mat0:Mat, d:Int = 0) = { 
    val opts = new LearnOptions
    opts.blockSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    new FM(opts), 
  	    null,
  	    new ADAGrad(opts), opts)
    (nn, opts)
  }
  
  def learn(mat0:Mat):(Learner, LearnOptions) = learn(mat0, 0)
  
  def learn(mat0:Mat, targ:Mat, d:Int) = {
    val opts = new LearnOptions
    opts.blockSize = math.min(100000, mat0.ncols/30 + 1)
    opts.links = izeros(targ.nrows,1)
    opts.links.set(d)
    val nn = new Learner(
        new MatDS(Array(mat0, targ), opts), 
        new FM(opts), 
        null,
        new ADAGrad(opts), opts)
    (nn, opts)
  }
  
  def learn(mat0:Mat, targ:Mat):(Learner, LearnOptions) = learn(mat0, targ, 0)
     
  def learnBatch(mat0:Mat, d:Int) = {
    val opts = new LearnOptions
    opts.blockSize = math.min(100000, mat0.ncols/30 + 1)
    val nn = new Learner(
        new MatDS(Array(mat0), opts), 
        new FM(opts), 
        null, 
        new ADAGrad(opts),
        opts)
    (nn, opts)
  }
  
  class LearnParOptions extends ParLearner.Options with FM.Opts with MatDS.Opts with ADAGrad.Opts
  
  def learnPar(mat0:Mat, d:Int) = {
    val opts = new LearnParOptions
    opts.blockSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new ParLearnerF(
  	    new MatDS(Array(mat0), opts), 
  	    opts, mkFMModel _,
  	    null, null,
  	    opts, mkUpdater _, 
  	    opts)
    (nn, opts)
  }
  
  def learnPar(mat0:Mat):(ParLearnerF, LearnParOptions) = learnPar(mat0, 0)
  
  def learnPar(mat0:Mat, targ:Mat, d:Int) = {
    val opts = new LearnParOptions
    opts.blockSize = math.min(100000, mat0.ncols/30 + 1)
    opts.links = izeros(targ.nrows,1)
    opts.links.set(d)
    val nn = new ParLearnerF(
        new MatDS(Array(mat0, targ), opts), 
        opts, mkFMModel _,
        null, null,
        opts, mkUpdater _, 
        opts)
    (nn, opts)
  }
  
  def learnPar(mat0:Mat, targ:Mat):(ParLearnerF, LearnParOptions) = learnPar(mat0, targ, 0)
  
  def learnFParx(
    nstart:Int=FilesDS.encodeDate(2012,3,1,0), 
		nend:Int=FilesDS.encodeDate(2012,12,1,0), 
		d:Int = 0
		) = {
  	class xopts extends ParLearner.Options with FM.Opts with SFilesDS.Opts with ADAGrad.Opts
  	val opts = new xopts
  	val nn = new ParLearnerxF(
  	    null,
  	    (dopts:DataSource.Opts, i:Int) => SFilesDS.twitterWords(nstart, nend, opts.nthreads, i),
  	    opts, mkFMModel _,
  	    null, null,
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
  	class xopts extends ParLearner.Options with FM.Opts with SFilesDS.Opts with IncNorm.Opts
  	val opts = new xopts
  	val nn = new ParLearnerF(
  	    SFilesDS.twitterWords(nstart, nend),
  	    opts, mkFMModel _, 
  	    null, null,
  	    opts, mkUpdater _,
  	    opts
  	)
  	(nn, opts)
  }
}

