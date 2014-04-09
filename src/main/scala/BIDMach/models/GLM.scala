package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.CUMAT
import scala.concurrent.future
import scala.concurrent.ExecutionContext.Implicits.global
import java.util.concurrent.CountDownLatch
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach._


class GLM(opts:GLM.Opts) extends RegressionModel(opts) {
  
  var mylinks:Mat = null
  
  val linkArray = Array[GLMlink](LinearLink, LogisticLink)
  
  var totflops = 0L
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    mylinks = if (useGPU) GIMat(opts.links) else opts.links
    if (mask.asInstanceOf[AnyRef] != null) modelmats(0) ~ modelmats(0) ∘ mask
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
    val ftarg = full(targ)
    val eta = modelmats(0) * in
    GLM.applymeans(eta, mylinks, eta, linkArray, totflops)
    eta ~ ftarg - eta
    updatemats(0) ~ eta *^ in
  }
  
  def meval(in:Mat):FMat = {
    val targs = targets * in
    min(targs, 1f, targs)
    val alltargs = targmap * targs
    meval2(in, alltargs)
  }
  
  def meval2(in:Mat, targ:Mat):FMat = {
    val ftarg = full(targ)
    val eta = modelmats(0) * in
    GLM.applymeans(eta, mylinks, eta, linkArray, totflops)
    GLM.llfun(eta, ftarg, mylinks, linkArray, totflops)
  }
}


object LinearLink extends GLMlink {
  def link(in:Float) = {
    in
  }
  
  def invlink(in:Float) = {
    in
  }
  
  def dlink(in:Float) = {
    1.0f
  }
  
  def likelihood(pred:Float, targ:Float) = {
    val diff = targ - pred
    - diff * diff
  }
     
  override val linkfn = link _
  
  override val dlinkfn = dlink _
  
  override val invlinkfn = invlink _
  
  override val likelihoodfn = likelihood _
  
  val fnflops = 2
}

object LogisticLink extends GLMlink {
  def link(in:Float) = {
    math.log(in / (1.0f - in)).toFloat
  }
  
  def invlink(in:Float) = {
    if (in > 0) {
    	val tmp = math.exp(-in)
    	(1.0 / (1.0 + tmp)).toFloat    
    } else {
    	val tmp = math.exp(in)
    	(tmp / (1.0 + tmp)).toFloat
    }
  }
  
  def dlink(in:Float) = {
    1 / (in * (1 - in))
  }
  
  def likelihood(pred:Float, targ:Float) = {
    math.log(targ * pred + (1.0f - targ) * (1.0f - pred) + 1e-20).toFloat
  }
  
  override val linkfn = link _
  
  override val dlinkfn = dlink _
  
  override val invlinkfn = invlink _
  
  override val likelihoodfn = likelihood _
  
  val fnflops = 20
}

object LinkEnum extends Enumeration {
  type LinkEnum = Value
  val Linear, Logistic = Value
}

abstract class GLMlink {
  val linkfn:(Float => Float)
  val dlinkfn:(Float => Float)
  val invlinkfn:(Float => Float)
  val likelihoodfn:((Float,Float) => Float)
  val fnflops:Int
}

object GLM {
  trait Opts extends RegressionModel.Opts {
    var links:IMat = null
  }
  
  class Options extends Opts {}
  
  def meanHelper(feta:FMat, fout:FMat, linkArray:Array[GLMlink], ilinks:IMat, istart:Int, iend:Int) {
    var i = istart
    while (i < iend) {
      var j = 0
      while (j < feta.nrows) { 
        val fun = linkArray(ilinks(j)).invlinkfn
        fout.data(j + i * fout.nrows) = fun(feta.data(j + i * feta.nrows))
        j += 1 
      }
      i += 1
    }     
  }
  
  def applymeans(eta:Mat, links:Mat, out:Mat, linkArray:Array[GLMlink], totflops:Long):Mat = {
    (eta, links, out) match {
      case (feta:FMat, ilinks:IMat, fout:FMat) => {
        Mat.nflops += totflops * feta.ncols
        meanHelper(feta, fout, linkArray, ilinks, 0, feta.ncols)
        out
      }
      case (geta:GMat, gilinks:GIMat, gout:GMat) => {
        Mat.nflops += totflops * geta.ncols
        CUMAT.applymeans(geta.data, gilinks.data, gout.data, geta.nrows, geta.ncols)
        out
      }
    }
  }
  
  def llfun(pred:Mat, targ:Mat, links:Mat, linkArray:Array[GLMlink], totflops:Long):FMat = {
    (pred, targ, links) match {
      case (fpred:FMat, ftarg:FMat, ilinks:IMat) => {
        Mat.nflops += 10L * ftarg.length
            var i = 0
            val out = (ftarg + 5f)
            while (i < ftarg.ncols) {
                var j = 0
                while (j < ftarg.nrows) {
                    val fun = linkArray(ilinks(j)).likelihoodfn
                    out.data(j + i * out.nrows) = fun(fpred.data(j + i * ftarg.nrows),  ftarg.data(j + i * ftarg.nrows))
                    j += 1
                }
                i += 1
            }
            mean(out,2)
      }
      case (gpred:GMat, gtarg:GMat, gilinks:GIMat) => {
        Mat.nflops += totflops * gpred.ncols
        val out = (gpred + 3f)
        CUMAT.applylls(gpred.data, gtarg.data, gilinks.data, out.data, gpred.nrows, gpred.ncols)
        FMat(mean(out,2))
      }
    }
  }
  
  def mkGLMModel(fopts:Model.Opts) = {
  	new GLM(fopts.asInstanceOf[GLM.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
  	new ADAGrad(nopts.asInstanceOf[ADAGrad.Opts])
  } 
  
  def mkRegularizer(nopts:Mixin.Opts):Array[Mixin] = {
    Array(new L1Regularizer(nopts.asInstanceOf[Regularizer.Opts]))
  }
  
  class LearnOptions extends Learner.Options with GLM.Opts with MatDS.Opts with ADAGrad.Opts with Regularizer.Opts
     
  def learn(mat0:Mat, d:Int = 0) = { 
    val opts = new LearnOptions
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    opts.alpha = 0.1f
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    new GLM(opts), 
  	    mkRegularizer(opts),
  	    new ADAGrad(opts), opts)
    (nn, opts)
  }
  
  def learn(mat0:Mat):(Learner, LearnOptions) = learn(mat0, 0)
  
  def learn(mat0:Mat, targ:Mat, d:Int) = {
    val opts = new LearnOptions
    opts.alpha = 0.1f
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    if (opts.links == null) opts.links = izeros(targ.nrows,1)
    opts.links.set(d)
    val nn = new Learner(
        new MatDS(Array(mat0, targ), opts), 
        new GLM(opts), 
        mkRegularizer(opts),
        new ADAGrad(opts), opts)
    (nn, opts)
  }
  
  def learn(mat0:Mat, targ:Mat):(Learner, LearnOptions) = learn(mat0, targ, 0)
     
  def learnBatch(mat0:Mat, targ:Mat, d:Int) = {
    val opts = new LearnOptions
    opts.alpha = 0.1f
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    if (opts.links == null) opts.links = izeros(targ.nrows,1)
    val nn = new Learner(
        new MatDS(Array(mat0, targ), opts), 
        new GLM(opts), 
        mkRegularizer(opts), 
        new ADAGrad(opts),
        opts)
    (nn, opts)
  }
  
  class LearnParOptions extends ParLearner.Options with GLM.Opts with MatDS.Opts with ADAGrad.Opts with Regularizer.Opts
  
  def learnPar(mat0:Mat, d:Int) = {
    val opts = new LearnParOptions
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    opts.alpha = 0.1f
  	val nn = new ParLearnerF(
  	    new MatDS(Array(mat0), opts), 
  	    opts, mkGLMModel _,
  	    opts, mkRegularizer _,
  	    opts, mkUpdater _, 
  	    opts)
    (nn, opts)
  }
  
  def learnPar(mat0:Mat):(ParLearnerF, LearnParOptions) = learnPar(mat0, 0)
  
  def learnPar(mat0:Mat, targ:Mat, d:Int) = {
    val opts = new LearnParOptions
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    opts.alpha = 0.1f
    if (opts.links == null) opts.links = izeros(targ.nrows,1)
    opts.links.set(d)
    val nn = new ParLearnerF(
        new MatDS(Array(mat0, targ), opts), 
        opts, mkGLMModel _,
        opts, mkRegularizer _,
        opts, mkUpdater _, 
        opts)
    (nn, opts)
  }
  
  def learnPar(mat0:Mat, targ:Mat):(ParLearnerF, LearnParOptions) = learnPar(mat0, targ, 0)
  
  class LearnFParOptions extends ParLearner.Options with GLM.Opts with SFilesDS.Opts with ADAGrad.Opts with Regularizer.Opts
  
  def learnFParx(
    nstart:Int=FilesDS.encodeDate(2012,3,1,0), 
		nend:Int=FilesDS.encodeDate(2012,12,1,0), 
		d:Int = 0
		) = {
  	
  	val opts = new LearnFParOptions
  	opts.alpha = 0.1f
  	val nn = new ParLearnerxF(
  	    null,
  	    (dopts:DataSource.Opts, i:Int) => SFilesDS.twitterWords(nstart, nend, opts.nthreads, i),
  	    opts, mkGLMModel _,
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
  	opts.alpha = 0.1f
  	val nn = new ParLearnerF(
  	    SFilesDS.twitterWords(nstart, nend),
  	    opts, mkGLMModel _, 
        opts, mkRegularizer _,
  	    opts, mkUpdater _,
  	    opts
  	)
  	(nn, opts)
  }
}

