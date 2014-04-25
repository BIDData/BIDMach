package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.CUMACH
import scala.concurrent.future
import scala.concurrent.ExecutionContext.Implicits.global
import java.util.concurrent.CountDownLatch
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach._


class GLM(opts:GLM.Opts) extends RegressionModel(opts) {
  
  var mylinks:Mat = null
  
  val linkArray = Array[GLMlink](LinearLink, LogisticLink, MaxpLink)
  
  var totflops = 0L
  
  override def init() = {
    super.init()
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
    GLM.preds(eta, eta, mylinks, linkArray, totflops)
    GLM.derivs(eta, ftarg, eta, mylinks, linkArray, totflops)
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
    GLM.preds(eta, eta, mylinks, linkArray, totflops)
    val v = GLM.llfun(eta, ftarg, mylinks, linkArray, totflops)
    if (putBack >= 0) {ftarg <-- eta}
    v
  }
}


object LinearLink extends GLMlink {
  def link(in:Float) = {
    in
  }
  
  def predlink(in:Float) = {
    in
  }
  
  def dlink(in:Float) = {
    1.0f
  }
  
  def derivlink(in:Float, targ:Float) = {
    targ - in
  }
  
  def likelihood(pred:Float, targ:Float) = {
    val diff = targ - pred
    - diff * diff
  }
     
  override val linkfn = link _
  
  override val dfn = dlink _
  
  override val derivfn = derivlink _
    
  override val predfn = predlink _
  
  override val likelihoodfn = likelihood _
  
  val fnflops = 2
}

object LogisticLink extends GLMlink {
  def link(in:Float) = {
    math.log(in / (1.0f - in)).toFloat
  }
  
  def predlink(in:Float) = {
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
  
  def derivlink(in:Float, targ:Float) = {
    targ - in
  }
  
  def likelihood(pred:Float, targ:Float) = {
    math.log(targ * pred + (1.0f - targ) * (1.0f - pred) + 1e-20).toFloat
  }
  
  override val linkfn = link _
  
  override val dfn = dlink _
  
  override val derivfn = derivlink _
  
  override val predfn = predlink _
  
  override val likelihoodfn = likelihood _
  
  val fnflops = 20
}


object MaxpLink extends GLMlink {
  def link(in:Float) = {
    math.log(in / (1.0f - in)).toFloat
  }
  
  def predlink(in:Float) = {
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
  
  def derivlink(p:Float, targ:Float) = {
    (2.0f * targ - 1.0f) * p * (1.0f - p)
  }
  
  def likelihood(pred:Float, targ:Float) = {
    targ * pred + (1.0f - targ) * (1.0f - pred) -1.0f
  }
  
  override val linkfn = link _
  
  override val dfn = dlink _
  
  override val derivfn = derivlink _
  
  override val predfn = predlink _
  
  override val likelihoodfn = likelihood _
  
  val fnflops = 20
}

object LinkEnum extends Enumeration {
  type LinkEnum = Value
  val Linear, Logistic, Maxp = Value
}

abstract class GLMlink {
  val linkfn:(Float => Float)
  val dfn:(Float => Float)
  val derivfn:((Float,Float) => Float)
  val predfn:(Float => Float)
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
        val fun = linkArray(ilinks(j)).predfn
        fout.data(j + i * fout.nrows) = fun(feta.data(j + i * feta.nrows))
        j += 1 
      }
      i += 1
    }     
  }
  
  def preds(eta:Mat, out:Mat, links:Mat, linkArray:Array[GLMlink], totflops:Long):Mat = {
    (eta, links, out) match {
      case (feta:FMat, ilinks:IMat, fout:FMat) => {
        Mat.nflops += totflops * feta.ncols
        meanHelper(feta, fout, linkArray, ilinks, 0, feta.ncols)
        out
      }
      case (geta:GMat, gilinks:GIMat, gout:GMat) => {
        Mat.nflops += totflops * geta.ncols
        CUMACH.applypreds(geta.data, gilinks.data, gout.data, geta.nrows, geta.ncols)
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
        CUMACH.applylls(gpred.data, gtarg.data, gilinks.data, out.data, gpred.nrows, gpred.ncols)
        FMat(mean(out,2))
      }
    }
  }
  
   def derivs(pred:Mat, targ:Mat, out:Mat, links:Mat, linkArray:Array[GLMlink], totflops:Long) = {
    (pred, targ, out, links) match {
      case (fpred:FMat, ftarg:FMat, fout:FMat, ilinks:IMat) => {
        Mat.nflops += 10L * ftarg.length
            var i = 0
            while (i < ftarg.ncols) {
                var j = 0
                while (j < ftarg.nrows) {
                    val fun = linkArray(ilinks(j)).derivfn
                    fout.data(j + i * out.nrows) = fun(fpred.data(j + i * ftarg.nrows),  ftarg.data(j + i * ftarg.nrows))
                    j += 1
                }
                i += 1
            }
            mean(out,2)
      }
      case (gpred:GMat, gtarg:GMat, gout:GMat, gilinks:GIMat) => {
        Mat.nflops += totflops * gpred.ncols
        CUMACH.applyderivs(gpred.data, gtarg.data, gilinks.data, gout.data, gpred.nrows, gpred.ncols)
        gout
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
     
  // Basic in-memory learner with generated target
  def learn(mat0:Mat, d:Int = 0) = { 
    val opts = new LearnOptions
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    opts.alpha = 1f
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    new GLM(opts), 
  	    mkRegularizer(opts),
  	    new ADAGrad(opts), opts)
    (nn, opts)
  }  
    
  def learn(mat0:Mat):(Learner, LearnOptions) = learn(mat0, 0)
  
  // Basic in-memory learner with explicit target
  def learn(mat0:Mat, targ:Mat, d:Int):(Learner, LearnOptions) = {
    val mopts = new LearnOptions;
    mopts.alpha = 1f
    mopts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    if (mopts.links == null) mopts.links = izeros(targ.nrows,1)
    mopts.links.set(d)
    val model = new GLM(mopts)
    val mm = new Learner(
        new MatDS(Array(mat0, targ), mopts), 
        model, 
        mkRegularizer(mopts),
        new ADAGrad(mopts), mopts)
    (mm, mopts)
  }
  
  def learn(mat0:Mat, targ:Mat):(Learner, LearnOptions) = learn(mat0, targ, 0)
  
  // This function constructs a learner and a predictor. 
  def learn(mat0:Mat, targ:Mat, mat1:Mat, preds:Mat, d:Int):(Learner, LearnOptions, Learner, LearnOptions) = {
    val mopts = new LearnOptions;
    val nopts = new LearnOptions;
    mopts.alpha = 1f
    mopts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    if (mopts.links == null) mopts.links = izeros(targ.nrows,1)
    nopts.links = mopts.links
    mopts.links.set(d)
    nopts.batchSize = mopts.batchSize
    nopts.putBack = 1
    val model = new GLM(mopts)
    val mm = new Learner(
        new MatDS(Array(mat0, targ), mopts), 
        model, 
        mkRegularizer(mopts),
        new ADAGrad(mopts), mopts)
    val nn = new Learner(
        new MatDS(Array(mat1, preds), nopts), 
        model, 
        mkRegularizer(mopts),
        new ADAGrad(mopts), mopts)
    (mm, mopts, nn, nopts)
  }
  
   // This function constructs a predictor from an existing model 
  def learn(model:Model, mat1:Mat, preds:Mat, d:Int):(Learner, LearnOptions) = {
    val nopts = new LearnOptions;
    nopts.batchSize = math.min(10000, mat1.ncols/30 + 1)
    if (nopts.links == null) nopts.links = izeros(preds.nrows,1)
    nopts.links.set(d)
    nopts.putBack = 1
    val nn = new Learner(
        new MatDS(Array(mat1, preds), nopts), 
        model.asInstanceOf[GLM], 
        null,
        null)
    (nn, nopts)
  }
     
  def learnBatch(mat0:Mat, targ:Mat, d:Int) = {
    val opts = new LearnOptions
    opts.alpha = 1f
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
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
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    opts.alpha = 1f
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
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    opts.alpha = 1f
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
  	opts.alpha = 1f
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
  	opts.alpha = 1f
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

