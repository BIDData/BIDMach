package BIDMach.causal

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
import BIDMach.models._
import BIDMach._


class IPTW(opts:IPTW.Opts) extends RegressionModel(opts) {
  
  var mylinks:Mat = null
  
  var otargets:Mat = null
  
  var totflops = 0L
  
  var ustep = 0
  
  override def init() = {
    super.init()
    mylinks = if (useGPU) GIMat(opts.links) else opts.links
    if (mask.asInstanceOf[AnyRef] != null) modelmats(0) ~ modelmats(0) ∘ mask
    totflops = 0L
    for (i <- 0 until opts.links.length) {
      totflops += GLM.linkArray(opts.links(i)).fnflops
    }
    otargets = targets.rowslice(targets.nrows/2, targets.nrows);
    val tmats = new Array[Mat](3)
    tmats(0) = modelmats(0)
    tmats(1) = modelmats(0).zeros(targets.nrows/2,1)
    tmats(2) = modelmats(0).zeros(targets.nrows/2,1)
    setmodelmats(tmats)
    val umats = new Array[Mat](3)
    umats(0) = updatemats(0)
    umats(1) = updatemats(0).zeros(targets.nrows/2,1)
    umats(2) = updatemats(0).zeros(targets.nrows/2,1)
    updatemats = umats
    ustep = 0
  }
    
  def mupdate(in:Mat, ipass:Int, pos:Long) = {
    val targs = targets * in
    mupdate2(in, targs, ipass, pos)
  }
  
  def mupdate2(in:Mat, targ:Mat, ipass:Int, pos:Long) = {
    val ftarg = full(targ)
    val treatment = ftarg.rowslice(0, ftarg.nrows/2);
    val outcome = ftarg.rowslice(ftarg.nrows/2, ftarg.nrows)
    val eta = modelmats(0) * in
    val feta = eta + 0f
    GLM.preds(eta, feta, mylinks, totflops)
   
    val propensity = feta.rowslice(0, feta.nrows/2)                         // Propensity score
    val iptw = (treatment ∘ outcome) / propensity - ((1 - treatment) ∘ outcome) / (1 - propensity)
    
    val tmodel = otargets ∘ modelmats(0).rowslice(targ.nrows/2, targ.nrows)
    val vx0 = eta.rowslice(eta.nrows/2, eta.nrows) - tmodel * in            // compute vx given T = 0
    val vx1 = vx0 + sum(tmodel, 2)                                          // compute vx given T = 1
    GLM.preds(vx0, vx0, mylinks, totflops)
    GLM.preds(vx1, vx1, mylinks, totflops)

    val tdiff = treatment - propensity
    val aiptw = iptw - (tdiff ∘ (vx0 / propensity + vx1 / (1 - propensity)))
//    println("%d effect %f" format (ustep, mean(iptw,2).dv))
    if (ustep > opts.cwait) {
      updatemats(1) ~ mean(iptw, 2) - modelmats(1)
      updatemats(2) ~ mean(aiptw, 2) - modelmats(2)
    }
    ustep += 1
    
    GLM.derivs(feta, ftarg, feta, mylinks, totflops)
    updatemats(0) ~ feta *^ in                                              // update the primary predictors
     if (mask.asInstanceOf[AnyRef] != null) {
      updatemats(0) ~ updatemats(0) ∘ mask
    }
  }
  
  def meval(in:Mat):FMat = {
    val targs = targets * in
    meval2(in, targs)
  }
  
  def meval2(in:Mat, targ:Mat):FMat = {
    val ftarg = full(targ)
    val eta = modelmats(0) * in
    GLM.preds(eta, eta, mylinks, totflops)
    val v = GLM.llfun(eta, ftarg, mylinks, totflops)
    if (putBack >= 0) {ftarg <-- eta}
    FMat(mean(v, 2))
  }
}


object IPTW {
  trait Opts extends RegressionModel.Opts {
    var links:IMat = null
    var cwait = 20
  }
  
  class Options extends Opts {}
  
  def mkModel(fopts:Model.Opts) = {
  	new IPTW(fopts.asInstanceOf[IPTW.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
  	new ADAGrad(nopts.asInstanceOf[ADAGrad.Opts])
  } 
  
  def mkRegularizer(nopts:Mixin.Opts):Array[Mixin] = {
    Array(new L1Regularizer(nopts.asInstanceOf[L1Regularizer.Opts]))
  }
  
  def mkL2Regularizer(nopts:Mixin.Opts):Array[Mixin] = {
    Array(new L2Regularizer(nopts.asInstanceOf[L2Regularizer.Opts]))
  }
  
  class LearnOptions extends Learner.Options with IPTW.Opts with MatDS.Opts with ADAGrad.Opts with L1Regularizer.Opts
     
  // Basic in-memory learner with generated target
  def learner(mat0:Mat) = { 
    val opts = new LearnOptions
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    opts.lrate = 1f
    opts.links = 1
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    new IPTW(opts), 
  	    mkRegularizer(opts),
  	    new ADAGrad(opts), 
  	    opts)
    (nn, opts)
  }  
  
  class LearnParOptions extends ParLearner.Options with IPTW.Opts with MatDS.Opts with ADAGrad.Opts with L1Regularizer.Opts
  
  def learnPar(mat0:Mat, d:Int) = {
    val opts = new LearnParOptions
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    opts.lrate = 1f
  	val nn = new ParLearnerF(
  	    new MatDS(Array(mat0), opts), 
  	    opts, mkModel _,
  	    opts, mkRegularizer _,
  	    opts, mkUpdater _, 
  	    opts)
    (nn, opts)
  }
  
  def learnPar(mat0:Mat):(ParLearnerF, LearnParOptions) = learnPar(mat0, 0)
  
  def learnPar(mat0:Mat, targ:Mat, d:Int) = {
    val opts = new LearnParOptions
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    opts.lrate = 1f
    if (opts.links == null) opts.links = izeros(targ.nrows,1)
    opts.links.set(d)
    val nn = new ParLearnerF(
        new MatDS(Array(mat0, targ), opts), 
        opts, mkModel _,
        opts, mkRegularizer _,
        opts, mkUpdater _, 
        opts)
    (nn, opts)
  }
  
  def learnPar(mat0:Mat, targ:Mat):(ParLearnerF, LearnParOptions) = learnPar(mat0, targ, 0)
  
  class LearnFParOptions extends ParLearner.Options with IPTW.Opts with SFilesDS.Opts with ADAGrad.Opts with L1Regularizer.Opts
  
  def learnFParx(
    nstart:Int=FilesDS.encodeDate(2012,3,1,0), 
		nend:Int=FilesDS.encodeDate(2012,12,1,0), 
		d:Int = 0
		) = {
  	
  	val opts = new LearnFParOptions
  	opts.lrate = 1f
  	val nn = new ParLearnerxF(
  	    null,
  	    (dopts:DataSource.Opts, i:Int) => Experiments.Twitter.twitterWords(nstart, nend, opts.nthreads, i),
  	    opts, mkModel _,
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
  	opts.lrate = 1f
  	val nn = new ParLearnerF(
  	    Experiments.Twitter.twitterWords(nstart, nend),
  	    opts, mkModel _, 
        opts, mkRegularizer _,
  	    opts, mkUpdater _,
  	    opts
  	)
  	(nn, opts)
  }
}

