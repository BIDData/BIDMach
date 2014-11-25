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
  
  var mylinks:Mat = null;
  var iweight:Mat = null;
  
  val linkArray = GLM.linkArray
  
  var totflops = 0L
  
  var mv:Mat = null
  var mm1:Mat = null
  var mm2:Mat = null
  var uv:Mat = null
  var um1:Mat = null
  var um2:Mat = null
  
  override def copyTo(mod:Model) = {
    super.copyTo(mod);
    val rmod = mod.asInstanceOf[FM];
    rmod.mylinks = mylinks;
    rmod.iweight = iweight;    
    rmod.mv = mv;
    rmod.mm1 = mm1;
    rmod.mm2 = mm2;
    rmod.uv = uv;
    rmod.um1 = um2;
    rmod.um2 = um2;
  }
  
  override def init() = {
    super.init()
    mylinks = if (useGPU) GIMat(opts.links) else opts.links
    iweight = if (useGPU && opts.iweight.asInstanceOf[AnyRef] != null) GMat(opts.iweight) else opts.iweight
    if (refresh) {
    	mv = modelmats(0);
    	val rmat1 = rand(opts.dim1, mv.ncols) - 0.5f ;
    	rmat1 ~ rmat1 *@ (sp * (1f/math.sqrt(opts.dim1).toFloat));
    	mm1 = if (useGPU) GMat(rmat1) else rmat1;
    	val rmat2 = rand(opts.dim2, mv.ncols) - 0.5f ;
    	rmat2 ~ rmat2 *@ (sp * (1f/math.sqrt(opts.dim2).toFloat));
    	mm2 = if (useGPU) GMat(rmat2) else rmat2;
    	modelmats = Array(mv, mm1, mm2);
    	if (mask.asInstanceOf[AnyRef] != null) {
    		mv ~ mv ∘ mask;
    		mm1 ~ mm1 ∘ mask;
    		mm2 ~ mm2 ∘ mask;
    	}
    }
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
    val alltargs = if (targmap.asInstanceOf[AnyRef] != null) targmap * targs else targs
    val dweights = if (iweight.asInstanceOf[AnyRef] != null) iweight * in else null
    mupdate3(in, alltargs, dweights)
  }
  
  def mupdate2(in:Mat, targ:Mat) = mupdate3(in, targ, null);
  
  def mupdate3(in:Mat, targ:Mat, dweights:Mat) = {
    val ftarg = full(targ);
    val vt1 = mm1 * in
    val vt2 = mm2 * in
    val eta = mv * in + (vt1 dot vt1) - (vt2 dot vt2)
    GLM.preds(eta, eta, mylinks, totflops)
    GLM.derivs(eta, ftarg, eta, mylinks, totflops)
    if (dweights.asInstanceOf[AnyRef] != null) eta ~ eta ∘ dweights
    uv ~ eta *^ in
    um1 ~ (vt1 ∘ (eta * 2f)) *^ in
    um2 ~ (vt2 ∘ (eta * -2f)) *^ in
    if (mask.asInstanceOf[AnyRef] != null) {
      uv ~ uv ∘ mask;
      um1 ~ um1 ∘ mask;
      um2 ~ um2 ∘ mask;
    }
  }
  
  def meval(in:Mat):FMat = {
    val targs = targets * in
    min(targs, 1f, targs)
    val alltargs = if (targmap.asInstanceOf[AnyRef] != null) targmap * targs else targs
    val dweights = if (iweight.asInstanceOf[AnyRef] != null) iweight * in else null
    meval3(in, alltargs, dweights)
  }
  
  def meval2(in:Mat, targ:Mat):FMat = meval3(in, targ, null)
  
  def meval3(in:Mat, targ:Mat, dweights:Mat):FMat = {
    val ftarg = full(targ)
    val vt1 = mm1 * in
    val vt2 = mm2 * in
    val eta = mv * in + (vt1 dot vt1) - (vt2 dot vt2)
    GLM.preds(eta, eta, mylinks, totflops)
    val v = GLM.llfun(eta, ftarg, mylinks, totflops)
    if (dweights.asInstanceOf[AnyRef] != null) {
      FMat(sum(v ∘  dweights, 2) / sum(dweights))
    } else {
      FMat(mean(v, 2))
    }
  }

}

object FM {
  trait Opts extends RegressionModel.Opts {
    var links:IMat = null;
    var iweight:FMat = null;
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
    Array(new L1Regularizer(nopts.asInstanceOf[L1Regularizer.Opts]))
  } 
  
  class LearnOptions extends Learner.Options with FM.Opts with MatDS.Opts with ADAGrad.Opts with L1Regularizer.Opts
     
  def learner(mat0:Mat, d:Int = 0) = { 
    val opts = new LearnOptions
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    new FM(opts), 
  	    mkRegularizer(opts),
  	    new ADAGrad(opts), opts)
    (nn, opts)
  }
  
  def learner(mat0:Mat):(Learner, LearnOptions) = learner(mat0, 0)
  
  def learner(mat0:Mat, targ:Mat, d:Int) = {
    val opts = new LearnOptions
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    if (opts.links == null) opts.links = izeros(targ.nrows,1)
    opts.links.set(d)
    val nn = new Learner(
        new MatDS(Array(mat0, targ), opts), 
        new FM(opts), 
        mkRegularizer(opts),
        new ADAGrad(opts),
        opts)
    (nn, opts)
  }
  
  def learner(mat0:Mat, targ:Mat):(Learner, LearnOptions) = learner(mat0, targ, 0)
     
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
  
  class LearnParOptions extends ParLearner.Options with FM.Opts with MatDS.Opts with ADAGrad.Opts with L1Regularizer.Opts
  
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
  
  class LearnFParOptions extends ParLearner.Options with FM.Opts with SFilesDS.Opts with ADAGrad.Opts with L1Regularizer.Opts
  
  def learnFParx(
    nstart:Int=FilesDS.encodeDate(2012,3,1,0), 
		nend:Int=FilesDS.encodeDate(2012,12,1,0), 
		d:Int = 0
		) = {
  	val opts = new LearnFParOptions
  	val nn = new ParLearnerxF(
  	    null,
  	    (dopts:DataSource.Opts, i:Int) => Twitter.twitterWords(nstart, nend, opts.nthreads, i),
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
  	    Twitter.twitterWords(nstart, nend),
  	    opts, mkFMModel _, 
        opts, mkRegularizer _,
  	    opts, mkUpdater _,
  	    opts
  	)
  	(nn, opts)
  }
}

