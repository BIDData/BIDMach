package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.CUMAT
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach._

/**
 * Factorization Machine Model.
 * This class computes a factorization machine model a la
 * 
 * Steffen Rendle (2012): Factorization Machines with libFM, in ACM Trans. Intell. Syst. Technol., 3(3), May.
 * 
 * We depart slightly from the original FM formulation by including both positive definite and negative definite factors.
 * While the positive definite factor can approximate any matrix in the limit, using both positive and negative definite factors
 * should give better performance for a fixed number of factors. This is what we observed on several datasets. 
 * With both positive definite and negative definite factors, there should also be no need to remove diagonal terms, since
 * the positive and negative factorizations already form a conventional eigendecomposition (a best least-squares fit for a given
 * number of factors) of the matrix of second-order interactions.
 * 
 * The types of model are given by the values of opts.links (IMat) and are the same as for GLM models. They are:
 - 0 = linear model (squared loss)
 - 1 = logistic model (logistic loss)
 - 2 = absolute logistic (hinge loss on logistic prediction)
 - 3 = SVM model (hinge loss)
 * 
 * Options are:
 -   links: an IMat whose nrows should equal the number of targets. Values as above. Can be different for different targets.
 -   iweight: an FMat typically used to select a weight row from the input. i.e. iweight = 0,1,0,0,0 uses the second
 *            row of input data as weights to be applied to input samples. The iweight field should be 0 in mask. 
 -   dim1: Dimension of the positive definite factor
 -   dim2: Dimension of the negative definite factor
 -   strictFM: the exact FM model zeros the diagonal terms of the factorization. As mentioned above, this probably isn't needed
 *             in our version of the model, but its available.
 *            
 * Inherited from Regression Model:
 -   rmask: FMat, optional, 0-1-valued. Used to ignore certain input rows (which are targets or weights). 
 *          Zero value in an element will ignore the corresponding row. 
 -   targets: FMat, optional, 0-1-valued. ntargs x nfeats. Used to specify which input features corresponding to targets. 
 -   targmap: FMat, optional, 0-1-valued. nntargs x ntargs. Used to replicate actual targets, e.g. to train multiple models
 *            (usually with different parameters) for the same target. 
 *            
 * Some convenience functions for training:
 * {{{
 * val (mm, opts) = FM.learner(a, d)     // On an input matrix a including targets (set opts.targets to specify them), 
 *                                       // learns an FM model of type d. 
 *                                       // returns the model (nn) and the options class (opts). 
 * val (mm, opts) = FM.learner(a, c, d)  // On an input matrix a and target matrix c, learns an FM model of type d. 
 *                                       // returns the model (nn) and the options class (opts). 
 * val (nn, nopts) = FM.predictor(model, ta, pc, d) // constructs a prediction learner from an existing model. returns the learner and options. 
 *                                       // pc should be the same dims as the test label matrix, and will contain results after nn.predict
 * val (mm, mopts, nn, nopts) = FM.learner(a, c, ta, pc, d) // a = training data, c = training labels, ta = test data, pc = prediction matrix, d = type.
 *                                       // returns a training learner mm, with options mopts. Also returns a prediction model nn with its own options.
 *                                       // typically set options, then do mm.train; nn.predict with results in pc.  
 * val (mm, opts) = learner(ds)          // Build a learner for a general datasource ds (e.g. a files data source). 
 * }}}
 * 
 */

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
  var xs:Mat = null
  var ulim:Mat = null
  var llim:Mat = null
  
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
    	val rmat1 = normrnd(0, opts.initscale/math.sqrt(opts.dim1).toFloat, opts.dim1, mv.ncols);
    	mm1 = if (useGPU) GMat(rmat1) else rmat1;
    	val rmat2 = normrnd(0, opts.initscale/math.sqrt(opts.dim2).toFloat, opts.dim2, mv.ncols)
    	mm2 = if (useGPU) GMat(rmat2) else rmat2;
    	ulim = if (useGPU) GMat(row(opts.lim)) else row(opts.lim)
    	llim = if (useGPU) GMat(row(-opts.lim)) else row(-opts.lim)
    	setmodelmats(Array(mv, mm1, mm2));
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
  
  // Update the positive/negative factorizations
  def mupdate3(in:Mat, targ:Mat, dweights:Mat) = {
    val ftarg = full(targ);
    val vt1 = mm1 * in
    val vt2 = mm2 * in
    val eta = mv * in + (vt1 ∙ vt1) - (vt2 ∙ vt2)
    if (opts.strictFM) {   // Strictly follow the FM formula (remove diag terms) vs. let linear predictor cancel them. 
      xs = in.copy
      (xs.contents ~ xs.contents) ∘ xs.contents          // xs is the element-wise square of in.
      eta ~ eta - (((mm1 ∘ mm1) - (mm2 ∘ mm2)) * xs) 
    }
    if (opts.lim > 0) {
      max(eta, llim, eta);
      min(eta, ulim, eta);
    }
    GLM.preds(eta, eta, mylinks, totflops)
    GLM.derivs(eta, ftarg, eta, mylinks, totflops)
    if (dweights.asInstanceOf[AnyRef] != null) eta ~ eta ∘ dweights
    uv ~ eta *^ in
    um1 ~ ((eta * 2f) ∘ vt1) *^ in
    um2 ~ ((eta * -2f) ∘ vt2) *^ in
    if (opts.strictFM) {
      val xeta = (eta * 2f) *^ xs
      um1 ~ um1 - (mm1 ∘ xeta);
      um2 ~ um2 + (mm2 ∘ xeta);
    }
    if (mask.asInstanceOf[AnyRef] != null) {
      uv ~ uv ∘ mask;
      um1 ~ um1 ∘ mask;
      um2 ~ um2 ∘ mask;
    }
  }
  
  // Update a simple factorization A*B for the second order terms. 
  def mupdate4(in:Mat, targ:Mat, dweights:Mat) = {
    val ftarg = full(targ);
    val vt1 = mm1 * in
    val vt2 = mm2 * in
    val eta = mv * in + (vt1 ∙ vt2)
    GLM.preds(eta, eta, mylinks, totflops)
    GLM.derivs(eta, ftarg, eta, mylinks, totflops)
    if (dweights.asInstanceOf[AnyRef] != null) eta ~ eta ∘ dweights
    uv ~ eta *^ in
    um1 ~ (eta ∘ vt2) *^ in
    um2 ~ (eta ∘ vt1) *^ in
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
  
  // Evaluate the positive/negative factorizations
  
  def meval3(in:Mat, targ:Mat, dweights:Mat):FMat = {
    val ftarg = full(targ)
    val vt1 = mm1 * in
    val vt2 = mm2 * in
    val eta = mv * in + (vt1 dot vt1) - (vt2 dot vt2)
    if (opts.strictFM) {
      in.contents ~ in.contents ∘ in.contents;
      eta ~ eta - ((mm1 ∘ mm1) * in);
      eta ~ eta + ((mm2 ∘ mm2) * in);
    }
    if (opts.lim > 0) {
      max(eta, - opts.lim, eta)
      min(eta, opts.lim, eta)
    }
    GLM.preds(eta, eta, mylinks, totflops)
    val v = GLM.llfun(eta, ftarg, mylinks, totflops)
    if (putBack >= 0) {targ <-- eta}
    if (dweights.asInstanceOf[AnyRef] != null) {
      FMat(sum(v ∘  dweights, 2) / sum(dweights))
    } else {
      FMat(mean(v, 2))
    }
  }
  
  // evaluate a simple A*B factorization of the interactions.
  
  def meval4(in:Mat, targ:Mat, dweights:Mat):FMat = {
    val ftarg = full(targ)
    val vt1 = mm1 * in
    val vt2 = mm2 * in
    val eta = mv * in + (vt1 dot vt2)
    GLM.preds(eta, eta, mylinks, totflops)
    val v = GLM.llfun(eta, ftarg, mylinks, totflops)
    if (putBack >= 0) {targ <-- eta}
    if (dweights.asInstanceOf[AnyRef] != null) {
      FMat(sum(v ∘  dweights, 2) / sum(dweights))
    } else {
      FMat(mean(v, 2))
    }
  }

}

object FM {
  trait Opts extends GLM.Opts {
    var strictFM = false;
    var dim1 = 32
    var dim2 = 32
    var initscale = 0.1f
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
  
   // This function constructs a learner and a predictor. 
  def learner(mat0:Mat, targ:Mat, mat1:Mat, preds:Mat, d:Int):(Learner, LearnOptions, Learner, LearnOptions) = {
    val mopts = new LearnOptions;
    val nopts = new LearnOptions;
    mopts.lrate = row(1f, 0.1f, 0.1f)
    mopts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    mopts.autoReset = false
    if (mopts.links == null) mopts.links = izeros(targ.nrows,1)
    nopts.links = mopts.links
    mopts.links.set(d)
    nopts.batchSize = mopts.batchSize
    nopts.putBack = 1
    val model = new FM(mopts)
    val mm = new Learner(
        new MatDS(Array(mat0, targ), mopts), 
        model, 
        mkRegularizer(mopts),
        new ADAGrad(mopts), mopts)
    val nn = new Learner(
        new MatDS(Array(mat1, preds), nopts), 
        model, 
        null,
        null, 
        nopts)
    (mm, mopts, nn, nopts)
  }
  
  // This function constructs a predictor from an existing model 
  def predictor(model:Model, mat1:Mat, preds:Mat, d:Int):(Learner, LearnOptions) = {
    val nopts = new LearnOptions;
    nopts.batchSize = math.min(10000, mat1.ncols/30 + 1)
    if (nopts.links == null) nopts.links = izeros(preds.nrows,1)
    nopts.links.set(d)
    nopts.putBack = 1
    val nn = new Learner(
        new MatDS(Array(mat1, preds), nopts), 
        model, 
        null,
        null,
        nopts)
    (nn, nopts)
  }
  
  class FMOptions extends Learner.Options with FM.Opts with ADAGrad.Opts with L1Regularizer.Opts 

  // A learner that uses a general data source (e.g. a files data source). 
  // The datasource options (like batchSize) need to be set externally. 
  def learner(ds:DataSource):(Learner, FMOptions) = {
    val mopts = new FMOptions;
    mopts.lrate = row(0.01f, 0.001f, 0.001f)
    mopts.autoReset = false
    val model = new FM(mopts)
    val mm = new Learner(
        ds, 
        model, 
        mkRegularizer(mopts),
        new ADAGrad(mopts), mopts)
    (mm, mopts)
  }
     
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
  	    (dopts:DataSource.Opts, i:Int) => Experiments.Twitter.twitterWords(nstart, nend, opts.nthreads, i),
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
  	    Experiments.Twitter.twitterWords(nstart, nend),
  	    opts, mkFMModel _, 
        opts, mkRegularizer _,
  	    opts, mkUpdater _,
  	    opts
  	)
  	(nn, opts)
  }
}

