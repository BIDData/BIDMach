package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GDMat,GIMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.Learner

/**
 * Sparse Matrix Factorization with L2 loss (similar to ALS). 
 * 
 * '''Parameters'''
 - dim(256): Model dimension
 - uiter(5): Number of iterations on one block of data
 - miter(5): Number of CG iterations for model updates - not currently used in the SGD implementation.
 - lambdau(5f): Prior on the user (data) factor
 - lambdam(5f): Prior on model 
 - regumean(0f): prior on instance mean
 - regmmean(0f): Prior on feature mean
 - startup(1): Skip CG for this many iterations
 - traceConvergence(false): Print out trace info for convergence of the u iterations.
 - doUser(false): Apply the per-instance mean estimate. 
 - weightByUser(false): Weight loss equally by users, rather than their number of choices. 
 - ueps(1e-10f): A safety floor constant
 - uconvg(1e-3f): Stop u iteration if error smaller than this. 
 *
 * Other key parameters inherited from the learner, datasource and updater:
 - batchSize: the number of samples processed in a block
 - npasses(2): number of complete passes over the dataset
 - useGPU(true): Use GPU acceleration if available.
 *
 * '''Example:'''
 * 
 * a is a sparse word x document matrix
 * {{{
 * val (nn, opts) = SFA.learner(a)
 * opts.what             // prints the available options
 * opts.uiter=2          // customize options
 * nn.train              // train the model
 * nn.modelmat           // get the final model
 * nn.datamat            // get the other factor (requires opts.putBack=1)
 * }}}
 */

class SFA(override val opts:SFA.Opts = new SFA.Options) extends FactorModel(opts) {

  var mm:Mat = null;
  var traceMem = false;
  var pm:Mat = null;
  var mzero:Mat = null;
  var Minv:Mat = null;
  var diagM:Mat = null;
  var slm:Mat = null; 
  var mlm:Mat = null; 
  var iavg:Mat = null;
  var avg:Mat = null;
  var lamu:Mat = null;
  var itemsum:Mat = null;
  var itemcount:Mat = null;
  var nfeats:Int = 0;
  var totratings:Double = 0;
  var nratings:Double = 0;
  // For integrated ADAGrad updater
  var vexp:Mat = null;
  var texp:Mat = null;
  var lrate:Mat = null;
  var sumsq:Mat = null;
  var firststep = -1f;
  var waitsteps = 0;
  var epsilon = 0f;

  
  override def init() = {
    mats = datasource.next;
  	datasource.reset;
	  nfeats = mats(0).nrows;
	  val batchSize = mats(0).ncols;
    val d = opts.dim;
    if (refresh) {
    	mm = normrnd(0,0.01f,d,nfeats);
    	mm = convertMat(mm);
    	avg = mm.zeros(1,1)
    	iavg = mm.zeros(nfeats,1);
    	itemsum = mm.zeros(nfeats, 1);
    	itemcount = mm.zeros(nfeats, 1);
    	diagM = mkdiag(ones(d,1));
    	Minv = mm.zeros(d, d);
    	Minv <-- diagM;
    	setmodelmats(Array(mm, iavg, avg, Minv));
    } 
    useGPU = opts.useGPU && Mat.hasCUDA > 0;    
	  if (useGPU || useDouble) {
	    gmats = new Array[Mat](mats.length);
	  } else {
	    gmats = mats;
	  }  
	  
	  modelmats(0) = convertMat(modelmats(0));
	  modelmats(1) = convertMat(modelmats(1));
	  modelmats(2) = convertMat(modelmats(2));
	  modelmats(3) = convertMat(modelmats(3));
	  mm = modelmats(0);
    iavg = modelmats(1);
    avg = modelmats(2);
    Minv = modelmats(3);
    lamu = mm.ones(d, 1) ∘ opts.lambdau 
    if (opts.doUsers) lamu(0) = opts.regumean;
    slm = mm.ones(1,1) ∘ (opts.lambdam * batchSize);
    mlm = mm.ones(1,1) ∘ (opts.regmmean * batchSize);
    mzero = mm.zeros(1,1)

    if (opts.doUsers) mm(0,?) = 1f
    updatemats = new Array[Mat](3);
    if (opts.aopts != null) initADAGrad(d, nfeats);
  }
  
  def initADAGrad(d:Int, m:Int) = {
    val aopts = opts.asInstanceOf[ADAGrad.Opts];
    firststep = -1f;
    lrate = convertMat(aopts.lrate);
    texp = if (aopts.texp.asInstanceOf[AnyRef] != null) convertMat(aopts.texp) else null;
    vexp = convertMat(aopts.vexp);
    sumsq = convertMat(zeros(d, m));
    sumsq.set(aopts.initsumsq);
    waitsteps = aopts.waitsteps;
    epsilon = aopts.epsilon;
  }
  
  def setpm(pm0:Mat) = {
    pm = pm0
  }
 
  def uupdate(sdata0:Mat, user:Mat, ipass:Int, pos:Long):Unit =  {
// 	  val slu = sum((sdata>mzero), 1) * opts.lambdau
    if (opts.doUsers) mm(0,?) = 1f; 
    if (pos == 0) println("start "+user(?,0).t.toString)
    val sdata = sdata0 - (iavg + avg);
	  val b = mm * sdata;
	  val r = if (ipass < opts.startup || putBack < 0) {
	    // Setup CG on the first pass, or if no saved state
	  	user.clear
	  	b + 0
	  } else {
	    b - ((user ∘ lamu) + mm * DDS(mm, user, sdata))  // r = b - Ax
	  }
	  val z = Minv * r
 	  val p = z + 0
	  for (i <- 0 until opts.uiter) {
	  	val Ap = (p ∘ lamu) + mm * DDS(mm, p, sdata);
	  	SFA.PreCGupdate(p, r, z, Ap, user, Minv, opts.ueps, opts.uconvg)  // Should scale preconditioner by number of predictions per user
	  	if (opts.traceConverge) {
	  		println("i=%d, r=%f" format (i, norm(r)));
	  	}
	  }
	  if (pos == 0) println("end "+user(?,0).t.toString)
  }
  
  def mupdate(sdata0:Mat, user:Mat, ipass:Int, pos:Long):Unit = {
    val sdata = sdata0 - (iavg + avg);
    // values to be accumulated
    val ddsmu = DDS(mm, user, sdata);
    val diffs = sdata + 1f;
    diffs.contents ~ sdata.contents - ddsmu.contents;
    if (ipass < 1) {
    	itemsum ~ itemsum + sum(sdata0, 2);
    	itemcount ~ itemcount + sum(sdata0 != 0f, 2);
    	avg ~ sum(itemsum) / sum(itemcount);
    	iavg ~ ((itemsum + avg) / (itemcount + 1)) - avg;
    }
    updatemats(1) = (sum(diffs,2) - iavg*mlm) / (1 + sum(diffs>0f,2));   // per-item term estimator
    updatemats(2) = sum(diffs.contents) / (1 + diffs.contents.length);
    if (opts.weightByUser) {
      val iwt = 100f / max(sum(sdata != 0f), 100f); 
      val suser = user ∘ iwt;
      if (opts.aopts != null) {
      	if (firststep <= 0) firststep = pos.toFloat;
      	val step = (pos + firststep)/firststep;
      	ADAGrad.multUpdate(suser, diffs, modelmats(0), sumsq, null, lrate, texp, vexp, epsilon, step, waitsteps);
      } else {
      	updatemats(0) = suser *^ diffs - (mm ∘ slm);   // simple derivative
      }
    } else {
    	if (opts.aopts != null) {
      	if (firststep <= 0) firststep = pos.toFloat;
      	val step = (pos + firststep)/firststep;
      	ADAGrad.multUpdate(user, diffs, modelmats(0), sumsq, null, lrate, texp, vexp, epsilon, step, waitsteps);
    	} else {
    		updatemats(0) = user *^ diffs - (mm ∘ slm);     // simple derivative
    	}
    }
  }
  
    
  def mupdate0(sdata:Mat, user:Mat, ipass:Int):Unit = {
    // values to be accumulated
    val slm = sum((sdata != mzero), 2).t * opts.lambdam
    val rm = user *^ sdata - ((mm ∘ slm) + user *^ DDS(mm, user, sdata))          // accumulate res = (b - Ax)
    pm <-- rm
    if (ipass < 2) {
      val mtmp = mm + 0
    	for (i <- 0 until opts.miter) {
    		val Ap = (pm ∘ slm) + user *^ DDS(pm, user, sdata) 
    		CG.CGupdate(pm, rm, Ap, mtmp, opts.ueps, opts.uconvg)
    	}
    	updatemats(0) = mtmp
    } else {
      updatemats(0) = rm
      updatemats(1) = (pm ∘ slm) + user *^ DDS(pm, user, sdata)                    // accumulate Ap
    }
  }
  
  override def updatePass(ipass:Int) = {
    Minv <-- inv(50f/nfeats*FMat(mm *^ mm) + opts.lambdau * diagM); 
  }
   
  def evalfun(sdata:Mat, user:Mat, ipass:Int, pos:Long):FMat = {
    val preds = DDS(mm, user, sdata) + (iavg + avg);
  	val dc = sdata.contents;
  	val pc = preds.contents;
  	val vv = (dc - pc) ddot (dc - pc);
  	-sqrt(row(vv/sdata.nnz))
  }
  
  override def evalfun(sdata:Mat, user:Mat, preds:Mat, ipass:Int, pos:Long):FMat = {
    val spreds = DDS(mm, user, sdata) + (iavg + avg);
  	val dc = sdata.contents;
  	val pc = spreds.contents;
  	val vv = (dc - pc) ddot (dc - pc);
  	val xpreds = DDS(mm, user, preds) + (iavg + avg);
  	preds.contents <-- xpreds.contents;
  	-sqrt(row(vv/sdata.nnz))
  }
}

object SFA  {
  trait Opts extends FactorModel.Opts {
  	var ueps = 1e-10f
  	var uconvg = 1e-3f
  	var miter = 5
  	var lambdau = 5f
  	var lambdam = 5f
  	var regumean = 0f
  	var regmmean = 0f
  	var startup = 1
  	var traceConverge = false
  	var doUsers = true
  	var weightByUser = false
  	var aopts:ADAGrad.Opts = null;
  	var minv = 1f;
  	var maxv = 5f;
  	
  }  
  class Options extends Opts {} 
  
  def learner(mat0:Mat, d:Int) = {
    class xopts extends Learner.Options with SFA.Opts with MatDS.Opts with Grad.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = -1
    opts.npasses = 4
    opts.lrate = 0.1
    opts.initUval = 0f;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts),
  	    new SFA(opts), 
  	    null,
  	    new Grad(opts), opts)
    (nn, opts)
  }
  
  def learnerX(mat0:Mat, d:Int) = {
    class xopts extends Learner.Options with SFA.Opts with MatDS.Opts with ADAGrad.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = -1
    opts.npasses = 4
    opts.lrate = 0.1;
    opts.initUval = 0f;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1);
    opts.aopts = opts;
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts),
  	    new SFA(opts), 
  	    null,
  	    null, opts);
    (nn, opts)
  }
  
  def learner(mat0:Mat, user0:Mat, d:Int) = {
    class xopts extends Learner.Options with SFA.Opts with MatDS.Opts with Grad.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = 1
    opts.npasses = 4
    opts.lrate = 0.1;
    opts.initUval = 0f;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    val nn = new Learner(
        new MatDS(Array(mat0, user0), opts),
        new SFA(opts), 
        null,
        new Grad(opts), opts)
    (nn, opts)
  }
  
  def learnerX(mat0:Mat, user0:Mat, d:Int) = {
    class xopts extends Learner.Options with SFA.Opts with MatDS.Opts with ADAGrad.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = 1
    opts.npasses = 4
    opts.lrate = 0.1;
    opts.initUval = 0f;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1);
    opts.aopts = opts;
    val nn = new Learner(
        new MatDS(Array(mat0, user0), opts),
        new SFA(opts), 
        null,
        null, opts)
    (nn, opts)
  }
  
   def predictor(model0:Model, mat1:Mat, preds:Mat) = {
  	class xopts extends Learner.Options with SFA.Opts with MatDS.Opts with Grad.Opts
    val model = model0.asInstanceOf[SFA]
    val nopts = new xopts;
    nopts.batchSize = math.min(10000, mat1.ncols/30 + 1)
    nopts.putBack = 1
    val newmod = new SFA(nopts);
    newmod.refresh = false
    newmod.copyFrom(model);
    newmod.Minv = model.Minv;
    val mopts = model.opts.asInstanceOf[SFA.Opts];
    nopts.dim = mopts.dim;
    nopts.uconvg = mopts.uconvg;
    nopts.miter = mopts.miter;
    nopts.lambdau = mopts.lambdau;
    nopts.lambdam = mopts.lambdam;
    nopts.regumean = mopts.regumean;
    nopts.doUsers = mopts.doUsers;
    nopts.weightByUser = mopts.weightByUser;
    val nn = new Learner(
        new MatDS(Array(mat1, preds), nopts), 
        newmod, 
        null,
        null,
        nopts)
    (nn, nopts)
  }
   
  def predictor(model0:Model, mat1:Mat, user:Mat, preds:Mat) = {
  	class xopts extends Learner.Options with SFA.Opts with MatDS.Opts with Grad.Opts
    val model = model0.asInstanceOf[SFA]
    val nopts = new xopts;
    nopts.batchSize = math.min(10000, mat1.ncols/30 + 1)
    nopts.putBack = 2
    val newmod = new SFA(nopts);
    newmod.refresh = false
    newmod.copyFrom(model);
    newmod.Minv = model.Minv;
    val mopts = model.opts.asInstanceOf[SFA.Opts];
    nopts.dim = mopts.dim;
    nopts.uconvg = mopts.uconvg;
    nopts.miter = mopts.miter;
    nopts.lambdau = mopts.lambdau;
    nopts.lambdam = mopts.lambdam;
    nopts.regumean = mopts.regumean;
    nopts.doUsers = mopts.doUsers;
    nopts.weightByUser = mopts.weightByUser;
    val nn = new Learner(
        new MatDS(Array(mat1, user, preds), nopts), 
        newmod, 
        null,
        null,
        nopts)
    (nn, nopts)
  }
    // Preconditioned CG update
  def PreCGupdate(p:Mat, r:Mat, z:Mat, Ap:Mat, x:Mat, Minv:Mat, weps:Float, convgd:Float) = {
    val safe = 300f;
  	val pAp = (p dot Ap);
  	max(pAp, weps, pAp);
  	val rsold = (r dot z);
  	val convec = rsold > convgd;             // Check convergence
  	val alpha = convec ∘ (rsold / pAp);       // Only process unconverged elements
  	min(alpha, safe, alpha);
  	x ~ x + (p ∘ alpha);
  	r ~ r - (Ap ∘ alpha);
  	z ~ Minv * r;
  	val rsnew = (z dot r);                    // order is important to avoid aliasing
  	max(rsold, weps, rsold);
  	val beta = convec ∘ (rsnew / rsold);
  	min(beta, safe, beta);  	
  	p ~ z + (p ∘ beta);
  }
}


