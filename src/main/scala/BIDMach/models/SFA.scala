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

  
  override def init() = {
    mats = datasource.next;
  	datasource.reset;
	  nfeats = mats(0).nrows;
	  val batchSize = mats(0).ncols;
    val d = opts.dim;
    mm = normrnd(0,0.01f,d,nfeats);
    diagM = mkdiag(ones(d,1)) 
    useGPU = opts.useGPU && Mat.hasCUDA > 0;
    
	  if (useGPU || useDouble) {
	    gmats = new Array[Mat](mats.length);
	  } else {
	    gmats = mats;
	  }  
	  if (useGPU) {
	    if (useDouble) mm = GDMat(mm) else mm = GMat(mm);
	  } else {
	    if (useDouble) mm = DMat(mm);
	  }
	  Minv = mm.zeros(d, d);
	  Minv <-- diagM;
    lamu = mm.ones(d, 1) ∘ opts.lambdau 
    if (opts.doUsers) lamu(0) = opts.regumean;
    slm = mm.ones(1,1) ∘ (opts.lambdam * batchSize);
    mlm = mm.ones(1,1) ∘ (opts.regmmean * batchSize);
	  iavg = mm.zeros(nfeats,1);
	  itemsum = mm.zeros(nfeats, 1);
	  itemcount = mm.zeros(nfeats, 1);
    mzero = mm.zeros(1,1)
    setmodelmats(Array(mm, iavg));
    if (opts.doUsers) mm(0,?) = 1f
    updatemats = new Array[Mat](2)
  }
  
  def setpm(pm0:Mat) = {
    pm = pm0
  }
 
  def uupdate(sdata0:Mat, user:Mat, ipass:Int):Unit =  {
// 	  val slu = sum((sdata>mzero), 1) * opts.lambdau
    if (opts.doUsers) mm(0,?) = 1f; 
    val sdata = sdata0 - iavg;
	  val b = mm * sdata
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
  }
  
  def mupdate(sdata0:Mat, user:Mat, ipass:Int):Unit = {
    val sdata = sdata0 - iavg;
    // values to be accumulated
    val ddsmu = DDS(mm, user, sdata);
    if (opts.weightByUser) {
      val iwt = 100f / max(sum(sdata != 0f), 100f); 
      val suser = user ∘ iwt;
      updatemats(0) = suser *^ sdata - ((mm ∘ slm) + suser *^ ddsmu);   // simple derivative for ADAGRAD
    } else {
    	updatemats(0) = user *^ sdata - ((mm ∘ slm) + user *^ ddsmu);     // simple derivative for ADAGRAD
    }
    if (ipass == 0) {
    	itemsum ~ itemsum + sum(sdata0, 2);
    	itemcount ~ itemcount + sum(sdata0 != 0, 2);
    	avg = sum(itemsum) / sum(itemcount);
    	iavg ~ (itemsum + avg) / (itemcount + 1);
    }
    updatemats(1) = sum(sdata, 2) - sum(ddsmu, 2) - (iavg - avg)*mlm;   // per-item term estimator
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
   
  def evalfun(sdata:Mat, user:Mat, ipass:Int):FMat = {
        val preds = DDS(mm, user, sdata) + iavg
  	val dc = sdata.contents
  	val pc = preds.contents
  	val vv = (dc - pc) ddot (dc - pc)
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
  	
  }  
  class Options extends Opts {} 
  
  def learner(mat0:Mat, d:Int) = {
    class xopts extends Learner.Options with SFA.Opts with MatDS.Opts with ADAGrad.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = -1
    opts.npasses = 4
    opts.lrate = 0.1
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts),
  	    new SFA(opts), 
  	    null,
  	    new ADAGrad(opts), opts)
    (nn, opts)
  }
  
  def learner(mat0:Mat, user0:Mat, d:Int) = {
    class xopts extends Learner.Options with SFA.Opts with MatDS.Opts with ADAGrad.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = 1
    opts.npasses = 4
    opts.lrate = 0.1
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    val nn = new Learner(
        new MatDS(Array(mat0, user0), opts),
        new SFA(opts), 
        null,
        new ADAGrad(opts), opts)
    (nn, opts)
  }
    // Preconditioned CG update
  def PreCGupdate(p:Mat, r:Mat, z:Mat, Ap:Mat, x:Mat, Minv:Mat, weps:Float, convgd:Float) = {
  	val pAp = (p dot Ap)
  	max(pAp, weps, pAp)
  	val rsold = (r dot z) 
  	val convec = rsold > convgd              // Check convergence
  	val alpha = convec ∘ (rsold / pAp)       // Only process unconverged elements
  	min(alpha, 10f, alpha)
  	x ~ x + (p ∘ alpha)
  	r ~ r - (Ap ∘ alpha)
  	z ~ Minv * r
  	val rsnew = (z dot r)                    // order is important to avoid aliasing
  	max(rsold, weps, rsold)
  	val beta = convec ∘ (rsnew / rsold)
  	min(beta, 10f, beta)
  	p ~ z + (p ∘ beta)
  }
}


