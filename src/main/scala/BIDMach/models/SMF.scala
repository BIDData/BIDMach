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

class SMF(override val opts:SMF.Opts = new SMF.Options) extends FactorModel(opts) {

  var mm:Mat = null;
  var traceMem = false;
  var mzero:Mat = null;
  var slm:Mat = null; 
  var mlm:Mat = null; 
  var iavg:Mat = null;
  var avg:Mat = null;
  var lamu:Mat = null;
  var itemsum:Mat = null;
  var itemcount:Mat = null;
  var nfeats:Int = 0;
  var nratings:Double = 0;
  // For integrated ADAGrad updater
  var vexp:Mat = null;
  var texp:Mat = null;
  var pexp:Mat = null;
  var cscale:Mat = null;
  var lrate:Mat = null;
  var uscale:Mat = null;
  var sumsq:Mat = null;
  var firststep = -1f;
  var waitsteps = 0;
  var epsilon = 0f;
  var aopts:ADAGrad.Opts = null;

  
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
    	setmodelmats(Array(mm, iavg, avg));
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
	  mm = modelmats(0);
    iavg = modelmats(1);
    avg = modelmats(2);
    lamu = mm.ones(d, 1) ∘ opts.lambdau 
    if (opts.doUsers) lamu(0) = opts.regumean;
    slm = mm.ones(1,1) ∘ (opts.lambdam * batchSize);
    mlm = mm.ones(1,1) ∘ (opts.regmmean * batchSize);
    mzero = mm.zeros(1,1);
    uscale = mm.zeros(1,1);
    cscale = mm.ones(d, 1);
    cscale(0,0) = 0.0001f;
    if (opts.doUsers) mm(0,?) = 1f
    updatemats = new Array[Mat](3);
    updatemats(2) = mm.zeros(1,1);
    if (opts.aopts != null) initADAGrad(d, nfeats);
    vexp = convertMat(row(0.5f));
  }
  
  def initADAGrad(d:Int, m:Int) = {
  	aopts = opts.asInstanceOf[ADAGrad.Opts]
    firststep = -1f;
    lrate = convertMat(aopts.lrate);
    texp = if (aopts.texp.asInstanceOf[AnyRef] != null) convertMat(aopts.texp) else null;
    pexp = if (aopts.pexp.asInstanceOf[AnyRef] != null) convertMat(aopts.pexp) else null;
    vexp = convertMat(aopts.vexp);
    sumsq = convertMat(zeros(d, m));
    sumsq.set(aopts.initsumsq);
    waitsteps = aopts.waitsteps;
    epsilon = aopts.epsilon;
  }
 
  def uupdate(sdata0:Mat, user:Mat, ipass:Int, pos:Long):Unit =  { 
  	if (firststep <= 0) firststep = pos.toFloat;
  	val step = (pos + firststep)/firststep;
  	val texp = if (opts.asInstanceOf[Grad.Opts].texp.asInstanceOf[AnyRef] != null) {
  	  opts.asInstanceOf[Grad.Opts].texp.dv
  	} else {
  	  opts.asInstanceOf[Grad.Opts].pexp.dv
  	}
  	uscale.set(opts.urate * math.pow(ipass+1, - texp).toFloat) 
    val sdata = sdata0 - (iavg + avg);
	  if (putBack < 0) {
	  	user.clear
	  }
	  val b = mm * sdata;
	  val ucounts = sum(sdata0 != 0f);
	  val uci = (ucounts + 1f) ^ (- vexp);
	  for (i <- 0 until opts.uiter) {
	  	val preds = DDS(mm, user, sdata);
	  	val deriv = b - mm * preds - (user ∘ lamu);
	  	val du = (deriv ∘ uscale ∘ uci);
	  	if (opts.lsgd >= 0) {
	  		val dpreds = DDS(mm, du, sdata);
	  	  accept(sdata, user, du, preds, dpreds, uscale, lamu, false);
	  	} else {
	  		user ~ user + du;
	  	}

	  	if (opts.traceConverge) {
	  		println("step %d, loss %f" format (i, ((norm(sdata.contents - preds.contents) ^ 2f) + (sum(user dot (user ∘ lamu)))).dv/sdata.nnz));
	  	}
	  }
  }
  
  def mupdate(sdata0:Mat, user:Mat, ipass:Int, pos:Long):Unit = {
    val sdata = sdata0 - (iavg + avg);
    // values to be accumulated
    val preds = DDS(mm, user, sdata);
    val diffs = sdata + 2f;
    diffs.contents ~ sdata.contents - preds.contents;
    if (ipass < 1) {
    	itemsum ~ itemsum + sum(sdata0, 2);
    	itemcount ~ itemcount + sum(sdata0 != 0f, 2);
    	avg ~ sum(itemsum) / sum(itemcount);
    	iavg ~ ((itemsum + avg) / (itemcount + 1)) - avg;
    } 
    val icomp = sdata0 != 0f
    val icount = sum(sdata0 != 0f, 2);
    updatemats(1) = (sum(diffs,2) - iavg*mlm) / (icount + 1f);   // per-item term estimator
    updatemats(2) ~ sum(diffs.contents) / (diffs.contents.length + 1f);
    val wuser = if (opts.weightByUser) {
    	val iwt = 100f / max(sum(sdata != 0f), 100f); 
      user ∘ iwt;
    } else {
      user;
    }
    if (firststep <= 0) firststep = pos.toFloat;
    if (opts.lsgd >= 0 || opts.aopts == null) {
    	updatemats(0) = (wuser *^ diffs - (mm ∘ slm)) / ((icount + 1).t ^ vexp);   // simple derivative
    	if (opts.lsgd >= 0) {
    		val step = (pos + firststep)/firststep;
    		uscale.set((lrate.dv * math.pow(step, - texp.dv)).toFloat);
    		val dm = updatemats(0) ∘ uscale ∘ cscale;
    	  val dpreds = DDS(dm, user, sdata);
    	  accept(sdata, mm, dm, preds, dpreds, uscale, slm, true);
    	}
    } else {
    	if (texp.asInstanceOf[AnyRef] != null) {
    		val step = (pos + firststep)/firststep;
    		ADAGrad.multUpdate(wuser, diffs, modelmats(0), sumsq, null, lrate, texp, vexp, epsilon, step, waitsteps);
    	} else {
    		ADAGrad.multUpdate(wuser, diffs, modelmats(0), sumsq, null, lrate, pexp, vexp, epsilon, ipass + 1, waitsteps);
    	}
    }
    if (opts.doUsers) mm(0,?) = 1f;
  }
  
   def accept(sdata:Mat, mmod:Mat, du:Mat, preds:Mat, dpreds:Mat, scale:Mat, lambda:Mat, flip:Boolean) = {
 // 	println("sdata " + FMat(sdata.contents)(0->5,0).t)
    	val diff1 = preds + 0f;
	  	diff1.contents ~ sdata.contents - preds.contents;
//	  	println("sdata %d %s" format (if (flip) 1 else 0, FMat(sdata.contents)(0->5,0).t.toString));
//	  	println("preds %d %s" format (if (flip) 1 else 0, FMat(preds.contents)(0->5,0).t.toString));
//	  	println("diff %d %s" format (if (flip) 1 else 0, FMat(diff1.contents)(0->5,0).t.toString));
//	  	println("sdata "+FMat(sdata.contents)(0->5,0).t.toString);
	  	val diff2 = diff1 + 0f;
	  	diff2.contents ~ diff1.contents - dpreds.contents;
	  	diff1.contents ~ diff1.contents ∘ diff1.contents;
	  	diff2.contents ~ diff2.contents ∘ diff2.contents;
	  	val rmmod = mmod + 1f;
	  	normrnd(0, opts.lsgd, rmmod);
	  	val mmod2 = mmod + du + rmmod ∘ scale;
	  	val loss1 = (if (flip) sum(diff1,2).t else sum(diff1)) + (mmod dot (mmod ∘ lambda));
	  	val loss2 = (if (flip) sum(diff2,2).t else sum(diff2)) + (mmod2 dot (mmod2 ∘ lambda));
	  	
	  	val accprob = erfc((loss2 - loss1) /scale);	  	
	  	val rsel = accprob + 0f;
	  	rand(rsel);
	  	val selector = rsel < accprob;
	  	mmod ~ (mmod2 ∘ selector) + (mmod ∘ (1f - selector));
	  	if (opts.traceConverge) {
	  	  println("accepted %d %f %f %f" format (if (flip) 1 else 0, mean(selector).dv, mean(loss1).dv, mean(loss2).dv));
	  	}
  }
  
  def evalfun(sdata0:Mat, user:Mat, ipass:Int, pos:Long):FMat = {
  	val sdata = sdata0 - (iavg + avg);
    val preds = DDS(mm, user, sdata);
  	val dc = sdata.contents
  	val pc = preds.contents
  	val diff = dc - pc;
  	val vv = diff ddot diff;
  	-sqrt(row(vv/sdata.nnz))
  }
}

object SMF {
  trait Opts extends FactorModel.Opts {
  	var ueps = 1e-10f
  	var uconvg = 1e-3f
  	var miter = 5
  	var lambdau = 5f
  	var lambdam = 5f
  	var regumean = 0f
  	var regmmean = 0f
  	var urate = 0.1f
  	var lsgd = 0.1f
  	var traceConverge = false
  	var doUsers = true
  	var weightByUser = false
  	var aopts:ADAGrad.Opts = null;
  	var minv = 1f;
  	var maxv = 5f;
  	
  }  
  class Options extends Opts {} 
  
  def learner(mat0:Mat, d:Int) = {
    class xopts extends Learner.Options with SMF.Opts with MatDS.Opts with Grad.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = -1
    opts.npasses = 4
    opts.lrate = 0.1
    opts.initUval = 0f;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts),
  	    new SMF(opts), 
  	    null,
  	    new Grad(opts), opts)
    (nn, opts)
  }
  
  def learnerX(mat0:Mat, d:Int) = {
    class xopts extends Learner.Options with SMF.Opts with MatDS.Opts with ADAGrad.Opts
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
  	    new SMF(opts), 
  	    null,
  	    null, opts);
    (nn, opts)
  }
  
  def learner(mat0:Mat, user0:Mat, d:Int) = {
    class xopts extends Learner.Options with SMF.Opts with MatDS.Opts with Grad.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = 1
    opts.npasses = 4
    opts.lrate = 0.1;
    opts.initUval = 0f;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    val nn = new Learner(
        new MatDS(Array(mat0, user0), opts),
        new SMF(opts), 
        null,
        new Grad(opts), opts)
    (nn, opts)
  }
  
  def learnerX(mat0:Mat, user0:Mat, d:Int) = {
    class xopts extends Learner.Options with SMF.Opts with MatDS.Opts with ADAGrad.Opts
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
        new SMF(opts), 
        null,
        null, opts)
    (nn, opts)
  }
  
   def predictor(model0:Model, mat1:Mat, preds:Mat) = {
  	class xopts extends Learner.Options with SMF.Opts with MatDS.Opts with Grad.Opts
    val model = model0.asInstanceOf[SMF]
    val nopts = new xopts;
    nopts.batchSize = math.min(10000, mat1.ncols/30 + 1)
    nopts.putBack = 1
    val newmod = new SMF(nopts);
    newmod.refresh = false
    newmod.copyFrom(model);
    val mopts = model.opts.asInstanceOf[SMF.Opts];
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
}


