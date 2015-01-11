package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.Learner

class SFA(override val opts:SFA.Opts = new SFA.Options) extends FactorModel(opts) {

  var mm:Mat = null
  var traceMem = false
  var pm:Mat = null
  var mzero:Mat = null
  var Minv:Mat = null
  var diagM:Mat = null
  var scalem:Mat = null
  var avgrating:Float = 0
  var avgmat:Mat = null
  var itemcounts:Mat = null
  var icm:Mat = null
  var mscale:Mat = null
  var nfeats:Int = 0
  var totratings:Double = 0
  var nratings:Double = 0

  
  override def init() = {
    mats = datasource.next
  	datasource.reset
	  nfeats = mats(0).nrows
    val d = opts.dim    
    mm = rand(d,nfeats) - 0.5f
    diagM = mkdiag(ones(d,1)) 
    useGPU = opts.useGPU && Mat.hasCUDA > 0
    mscale = 0.9 ^ icol(0 until d)
    
	  if (useGPU) {
	    gmats = new Array[Mat](mats.length)
	    mm = GMat(mm)
	    Minv = GMat(diagM)
	    scalem = gzeros(d, d)
	    itemcounts = gzeros(nfeats,1)
	    mscale = GMat(mscale)
	  } else {
	    gmats = mats
	    Minv = diagM + 0
	    scalem = zeros(d, d)
	    itemcounts = zeros(nfeats,1)
	  }     
    mzero = mm.zeros(1,1)
    avgmat = mm.zeros(1,1)
    modelmats = Array(mm)
    if (opts.forceOnes) mm(1,?) = 1f
    updatemats = new Array[Mat](2)
  }
  
  def setpm(pm0:Mat) = {
    pm = pm0
  }
 
  def uupdate(sdata:Mat, user:Mat, ipass:Int):Unit =  {
// 	  val slu = sum((sdata>mzero), 1) * opts.lambdau
    if (opts.forceOnes) mm(1,?) = 1f;
    if (ipass == 0) {
	    totratings += sum(sdata.contents).dv;
	    nratings += sdata.nnz
	  } else {
	    sdata.contents ~ sdata.contents - avgmat;
	  }
	  val slu = opts.lambdau
	  val b = mm * sdata
	  val r = if (ipass < opts.startup || putBack < 0) {
	    // Setup CG on the first pass, or if no saved state
	  	user.clear
	  	b + 0
	  } else {
	    b - ((user ∘ slu) + mm * DDS(mm, user, sdata))  // r = b - Ax
	  }
	  val z = Minv * r
 	  val p = z + 0
	  for (i <- 0 until opts.uiter) {
	  	val Ap = (p ∘ slu) + mm * DDS(mm, p, sdata);
//	  	CG.CGupdate(p, r, Ap, user, opts.ueps, opts.uconvg)
	  	SFA.PreCGupdate(p, r, z, Ap, user, Minv, opts.ueps, opts.uconvg)  // Should scale preconditioner by number of predictions per user
	  	if (opts.traceConverge) {
	  		println("i=%d, r=%f" format (i, norm(r)));
	  	}
	  }
  }
  
  def mupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    // values to be accumulated
    if (opts.forceOnes) user(0,?) = 1f;
    val slm = opts.lambdam;
 //   val nratings = sum(sdata != 0);
//    user ~ user ∘ (nratings > 20f)
    if (opts.weightByUser) {
      val iwt = 100f / max(sum(sdata != 0f), 100f); 
      val suser = user ∘ iwt;
      updatemats(0) = suser *^ sdata - ((mm ∘ slm) + suser *^ DDS(mm, user, sdata))   // simple derivative for ADAGRAD
    } else {
    	updatemats(0) = user *^ sdata - ((mm ∘ slm) + user *^ DDS(mm, user, sdata))   // simple derivative for ADAGRAD
    }
  }
  

  def mupdate1(sdata:Mat, user:Mat, ipass:Int):Unit = {
  	val regularizer = if (opts.forceOnes) {
  	  user(0,?) = 1f;
  	  val mmc = mm.copy;
  	  mmc(1,?) = 0f;
  	  mscale ∘ mmc - sum(mmc);
  	} else {
  	  mscale ∘ mm - sum(mm);
  	}
//	  updatemats(0) = user *^ sdata - user *^ DDS(mm, user, sdata) + regularizer ∘ opts.lambdam;
	  updatemats(0) = user *^ sdata - ((mm ∘ opts.lambdam) + user *^ DDS(mm, user, sdata))   // simple derivative for ADAGRAD
	  
	  // scale model to orthonormal rows
//    val prod = FMat(mm *^ mm);
//    val ch = triinv(chol(prod));
//    scalem <-- ch;
//    val mtmp = scalem ^* mm;
//    mm <-- mtmp;
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
    if (ipass == 0) {
      avgrating = (totratings / nratings).toFloat;
      avgmat.set(avgrating);
      icm = maxi(itemcounts) / (itemcounts.t + 100f);
    }
    Minv <-- inv(50f/nfeats*FMat(mm *^ mm) + opts.lambdau * diagM); 
  }
   
  def evalfun(sdata:Mat, user:Mat):FMat = {  
  	val preds = DDS(mm, user, sdata)
  	val dc = sdata.contents
  	val pc = preds.contents
  	val vv = (dc - pc) ddot (dc - pc)
//  	println("pc: " + pc)
  	row(vv/sdata.nnz)
  }
}

object SFA  {
  trait Opts extends FactorModel.Opts {
  	var ueps = 1e-10f
  	var uconvg = 1e-3f
  	var miter = 8
  	var lambdau = 5f
  	var lambdam = 5f
  	var startup = 5
  	var traceConverge = false
  	var forceOnes = false
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


