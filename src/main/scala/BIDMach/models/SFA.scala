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
  
  override def init() = {
    mats = datasource.next
  	datasource.reset
	  val m = size(mats(0), 1)
    val d = opts.dim    
    modelmats = new Array[Mat](1)
    mm = rand(d,m) - 0.5f
    diagM = mkdiag(ones(d,1)) 
    useGPU = opts.useGPU && Mat.hasCUDA > 0
	  if (useGPU) {
	    gmats = new Array[Mat](mats.length)
	    mm = GMat(mm)
	    Minv = GMat(diagM)
	  } else {
	    gmats = mats
	    Minv = diagM + 0
	  }     
    modelmats(0) = mm
    mzero = mm.zeros(1,1)
    if (opts.forceOnes) mm(1,?) = 1f
    updatemats = new Array[Mat](2)
  }
  
  def setpm(pm0:Mat) = {
    pm = pm0
  }
 
  def uupdate(sdata:Mat, user:Mat, ipass:Int):Unit =  {
// 	  val slu = sum((sdata>mzero), 1) * opts.lambdau
    if (opts.forceOnes) mm(1,?) = 1f;
	  val slu = opts.lambdau
	  val b = mm * sdata
	  val r = if (ipass < opts.startup || putBack < 0) {
	    // Setup CG on the first pass, or if no saved state
	  	user.clear
	  	b
	  } else {
	    b - (user ∘ slu + mm * DDS(mm, user, sdata))  // r = b - Ax
	  }
	  val z = Minv * r
 	  val p = z + 0
	  for (i <- 0 until opts.uiter) {
	  	val Ap = (p ∘ slu) + mm * DDS(mm, p, sdata)
//	  	CG.CGupdate(p, r, Ap, user, opts.ueps, opts.uconvg)
	  	CG.PreCGupdate(p, r, z, Ap, user, Minv, opts.ueps, opts.uconvg)  // Should scale preconditioner by number of predictions per user
	  }
  }
  
  def mupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    // values to be accumulated
    if (opts.forceOnes) user(0,?) = 1f;
    val slm = opts.lambdam
    updatemats(0) = user *^ sdata - ((mm ∘ slm) + user *^ DDS(mm, user, sdata))   // simple derivative for ADAGRAD
  }
    
  def mupdate0(sdata:Mat, user:Mat, ipass:Int):Unit = {
    // values to be accumulated
    val slm = sum((sdata>mzero), 2).t * opts.lambdam
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
    Minv <-- inv(0.01f*FMat(mm *^ mm) + opts.lambdau * diagM); 
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
  	var uconvg = 1e-2f
  	var miter = 4
  	var lambdau = 0.2f
  	var lambdam = 0.2f
  	var startup = 5
  	var forceOnes = false
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
}


