package BIDMach.models

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.Learner

class ALSModel(override val opts:ALSModel.Opts = new ALSModel.Options) extends FactorModel(opts) { 
  var mm:Mat = null
  var traceMem = false
  var pm:Mat = null
  var mzero:Mat = null
  
  override def init(datasource:DataSource) = {
    mats = datasource.next
	  datasource.reset
	  val m = size(mats(0), 1)
    val d = opts.dim    
    modelmats = new Array[Mat](1)
    mm = rand(d,m) - 0.5f
	  useGPU = opts.useGPU && Mat.hasCUDA > 0
	  if (useGPU) {
	    gmats = new Array[Mat](mats.length)
	    mm = GMat(mm)
	  } else {
	    gmats = mats
	  }     
    modelmats(0) = mm
    mzero = mm.zeros(1,1)
    updatemats = new Array[Mat](2)
  }
  
  def setpm(pm0:Mat) = {
    pm = pm0
  }
 
  def uupdate(sdata:Mat, user:Mat, ipass:Int):Unit =  {
// 	  val slu = sum((sdata>mzero), 1) * opts.lambdau
	  val slu = opts.lambdau
	  val b = mm * sdata
	  val r = if (ipass < opts.startup || opts.putBack < 0) {
	    // Setup CG on the first pass, or if no saved state
	  	user.clear
	  	b
	  } else {
	    b - (user ∘ slu + mm * DDS(mm, user, sdata))  // r = b - Ax
	  }
 	  val p = b + 0
	  for (i <- 0 until opts.uiter) {
	  	val Ap = (p ∘ slu) + mm * DDS(mm, p, sdata)
	  	CGUpdater.CGupdate(p, r, Ap, user, opts.ueps, opts.uconvg)
	  }
  }
  
  def mupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    // values to be accumulated
    val slm = opts.lambdam
    updatemats(0) = user *^ sdata - ((mm ∘ slm) + user *^ DDS(mm, user, sdata))   // derivative
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
    		CGUpdater.CGupdate(pm, rm, Ap, mtmp, opts.ueps, opts.uconvg)
    	}
    	updatemats(0) = mtmp
    } else {
      updatemats(0) = rm
      updatemats(1) = (pm ∘ slm) + user *^ DDS(pm, user, sdata)                    // accumulate Ap
    }
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

object ALSModel  {
  trait Opts extends FactorModel.Opts {
  	var ueps = 1e-10f
  	var uconvg = 1e-2f
  	var miter = 4
  	var lambdau = 0.2f
  	var lambdam = 0.2f
  	var startup = 5
  }  
  class Options extends Opts {} 
  
  def learn(mat0:Mat, d:Int = 256) = {
    class xopts extends Learner.Options with ALSModel.Opts with MatDataSource.Opts with ADAGradUpdater.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = 2
    opts.blockSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatDataSource(Array(mat0:Mat), opts), 
  			new ALSModel(opts), 
  			null, 
  			new ADAGradUpdater(opts), opts)
    (nn, opts)
  }
}


