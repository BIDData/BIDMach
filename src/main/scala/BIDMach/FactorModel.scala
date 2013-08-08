package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._


class LDAModel(override val opts:LDAModel.Options = new LDAModel.Options) extends FactorModel(opts) { 
  var mm:Mat = null
  var alpha:Mat = null
  
  var traceMem = true
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    updatemats = new Array[Mat](1)
    mm = modelmats(0)
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols)
  }
  
  def uupdate(sdata:Mat, user:Mat):Unit = {
    if (opts.putBack < 0) user.set(1f)
	  for (i <- 0 until opts.uiter) {
	  	val preds = DDS(mm, user, sdata)	
	  	if (traceMem) println("uupdate %d %d %d, %d %f %d" format (mm.GUID, user.GUID, sdata.GUID, preds.GUID, GPUmem._1, getGPU))
	  	val dc = sdata.contents
	  	val pc = preds.contents
	  	println("uupdate1 %d %d %d %d %d" format (sdata.nnz, preds.nnz, dc.nrows, pc.nrows, getGPU))
	  	max(opts.weps, pc, pc)
	  	pc ~ dc / pc
	  	println("uupdate2 %d %d %d %d %d" format (sdata.nnz, preds.nnz, dc.nrows, pc.nrows, getGPU))
	  	val unew = user *@ (mm * preds) + opts.alpha
/*	  	val unew1 = mm * preds
	  	val unew = user *@ unew1
	  	unew ~ unew + opts.alpha */
	  	if (traceMem) println("uupdate %d %d %d, %d %d %d %d %f %d" format (mm.GUID, user.GUID, sdata.GUID, preds.GUID, dc.GUID, pc.GUID, unew.GUID, GPUmem._1, getGPU))
	  	if (opts.exppsi) exppsi(unew, unew)
	  	user <-- unew                                                     
	  }	  
  }
  
  def mupdate(sdata:Mat, user:Mat):Unit = {
    val ud = user *^ sdata
  	updatemats(0) <-- ud         
  	if (traceMem) println("mupdate %d %d %d %d" format (sdata.GUID, user.GUID, ud.GUID, updatemats(0).GUID))
  }
  
  def evalfun(sdata:Mat, user:Mat):FMat = {  
  	val preds = DDS(mm, user, sdata)
  	val dc = sdata.contents
  	val pc = preds.contents
  	max(opts.weps, pc, pc)
  	ln(pc, pc)
  	val sdat = sum(sdata,1)
  	val mms = sum(mm,2)
  	val suu = ln(mms ^* user)
  	if (traceMem) println("evalfun %d %d %d, %d %d %d, %d %f" format (sdata.GUID, user.GUID, preds.GUID, pc.GUID, sdat.GUID, mms.GUID, suu.GUID, GPUmem._1))
  	val vv = ((pc ddot dc) - (sdat ddot suu))/sum(sdat,2).dv
  	row(vv, math.exp(-vv))
  }
}

class NMFModel(opts:NMFModel.Options = new NMFModel.Options) extends FactorModel(opts) { 
  
  var mm:Mat = null
  var mdiag:Mat = null
  var udiag:Mat = null
  
  override def init(datasource:DataSource) = {
  	super.init(datasource)
  	updatemats = new Array[Mat](2)
    mm = modelmats(0)
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols)
    updatemats(0) = mm.zeros(mm.nrows, 1)
    udiag = mkdiag(opts.uprior*ones(opts.dim,1))
  	mdiag = mkdiag(opts.mprior*ones(opts.dim,1))
    if (opts.useGPU) {
      udiag = GMat(udiag)
      mdiag = GMat(mdiag)
    }
  }
  
  override def uupdate(sdata:Mat, user:Mat) = {
	  val modeldata = mm * sdata
  	val mmu = mm *^ mm + udiag
    for (i <- 0 until opts.uiter) {
    	val quot =  modeldata / (mmu * user)               
    	min(10.0f, max(0.1f, quot, quot), quot)
    	user ~ user *@ quot
    	max(opts.minuser, user, user)
    }
  }
   
  override def mupdate(sdata:Mat, user:Mat):Unit = {
    updatemats(0) <-- user *^ sdata
    val uu = user *^ user + mdiag * (1.0f*size(user,2)/opts.nusers)
    updatemats(1) <-- uu * mm    
  }
  
  override def evalfun(sdata:Mat, user:Mat):FMat = {

	  val modeldata =  mm * sdata
    val uu = user *^ user + mdiag * (1.0f*size(user,2)/opts.nusers)
    val mmm = mm *^ mm

    val ll0 =  sdata.contents ddot sdata.contents
    val ll1 =  modeldata ddot user
    val ll2 =  uu ddot mmm
//    println("ll %f %f %f" format (ll0, ll1, ll2))
    val v1  =              (-ll0 + 2*ll1 - ll2)/sdata.nnz
    val v2 =               -opts.uprior*(user ddot user)/sdata.nnz
    row(v1,v2)
  }
}



abstract class FactorModel(override val opts:FactorModel.Options) extends Model(opts) {
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    val data0 = mats(0)
    val m = size(data0, 1)
    val d = opts.dim
    val sdat = (sum(data0,2).t + 1.0f).asInstanceOf[FMat]
    val sp = sdat / sum(sdat)
    println("initial perplexity=%f" format (sp ddot ln(sp)) )
    
    val modelmat = rand(d,m) 
    modelmat ~ modelmat *@ sdat
    val msum = sum(modelmat, 2)
    modelmat ~ modelmat / msum
    modelmats = Array[Mat](1)
    modelmats(0) = if (opts.useGPU) GMat(modelmat) else modelmat
    datasource.reset
    
    if (mats.size > 1) {
      while (datasource.hasNext) {
        mats = datasource.next
        val dmat = mats(1)
        dmat.set(1.0f/d)
        datasource.putBack(mats,1)
      }
    }
  } 
  
  def reuseuser(a:Mat):Mat = {
    val out = a match {
      case aa:SMat => FMat.newOrCheckFMat(opts.dim, a.ncols, null, a.GUID, "reuseuser".##)
      case aa:GSMat => GMat.newOrCheckGMat(opts.dim, a.ncols, null, a.GUID, "reuseuser".##)
    }
    out.set(1f)
    out
  }
  
  def uupdate(data:Mat, user:Mat)
  
  def mupdate(data:Mat, user:Mat)
  
  def evalfun(data:Mat, user:Mat):FMat
  
  def doblock(gmats:Array[Mat], i:Long) = {
    val sdata = gmats(0)
    val user = if (gmats.length > 1) gmats(1) else reuseuser(gmats(0))
    uupdate(sdata, user)
    mupdate(sdata, user)
  }
  
  def evalblock(mats:Array[Mat]):FMat = {
    val sdata = gmats(0)
    val user = if (gmats.length > 1) gmats(1) else reuseuser(gmats(0))
    uupdate(sdata, user)
    evalfun(sdata, user)
  }
  
}

object NMFModel  {
  class Options extends FactorModel.Options {
    var NMFeps = 1e-9
    var uprior = 0.01f
    var mprior = 1e-4f
    var nusers = 100000
  }
} 

object LDAModel  {
  class Options extends FactorModel.Options {
    var LDAeps = 1e-9
    var exppsi = true
    var alpha = 0.1f
    putBack = -1
    useGPU = true
  }
}

object FactorModel { 
  class Options extends Model.Options { 
    var dim = 100
    var uiter = 1
    var weps = 1e-10f
    var minuser = 1e-8f
  }
} 


