package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._


class LDAModel(override val opts:LDAModel.Options = new LDAModel.Options) extends FactorModel(opts) { 
  var mm:Mat = null
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    updatemats = new Array[Mat](2)
    mm = modelmats(0)
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols)
    updatemats(0) = mm.zeros(mm.nrows, 1)
  }
  
  def uupdate(sdata:Mat, user:Mat):Unit = {
	  for (i <- 0 until opts.uiter) {
	  	val preds = DDS(mm, user, sdata)
	  	max(opts.weps, preds, preds)
	  	val prat = sdata / preds
	  	val unew = user *@ (mm * prat) + opts.alpha
	  	if (opts.exppsi) exppsi(unew, unew)
	  	user <-- unew                                                     
	  }
  }
  
  def mupdate(sdata:Mat, user:Mat):Unit = {
  	val preds = DDS(mm, user, sdata)
  	max(opts.weps, preds, preds)
  	val prat = sdata / preds
  	updatemats(0) <-- user xT prat           
  	updatemats(1) <-- sum(user,2)
  }
  
  def evalfun(sdata:Mat, user:Mat):FMat = {  
  	val preds = DDS(mm, user, sdata)
  	max(opts.weps, preds, preds)
  	val ll = ln(preds.contents)
  	val sdat = sum(sdata,1)
  	val su1 = ln(sum(user,1))
//  	val nvv = sum(sdat,2).dv
//  	println("vals %f, %f, %f, %f, %f" format (nvv, sum(data.contents,1).dv, sum(ll,1).dv/nvv, (ll dot data.contents)/nvv,(sdat dot su1)/nvv))
  	row(((ll ddot sdata.contents) - (sdat ddot su1))/sum(sdat,2).dv,0)
  }
}

class NMFModel(opts:FactorModel.Options = new NMFModel.Options) extends FactorModel(opts) { 
  
  var mm:Mat = null
  var mdiag:Mat = null
  var udiag:Mat = null
  var datasource:DataSource = null
  
  override def init(datasourcex:DataSource) = {
    datasource = datasourcex
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
  	val mmu = mm xT mm + udiag
    for (i <- 0 until opts.uiter) {
    	val mmtu = mmu * user
    	val quot =  mm / mmtu                
    	min(10.0f, max(0.1f, quot, quot), quot)
    	user ~ user *@ quot
    	max(opts.minuser, user, user)
    }
  }
   
  override def mupdate(sdata:Mat, user:Mat):Unit = {
    updatemats(0) <-- user xT sdata
    val uu = user xT user + mdiag * (1.0f*size(user,2)/datasource.opts.nusers)
    updatemats(1) <-- uu * mm    
  }
  
  override def evalfun(sdata:Mat, user:Mat):FMat = {

	  val modeldata =  mm * sdata
    val uu = user xT user + mdiag * (1.0f*size(user,2)/datasource.opts.nusers)
    val mmm = mm xT mm

    val ll0 =  sdata.contents ddot sdata.contents
    val ll1 =  modeldata ddot user
    val ll2 =  uu ddot mmm
//    println("ll %f %f %f" format (ll0, ll1, ll2))
    val v1  =              (-ll0 + 2*ll1 - ll2)/sdata.nnz
    val v2 =               -opts.uprior*(user ddot user)/sdata.nnz
    row(v1,v2)
  }
}



abstract class FactorModel(val opts:FactorModel.Options) extends Model {
  
  var mats:Array[Mat] = null
  
  override def init(datasource:DataSource) = {
    mats = datasource.next
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
        val dmat = mats(1).asInstanceOf[FMat]
        dmat(?) = 1.0/d
      }
    }

    if (opts.useGPU) {
      modelmats(0) = GMat(modelmat)
    }
  } 
  
  def uupdate(data:Mat, user:Mat)
  
  def mupdate(data:Mat, user:Mat)
  
  def evalfun(data:Mat, user:Mat):FMat
  
  def doblock(mats:Array[Mat], i:Long) = {
    val sdata = mats(0)
    val user = mats(1)
    uupdate(sdata, user)
    mupdate(sdata, user)
  }
  
  def evalblock(mats:Array[Mat]):FMat = {
    val sdata = mats(0)
    val user = mats(1)
    uupdate(sdata, user)
    evalfun(sdata, user)
  }
  
}

object NMFModel  {
  class Options extends FactorModel.Options {
    var NMFeps = 1e-9
  }
} 

object LDAModel  {
  class Options extends FactorModel.Options {
    var LDAeps = 1e-9
    var exppsi = true
    var alpha = 1.0f
  }
}

object FactorModel { 
  class Options extends Model.Options { 
    var dim = 10
    var uiter = 10
    var weps = 1e-10f
    var uprior = 0.01f
    var mprior = 1e-4f
    var minuser = 1e-8f
  }
} 


