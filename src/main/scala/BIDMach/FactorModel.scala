package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._


class LDAModel(opts:LDAModel.Options = new LDAModel.Options) extends FactorModel(opts) { 
  
  override def init() = {
    super.init()
    updatemats = new Array[Mat](2)
    val mm = modelmats(0)
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols)
    updatemats(0) = mm.zeros(mm.nrows, 1)
  }
  
  def uupdate(sdata:Mat, user:Mat):Unit = {
    val mm = modelmats(0)
    for (i <- 0 until opts.uiter) {
    	val preds = DDS(mm, user, sdata)
    	max(opts.weps, preds, preds)
    	val prat = sdata / preds
    	val unew = user *@ (mm * prat) + opts.alpha
    	user <-- unew                                                      // Or apply LDA function exp(Psi(*))
    }
  }
  
  def mupdate(sdata:Mat, user:Mat):Unit = {
  	val mm = modelmats(0)
  	updatemats(0) = (user xT sdata) *@ mm
  }
  
  def evalfun(data:Mat, user:Mat):(Double, Double) = {  
  	val preds = DDS(modelmats(0), user, data)
  	max(opts.weps, preds, preds)
  	val ll = ln(preds.contents)
  	val sdat = sum(data,1)
  	val su1 = ln(sum(user,1))
//  	val nvv = sum(sdat,2).dv
//  	println("vals %f, %f, %f, %f, %f" format (nvv, sum(data.contents,1).dv, sum(ll,1).dv/nvv, (ll dot data.contents)/nvv,(sdat dot su1)/nvv))
  	(((ll ddot data.contents) - (sdat ddot su1))/sum(sdat,2).dv,0)
  }
}
/*
class NMFmodel(opts:FactorModel.Options = new NMFmodel.Options) extends FactorModel(opts) { 
  
  def make(opts:Model.Options) = new NMFmodel(opts.asInstanceOf[NMFmodel.Options])

  var modeldata    = blank
  var mm           = blank
  var maxeps       = blank
  var mmodeltuser  = blank
  var udiag        = blank
  var mdiag        = blank
  var smdiag       = blank
  var quot         = blank
  var uu           = blank
  var diff         = blank
  
  override  def initmodel(data0:Mat, user0:Mat, datatest0:Mat, usertest0:Mat):(Mat, Mat) = {
  	val v = super.initmodel(data0, user0, datatest0, usertest0)
  	udiag = mkdiag(options.uprior*ones(options.dim,1)):FMat
  	mdiag = mkdiag(options.mprior*ones(options.dim,1)):FMat
    if (options.useGPU) {
      udiag = GMat(udiag)
      mdiag = GMat(mdiag)
      val k = options.startBlock
      idata = GSMat(size(data0,1), k, k * options.nzPerColumn)
      iuser = GMat(options.dim, k)
    }
  	v
  }
  
  override def initupdate(sdata:Mat, user:Mat) = {
    val d = options.dim
	  modeldata = modeldata  ~ modelmat * sdata
  	mm        = mm         ~ modelmat xT modelmat
  	mm                     ~ mm + udiag
  }
   
  override def uupdate(sdata:Mat, user:Mat):Unit = { 
    mmodeltuser = mmodeltuser  ~ mm * user
    quot        = quot         ~ modeldata / mmodeltuser                
    quot        =                min(10.0f, max(0.1f, quot, quot), quot)
	  user                       ~ user *@ quot
//	  println("norm user %f" format norm(user))
                                 max(options.minuser, user, user)
  }
   
  override def mupdate(sdata:Mat, user:Mat):Unit = {
    val d = options.dim
    updatemat = updatemat ~ user xT sdata
//    updatemat       ~ updatemat *@ modelmat
    uu =     uu     ~ user xT user
    smdiag = smdiag ~ mdiag * (1.0f*size(user,2)/nusers)
    uu              ~ uu + smdiag
    updateDenom = updateDenom ~ uu * modelmat    
  }
  
  override def eval(data0:Mat, user0:Mat):(Double, Double) = {
    modelmat match {
	    case m:FMat => {
	      idata = data0
	      iuser = user0
	    }
	    case m:GMat => {
	    	idata = GSMat.fromSMat(data0.asInstanceOf[SMat], idata.asInstanceOf[GSMat])
	    	iuser = GMat.fromFMat(user0.asInstanceOf[FMat], iuser.asInstanceOf[GMat])
	    }
	  }
	  modeldata = modeldata ~ modelmat * idata
    uu        = uu        ~ iuser xT iuser 
    smdiag    = smdiag    ~ mdiag * (1.0f*size(iuser,2)/nusers)
                uu        ~ uu + smdiag
    mm        = mm        ~ modelmat xT modelmat

    val ll0 =               idata.contents ddot idata.contents
    val ll1 =               modeldata ddot iuser
    val ll2 =               uu ddot mm
//    println("ll %f %f %f" format (ll0, ll1, ll2))
    val v1  =              (-ll0 + 2*ll1 - ll2)/idata.nnz
    val v2 =               -options.uprior*(iuser ddot iuser)/idata.nnz
    (v1,v2)
  }
}

} */

abstract class FactorModel(opts:FactorModel.Options) extends Model {
  
  var mats:Array[Mat] = null
  
  override def init() = {
    mats = datasource.next
    val data0 = mats(0)
    val m = size(data0, 1)
    val d = opts.dim
    val sdat = (sum(data0,2).t + 1.0f).asInstanceOf[FMat]
    val sp = sdat / sum(sdat)
    println("initial perplexity=%f" format (sp dot ln(sp)) )
    
    val modelmat = rand(d,m) 
    modelmat ~ modelmat *@ sdat
    val msum = sum(modelmat, 2)
    modelmat ~ modelmat / msum
    modelmats = Array[Mat](1)
    modelmats(0) = modelmat
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
  
  def doblock(i:Int) = {
    val data = mats(0)
    val user = mats(1)
    uupdate(data, user)
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
    var alpha = 1.0f
  }
}

object FactorModel { 
  class Options extends Model.Options { 
    var dim = 20
    var uiter = 10
    var weps = 1e-10f
    var uprior = 0.01f
    var mprior = 1e-4f
    var minuser = 1e-8f
  }
} 
