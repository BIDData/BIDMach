package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

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

class LDAmodel(opts:FactorModel.Options = new FactorModel.Options) extends FactorModel(opts) { 
  
  def make(opts:Model.Options) = new LDAmodel(opts.asInstanceOf[FactorModel.Options])

  var preds = blank
  var prat = blank
  var prod = blank
  var mones = blank
  var sdat = blank
  
  
  override  def initmodel(data0:Mat, user0:Mat, datatest0:Mat, usertest0:Mat):(Mat, Mat) = {
  	val v = super.initmodel(data0, user0, datatest0, usertest0)
  	mones = ones(1, size(modelmat,2))
  	v
  }
  
  override def uupdate(data:Mat, user:Mat):Unit = uupdate2(data, user)
  
  def uupdate1(data:Mat, user:Mat):Unit = { 
    (data, modelmat, user) match {
    case (sdata:SMat, fmodel:FMat, fuser:FMat) => {
    	preds = DDS(fmodel, fuser, sdata, preds)
    	val prat = sdata.ssMatOp(preds.asInstanceOf[SMat], (x:Float, y:Float) => x / math.max(options.weps, y), null)
    	val prod = fmodel * prat
    	var i = 0;  while (i < prod.ncols) { 
    		var j = 0;  while (j < prod.nrows) { 
    			val ji = j + i*prod.nrows
    			val v = fuser.data(ji)*prod.data(ji) + options.uprior
    			fuser.data(ji) = 
    				if (v > 1) {
    					v - 0.5f
    				} else {
    					0.5f * v * v
    				}
    			j += 1}
    		i += 1}
      }
    }
  }
  
  def uupdate2(data:Mat, user:Mat):Unit = {
//  	println("norm model=%f, user=%f" format (norm(modelmat),norm(user)))
  	preds = DDS(modelmat, user, data, preds)
  	max(options.weps, preds, preds)
  	prat = prat ~ data / preds

  	prod = prod ~ modelmat * prat
  	       user ~ prod *@ user
  	       user ~ user + options.uprior
  	       exppsi(user, user)
  }
  
  var ll = blank
  var glvdat = blank
  var su1 = blank
  var su2 = blank
  
  override def mupdate(data:Mat, user:Mat):Unit = { 
  	preds = DDS(modelmat, user, data, preds)
  	max(options.weps, preds, preds)
  	prat      = prat        ~ data / preds
  	updatemat = updatemat   ~ user xT prat;
  	su2       =               sum(user,2,su2)
  	updateDenom =             su2
  }
  
  override def eval(data:Mat, user:Mat):(Double, Double) = {  
  	preds = DDS(modelmat, user, data, preds)
  	max(options.weps, preds, preds)
  	ll = ln(preds.contents, ll)
  	sdat = sum(data,1,sdat)
  	su1 = sum(user,1,su1)
  	su1 = ln(su1, su1)
//  	val nvv = sum(sdat,2).dv
//  	println("vals %f, %f, %f, %f, %f" format (nvv, sum(data.contents,1).dv, sum(ll,1).dv/nvv, (ll dot data.contents)/nvv,(sdat dot su1)/nvv))
  	(((ll ddot data.contents) - (sdat ddot su1))/sum(sdat,2).dv,0)
  }
}

abstract class FactorModel(opts:FactorModel.Options) extends Model {

  override val options = opts
  var idata = blank
  var iuser = blank
  var usertest = blank
  var updateDenom = blank
  var msum = blank
  
  override def initmodel(data0:Mat, user0:Mat, datatest0:Mat, usertest0:Mat):(Mat, Mat) = {
    val m = size(data0, 1)
    nusers = size(data0, 2)
    val nt = size(datatest0, 2)
    val d = options.dim
    val sdat = (sum(data0,2).t + 1.0f).asInstanceOf[FMat]
    val sp = sdat / sum(sdat)
    println("initial perplexity=%f" format (sp dot ln(sp)) )
    modelmat = rand(d,m)  
    modelmat ~ modelmat *@ sdat
    msum = sum(modelmat, 2, msum)
    modelmat ~ modelmat / msum
    val target = if (user0.asInstanceOf[AnyRef] == null) {
      rand(d,nusers)
    } else{
      user0
    }
    val testtarg = if (usertest0.asInstanceOf[AnyRef] == null) {
      rand(d,nt)
    } else{
      usertest0
    }
    if (options.useGPU) {
      modelmat = GMat(modelmat.asInstanceOf[FMat])
    }
    (target, testtarg)
  } 
  
  def initupdate(data:Mat, user:Mat) = {}
  
  def uupdate(data:Mat, user:Mat):Unit
  
  def mupdate(sdata:Mat, user:Mat):Unit
  
  override def gradfun(data0:Mat, user0:Mat) = { 
	  var i = 0
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
	  initupdate(idata, iuser)
	  while (i < options.uiter) { 		    
	  	uupdate(idata, iuser)
	  	i += 1
	  }
	  (iuser, user0) match {
	  case (guser:GMat, fuser0:FMat) => {
      fuser0 <-- guser
    }
	  case _ => {}
	  }
	  mupdate(idata, iuser)
  }
}

object NMFmodel  {
  class Options extends FactorModel.Options {
    var NMFeps = 1e-9
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
} */
