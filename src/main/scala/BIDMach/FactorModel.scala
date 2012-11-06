package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import Learner._


class NMFmodel(data0:SMat, opts:FactorModel.Options = new NMFmodel.Options) extends FactorModel(data0, opts) { 

  var modeldata:Mat = null
  var mmodelt:Mat = null
  var maxeps:Mat = null
  var mmodeltuser:Mat = null
  var idiag:Mat = null
  var quot:Mat = null
  var ud:Mat = null
  var uu:Mat = null
  var uum:Mat = null
  
  override  def initmodelf(data:Mat, user:Mat):Mat = {
  	val v = super.initmodelf(data, user)
  	idiag = mkdiag(options.uprior*ones(options.dim,1)):FMat
  	v
  }
  
  override def initupdate(sdata:Mat, model:Mat, user:Mat) = {
	  modeldata = recycleTry(modeldata, user)      ~ modelmat * sdata
  	mmodelt = recycleTry(mmodelt, d, d, model)   ~ modelmat xT modelmat
  	mmodelt                                      ~ mmodelt + idiag
  }
   
  override def uupdate(sdata:Mat, model:Mat, user:Mat):Unit = { 
    mmodeltuser = recycleTry(mmodeltuser, user)  ~ mmodelt * user
    quot = recycleTry(quot, user)                ~ modeldata /@ mmodeltuser
                                                 max(0.1f, quot, quot)
                                                 min(10.0f, quot, quot)
//    println("max quot=%f" format maxi(quot(?),1).dv)
	  user                                         ~ user *@ quot
//	  println("norm diff=%f" format norm(modeldata-mmodeltuser).dv)
  }
   
  override def mupdate(sdata:Mat, model:Mat, user:Mat, update:Mat) = {  
    ud = recycleTry(ud, modelmat)                ~ user xT sdata
    uu = recycleTry(uu, d, d, modelmat)          ~ user xT user
    uum = recycleTry(uum, modelmat)              ~ uu * modelmat
    update                                       ~ ud - uum
    val tmp0 = (sum(user,2).t * modelmat).t
    val diff = sum(sdata,2) - tmp0
    val v = -(diff dot diff)/sdata.nrows/sdata.ncols
//    println("norm diff=%f, v=%f" format (norm(diff).dv/math.sqrt(diff.length), v))
    v
  }
}

class LDAmodel(data0:SMat, opts:FactorModel.Options = new FactorModel.Options) extends FactorModel(data0, opts) { 

  override def uupdate(data:Mat, model:Mat, user:Mat):Unit = { 
    (data, model, user) match {
    case (sdata:SMat, fmodel:FMat, fuser:FMat) => {
    	val preds = DDS(fmodel, fuser, sdata)
    	val prat = sdata.ssMatOp(preds, (x:Float, y:Float) => x / math.max(options.weps, y))
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
  
  override def mupdate(data:Mat, model:Mat, user:Mat, update:Mat) = { 
  	(data, model, user, update) match {
  	case (sdata:SMat, fmodel:FMat, fuser:FMat, fupdate:FMat) => {
  		val preds = DDS(fmodel, fuser, sdata)
  		val prat = sdata.ssMatOp(preds, (x:Float, y:Float) => x / math.max(options.weps, y))
  		fupdate ~ (prat * fuser.t).t - sum(fuser,2) * ones(1,size(fuser,2))
  		val vdat = row(sdata.data)
  		val vpreds = row(preds.data)
  		mean(vdat *@ ln(max(options.weps, vpreds)) - gammaln(vdat)).dv
  	}
  	}
  }
}

abstract class FactorModel(data0:SMat, opts:FactorModel.Options) extends Model(data0, null, opts) {

  val options = opts
  var d = 0

  override def initmodel(data:Mat, user:Mat):Mat = initmodelf(data, user)
  
  def initmodelf(data:Mat, user:Mat):Mat = {
    val m = size(data, 1)
    val n =  size(data, 2)
    d = options.dim
    modelmat = 0.1f*normrnd(0,1,d,m)  
    val out = if (user.asInstanceOf[AnyRef] == null) {
      0.1f*normrnd(0,1,d,n)
    } else{
      user
    }
    updatemat = modelmat.zeros(d, m)
    out
  } 
  
  def initupdate(data:Mat, model:Mat, user:Mat)  = {}
  
  def uupdate(data:Mat, model:Mat, user:Mat):Unit
  
  def mupdate(sdata:Mat, model:Mat, user:Mat, update:Mat):Double
  
  override def gradfun(data:Mat, user:Mat):Double = { 
	  var i = 0
	  initupdate(data, modelmat, user)
	  while (i < options.uiter) { 		    
	  	uupdate(data, modelmat, user)
	  	i += 1
	  }
	  val v = mupdate(data, modelmat, user, updatemat)
	  v
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
  }
}
