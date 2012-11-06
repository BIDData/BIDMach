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
  var quot:Mat = null
  var ud:Mat = null
  var uu:Mat = null
  var uum:Mat = null
  
  override def initupdate(sdata:Mat, model:Mat, user:Mat) = {
	  modeldata = recycleTry(modeldata, user)      ~ model * sdata
  	mmodelt = recycleTry(mmodelt, d, d, model)   ~ model * model.t    
  }
   
  override def uupdate(sdata:Mat, model:Mat, user:Mat):Unit = { 
    val NMFeps = options.asInstanceOf[NMFmodel.Options].NMFeps
    maxeps = recycleTry(maxeps, mmodelt)
    maxeps =                                      max(NMFeps, mmodelt, maxeps):Mat
    mmodeltuser = recycleTry(mmodeltuser, user)  ~ maxeps * user
    quot = recycleTry(quot, user)                ~ modeldata /@ mmodeltuser
	  user                                        ~ user *@ quot
  }
   
  override def mupdate(sdata:Mat, model:Mat, user:Mat, update:Mat) = {  
    ud = recycleTry(ud, model)                   ~ user xT sdata
    uu = recycleTry(uu, d, d, model)             ~ user * user.t
    uum = recycleTry(uum, model)                 ~ uu * model
    update                                      ~ ud - uum
    val diff = sum(sdata,2) - (sum(user,2).t * model).t
    mean(diff *@ diff).dv
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
    			val v = fuser.data(ji)*prod.data(ji) + options.prior(j)
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

  var modelmat:Mat = null
  var updatemat:Mat = null
  var usermat:Mat = null
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
  
  usermat = initmodel(data0, null) 
  
  def initupdate(data:Mat, model:Mat, user:Mat)  = {}
  
  def uupdate(data:Mat, model:Mat, user:Mat):Unit
  
  def mupdate(sdata:Mat, model:Mat, user:Mat, update:Mat):Double
  
  override def gradfun(data:Mat, user:Mat):Double = { 
	  var i = 0
	  initupdate(data, modelmat, user)
	  while (i < options.niter) { 
	  	uupdate(data, modelmat, user)
	  }
	  mupdate(data, modelmat, user, updatemat)
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
    var niter = 10
    var weps = 1e-10f
    var prior:FMat = null
  }
}
