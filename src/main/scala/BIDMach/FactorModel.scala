package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import Learner._

class LDAmodel(data0:SMat, opts:FactorModel.Options = new FactorModel.Options) extends FactorModel(data0, opts) { 

  def uupdate(sdata:SMat, model:FMat, user:FMat):Unit = { 
	val preds = DDS(model, user, sdata)
  	val prat = sdata.ssMatOp(preds, (x:Float, y:Float) => x / math.max(options.weps, y))
  	val prod = model * prat
    var i = 0;  while (i < prod.ncols) { 
      var j = 0;  while (j < prod.nrows) { 
        val ji = j + i*prod.nrows
        val v = user.data(ji)*prod.data(ji) + options.prior(j)
        user.data(ji) = 
          if (v > 1) {
        	v - 0.5f
          } else {
        	0.5f * v * v
          }
        j += 1}
      i += 1}
  }
  
  def mupdate(sdata:SMat, model:FMat, user:FMat, update:FMat) = {  
	val preds = DDS(model, user, sdata)
	val prat = sdata.ssMatOp(preds, (x:Float, y:Float) => x / math.max(options.weps, y))
	update ~ (prat * user.t).t - sum(user,2) * ones(1,size(user,2))
	val vdat = row(sdata.data)
    val vpreds = row(preds.data)
    mean(vdat *@ ln(max(options.weps, vpreds)) - gammaln(vdat)).dv
  }
}

abstract class FactorModel(data0:SMat, opts:FactorModel.Options) extends Model(data0, null, opts) {

  var modelmat:Mat = null
  var updatemat:Mat = null
  var usermat:Mat = null
  val options = opts

  override def initmodel(data:Mat, user:Mat):Mat = initmodelf(data, user)
  
  def initmodelf(data:Mat, user:Mat):Mat = {
    val m = size(data, 1)
    val n =  size(data, 2)
    val d = options.dim
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
  
  def initupdate(data:SMat, prod:FMat) = {}
  
  def uupdate(data:SMat, prod:FMat, user:FMat):Unit
  
  def mupdate(sdata:SMat, fmodel:FMat, user:FMat, update:FMat):Double
  
  override def gradfun(idata:Mat, iuser:Mat):Double = { 
    (idata, modelmat, iuser, updatemat) match {
  	  case (sdata:SMat, fmodel:FMat, fuser:FMat, fupdate:FMat) => {
        var i = 0
        initupdate(sdata, fmodel)
        while (i < options.niter) { 
          uupdate(sdata, fmodel, fuser)
        }
        mupdate(sdata, fmodel, fuser, fupdate)
  	  }
    }
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
