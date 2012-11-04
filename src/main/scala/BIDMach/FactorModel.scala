package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import Learner._

class LDAmodel(data0:SMat, opts:FactorModel.Options = new FactorModel.Options) extends FactorModel(data0, opts) { 

  def lossfn(preds:SMat, data:SMat):SMat = data.ssMatOp(preds, (x:Float, y:Float) => x / math.max(options.weps, y))

  def ldaf(v:Float):Float = { 
//    exp(psi(v))
    if (v > 1) {
      v - 0.5f
    } else {
      0.5f * v * v
    }
  }

  def uupdate(prod:FMat, user:FMat):Unit = { 
    var i = 0;  while (i < prod.ncols) { 
      var j = 0;  while (j < prod.nrows) { 
        val ji = j + i*prod.nrows
        user.data(ji) = ldaf(user.data(ji)*prod.data(ji) + options.prior(j))
        j += 1}
      i += 1}
  }

  def likelihood(data:SMat, preds:SMat):Double = { 
    val vdat = row(data.data)
    val vpreds = row(preds.data)
    mean(vdat *@ ln(max(options.weps, vpreds)) - gammaln(vdat)).dv      
  }
}


abstract class FactorModel(data0:SMat, opts:FactorModel.Options) extends Model(data0, null, opts) {

  var modelmat:Mat = null
  var updatemat:Mat = null
  var usermat:Mat = null
  val options = opts

  def lossfn(preds:SMat, data:SMat):SMat

  def uupdate(prod:FMat, user:FMat):Unit

  def likelihood(data:SMat, preds:SMat):Double
  
  override def initmodel(data:Mat, user:Mat):Mat = initmodelf(data, user)
  
  def initmodelf(data:Mat, user:Mat):Mat = {
    val m = size(data, 1)
    val n =  size(data, 2)
    val d = options.dim
    modelmat = 0.1f*normrnd(0,1,d,m)
    val out = 0.1f*normrnd(0,1,d,n)
    updatemat = modelmat.zeros(d, m)
    out
  }
  
  usermat = initmodel(data0, null)
  
  override def gradfun(idata:Mat, iuser:Mat):Double = { 
    (idata, modelmat, iuser, updatemat) match {
  	  case (sdata:SMat, fmodel:FMat, fuser:FMat, fupdate:FMat) => {
        var i = 0
        var fpreds:SMat = null
        var loss:SMat = null
        while (i < options.niter) { 
          fpreds = DDS(fmodel, fuser, sdata)
  	  	  loss = lossfn(fpreds, sdata)
  	  	  val prod = fmodel * loss
          uupdate(prod, fuser)
        }
        fupdate ~ loss * fuser.t
  	  	likelihood(sdata, fpreds)
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
