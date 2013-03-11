package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

class LinearPredictor {
    
  def derivfn(targ:Mat, pred:Mat, lls:Mat):Mat =  linearDeriv(targ, pred, lls)

  def linearDeriv(targ:Mat, pred:Mat, lls:Mat):Mat= {
    val diff = targ - pred  
    lls  ~ lls - diff ∙∙ diff
    2 * diff
  }  
}

class LogisticPredictor {
  
  def derivfn(targ:Mat, pred:Mat, lls:Mat):Mat =  logisticDeriv(targ, pred, lls)

  def logisticDeriv(targ:Mat, pred:Mat, lls:Mat):Mat= {
    val mpred = max(-40, min(40, pred))
    val tp = (2 * targ - 1)
    val ep = exp(tp ∘ mpred)
    val expp1 = ep + 1
    lls  ~ lls - sum(1 / expp1, 2)
    tp ∘ ep / expp1 / expp1
  }  
}

abstract class RegressionModel(opts:RegressionModel.Options) 
  extends Model {
  
  val options = opts   
  var lls:Mat = null

  override def initmodel:Unit = {
    modelmats = new Array[Mat](1)
    modelmats(0) = if (opts.useGPU) GMat(opts.nrows, opts.nmodels) else FMat(opts.nrows, opts.nmodels)
    lls = if (opts.useGPU) gzeros(opts.nmodels, 1) else zeros(opts.nmodels, 1)
  }
  
  def derivfn(targ:Mat, pred:Mat, lls:Mat):Mat 
  
  override def doblock(datamats:Array[Mat], updatemats:Array[Mat]):Unit = {
    val sdata = datamats(0)
    val target = datamats(1)
    val model = modelmats(0)
    val mupdate = updatemats(0)
    
    val mvals = model * sdata
    val dd = derivfn(target, mvals, lls)    
    mupdate ~ mupdate + dd *^ sdata
  }
 
}

object RegressionModel {
  class Options extends Model.Options {
    var nrows = 0
    var nmodels = 0
    var transpose:Boolean = false
  }
}
