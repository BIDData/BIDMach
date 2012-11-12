package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

abstract class Regularizer(val model:Model, opts:Regularizer.Options = new Regularizer.Options) { 
  val options = opts
  var modelmat:Mat = null
  var updatemat:Mat = null
  var tmp0:Mat = null
  var tmp1:Mat = null 
  
  def compute(step:Float)
  
  def initregularizer = {
    modelmat = model.modelmat
    updatemat = model.updatemat
    tmp0 = modelmat.zeros(size(modelmat,1), size(modelmat,2))
    tmp1 = modelmat.zeros(size(modelmat,1), size(modelmat,2))
  }
}

class L1Regularizer(model:Model, opts:Regularizer.Options = new Regularizer.Options) extends Regularizer(model, opts) { 
   def compute(step:Float) = {
     sign(modelmat, tmp0)
     updatemat ~ updatemat + (tmp1 ~  tmp0 * (-step * options.mprior))
   }
}

class L2Regularizer(model:Model, opts:Regularizer.Options = new Regularizer.Options) extends Regularizer(model, opts) { 
   def compute(step:Float) = {
     updatemat ~ updatemat + (tmp1 ~ modelmat * (-options.mprior * step))
   }
}

class L2MultRegularizer(model:Model, opts:Regularizer.Options = new Regularizer.Options) extends Regularizer(model, opts) { 
   def compute(step:Float) = {
     val updateDenom = model.asInstanceOf[FactorModel].updateDenom
     updateDenom ~ updateDenom + (tmp1 ~ updateDenom * (step*options.mprior))
   }
}

object Regularizer {
  class Options {
    var mprior:Float = 1e-7f
  }
}
