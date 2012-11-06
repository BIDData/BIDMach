package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

abstract class Regularizer(val model:Model, opts:Regularizer.Options = new Regularizer.Options) { 
  val options = opts
  def compute(step:Int)
  val modelmat = model.modelmat
  val updatemat = model.updatemat
  var tmp0 = modelmat.zeros(size(modelmat,1), size(modelmat,2))
  var tmp1 = modelmat.zeros(size(modelmat,1), size(modelmat,2))
}

class L1Regularizer(model:Model, opts:Regularizer.Options = new Regularizer.Options) extends Regularizer(model, opts) { 
   def compute(step:Int) = {
     sign(modelmat, tmp0)
     updatemat ~ updatemat + (tmp1 ~  tmp0 * (-step * options.mprior))
   }
}

class L2Regularizer(model:Model, opts:Regularizer.Options = new Regularizer.Options) extends Regularizer(model, opts) { 
   def compute(step:Int) = {
     updatemat ~ updatemat + (tmp1 ~ modelmat * (-options.mprior * step))
   }
}

object Regularizer {
  class Options {
    var mprior:Float = 1e-7f
  }
}
