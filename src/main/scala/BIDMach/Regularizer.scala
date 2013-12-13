package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

abstract class Regularizer(val opts:Regularizer.Opts = new Regularizer.Options) { 
  val options = opts
  var modelmats:Array[Mat] = null
  var updatemats:Array[Mat] = null
  
  def compute(step:Float)
  
  def init(model:Model) = {
    modelmats = model.modelmats
    updatemats = model.updatemats
  }
}

class L1Regularizer(override val opts:Regularizer.Options = new Regularizer.Options) extends Regularizer(opts) { 
   def compute(step:Float) = {
     for (i <- 0 until modelmats.length) {
       updatemats(i) ~ updatemats(i) + (sign(modelmats(i)) * (-step*options.mprior)) 
     }
   }
}

class L2Regularizer(override val opts:Regularizer.Options = new Regularizer.Options) extends Regularizer(opts) { 
   def compute(step:Float) = {
  	 for (i <- 0 until modelmats.length) {
  		 updatemats(i) ~ updatemats(i) + (modelmats(i) * (-options.mprior * step))
  	 }
   }
}

object Regularizer {
	trait Opts {
		var mprior:FMat = 1e-7f 
	}
	
	class Options extends Opts {}
}
