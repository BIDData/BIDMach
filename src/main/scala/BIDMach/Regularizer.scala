package BIDMach
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._

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

class L1Regularizer(override val opts:Regularizer.Opts = new Regularizer.Options) extends Regularizer(opts) { 
   def compute(step:Float) = {
     for (i <- 0 until modelmats.length) {
       val v = if (opts.regweight.length == 1) - opts.regweight(0) else - opts.regweight(i)
       if (v != 0) {
         updatemats(i) ~ updatemats(i) + (sign(modelmats(i)) * v) 
       }
     }
   }
}

class L2Regularizer(override val opts:Regularizer.Opts = new Regularizer.Options) extends Regularizer(opts) { 
   def compute(step:Float) = {
  	 for (i <- 0 until modelmats.length) {
  	   val v = if (opts.regweight.length == 1) - opts.regweight(0) else - opts.regweight(i)
  	   if (v != 0) {
  	     updatemats(i) ~ updatemats(i) + (modelmats(i) * v)
  	   }
  	 }
   }
}

object Regularizer {
	trait Opts {
		var regweight:FMat = 1e-7f 
	}
	
	class Options extends Opts {}
}
