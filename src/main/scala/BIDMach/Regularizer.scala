package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

abstract class Regularizer(opts:Regularizer.Options = new Regularizer.Options) { 
  val options = opts
  
  def compute(modelmats:Array[Mat], updatemats:Array[Mat], step:Float)
  
  def initregularizer = {
  }
}

class L1Regularizer(opts:Regularizer.Options = new Regularizer.Options) extends Regularizer(opts) { 
   def compute(modelmats:Array[Mat], updatemats:Array[Mat], step:Float) = {
     for (i <- 0 until modelmats.length) {
       updatemats(i) ~ updatemats(i) + (sign(modelmats(i)) * (-step*options.mprior)) 
     }
   }
}

class L2Regularizer(opts:Regularizer.Options = new Regularizer.Options) extends Regularizer(opts) { 
   def compute(modelmats:Array[Mat], updatemats:Array[Mat], step:Float) = {
  	 for (i <- 0 until modelmats.length) {
  		 updatemats(i) ~ updatemats(i) + (modelmats(i) * (-options.mprior * step))
  	 }
   }
}

object Regularizer {
  class Options {
    var mprior:FMat = 1e-7f }
}
