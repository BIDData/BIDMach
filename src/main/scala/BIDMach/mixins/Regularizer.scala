package BIDMach.mixins
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._

class L1Regularizer(override val opts:L1Regularizer.Opts = new L1Regularizer.Options) extends Mixin(opts) { 
   def compute(mats:Array[Mat], step:Float) = {
     for (i <- 0 until opts.r1nmats) {
       val v = if (opts.reg1weight.ncols == 1) - opts.reg1weight else - opts.reg1weight(?,i)
       updatemats(i) ~ updatemats(i) + (sign(modelmats(i)) ∘  v) 
     }
   }
   
   def score(mats:Array[Mat], step:Float):FMat = {
     val sc = zeros(opts.r1nmats,1)
     for (i <- 0 until opts.r1nmats) {
       sc(i) = mean(sum(abs(modelmats(i)),2)).dv
     }
     sc
   }
}

class L2Regularizer(override val opts:L2Regularizer.Opts = new L2Regularizer.Options) extends Mixin(opts) { 
   def compute(mats:Array[Mat], step:Float) = {
     for (i <- 0 until opts.r2nmats) {
       val v = if (opts.reg2weight.ncols == 1) - opts.reg2weight else - opts.reg2weight(?,i)
       updatemats(i) ~ updatemats(i) + (modelmats(i) ∘  v)
     }
   }
   
   def score(mats:Array[Mat], step:Float):FMat = {
     val sc = zeros(opts.r2nmats,1)
     for (i <- 0 until opts.r2nmats) {
       sc(i) = mean(modelmats(i) dotr modelmats(i)).dv
     }
     sc
   }
}


object L1Regularizer {
  trait Opts extends Mixin.Opts {
    var reg1weight:FMat = 1e-7f
    var r1nmats:Int = 1
  }
  
  class Options extends Opts {}
}

object L2Regularizer {
    trait Opts extends Mixin.Opts {
        var reg2weight:FMat = 1e-7f
        var r2nmats:Int = 1
    }
    
    class Options extends Opts {}
}


