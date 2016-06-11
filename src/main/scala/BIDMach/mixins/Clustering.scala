package BIDMach.mixins
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._

// Minimize the pairwise cosine of all model vectors
class CosineSim(override val opts:CosineSim.Opts = new CosineSim.Options) extends Mixin(opts) { 
   def compute(mats:Array[Mat], step:Float) = {
     for (i <- 0 until opts.cosnmats) {
       val v = if (opts.cosweight.length == 1) - opts.cosweight(0) else - opts.cosweight(i)
       if (v != 0) {
         val delta = if (opts.cosorthog) {
           val normalize = max(opts.coseps, sum(modelmats(i), 2))
           val nmodel = modelmats(i) / normalize
           val tmp = mean(nmodel)
           tmp - mean(tmp)                     // Orthogonalize to the constraint Sum pi = 1
         } else {
           mean(modelmats(i))
         }
         updatemats(i) ~ updatemats(i) + (delta * v)
       }
     }
   }
   
   def score(mats:Array[Mat], step:Float):FMat = {
     val sc = zeros(opts.cosnmats,1)
     for (i <- 0 until opts.cosnmats) {
       val mv = if (opts.cosorthog) {
         val normalize = max(opts.coseps, sum(modelmats(i), 2))
         mean(modelmats(i) / normalize)
       } else {
         mean(modelmats(i))
       }
       sc(i) = (mv dotr mv).dv
     }
     sc
   }
}

// Minimize the within-cluster perplexity
class Perplexity(override val opts:Perplexity.Opts = new Perplexity.Options) extends Mixin(opts) { 
   def compute(mats:Array[Mat], step:Float) = {
     for (i <- 0 until opts.perpnmats) {
       val v = if (opts.perpweight.length == 1) opts.perpweight(0) else opts.perpweight(i)
       if (v != 0) {
         val delta = if (opts.perporthog) {
           val normalize = max(opts.perpeps, sum(modelmats(i), 2))
           val nmodel = modelmats(i) / normalize
           max(opts.perpeps, nmodel, nmodel)
           ln(nmodel, nmodel)
           nmodel ~ nmodel - mean(nmodel, 2)  // Orthogonalize to the constraint Sum pi = 1
           nmodel
         } else {
           val nmodel = max(opts.perpeps, modelmats(i))
           ln(nmodel, nmodel)
           nmodel
         }                     
         updatemats(i) ~ updatemats(i) + (delta * v)
       }
     }
   }
   
   def score(mats:Array[Mat], step:Float):FMat = {
     val sc = zeros(opts.perpnmats,1)
     for (i <- 0 until opts.perpnmats) {
       val nmodel = if (opts.perporthog) {
         val normalize = max(opts.perpeps, sum(modelmats(i), 2))
         modelmats(i) / normalize
       } else {
         modelmats(i) + 0
       }
       max(opts.perpeps, nmodel, nmodel)       
       sc(i) = - mean(nmodel dotr ln(nmodel)).dv
     }
     sc
   }
}

// Minimize the non-top weights
class Top(override val opts:Top.Opts = new Top.Options) extends Mixin(opts) { 
   def compute(mats:Array[Mat], step:Float) = {
     for (i <- 0 until opts.topnmats) {
       val v = if (opts.topweight.length == 1) opts.topweight(0) else opts.topweight(i)
       if (v != 0) {
         val nmodel = modelmats(i) / max(opts.topeps, sum(modelmats(i), 2))
         val mask = nmodel < opts.topthreshold
         updatemats(i) ~ updatemats(i) + (sign(modelmats(i)) *@ mask * v)
       }
     }
   }
   
   def score(mats:Array[Mat], step:Float):FMat = {
     val sc = zeros(opts.topnmats,1)
     for (i <- 0 until opts.topnmats) {
       val nmodel = if (opts.toporthog) {
         modelmats(i) / max(opts.topeps, sum(modelmats(i), 2))
       } else {
         modelmats(i) + 0
       }
       val mask = nmodel < opts.topthreshold
       sc(i) = mean(sum(abs(modelmats(i) *@ mask),2)).dv
     }
     sc
   }
}


object CosineSim {
    trait Opts extends Mixin.Opts {
        var cosweight:FMat = 1e-7f 
        var coseps:Float = 1e-6f
        var cosorthog:Boolean = true
        var cosnmats:Int = 1
    }
    
    class Options extends Opts {}
}

object Perplexity {
    trait Opts extends Mixin.Opts {
        var perpweight:FMat = 1e-7f 
        var perpeps:Float = 1e-6f
        var perporthog:Boolean = true
        var perpnmats:Int = 1
    }
    
    class Options extends Opts {}
}

object Top {
    trait Opts extends Mixin.Opts {
        var topweight:FMat = 1e-7f 
        var topeps:Float = 1e-6f
        var topthreshold:Float = 0.001f
        var toporthog:Boolean = true
        var topnmats:Int = 1
    }
    
    class Options extends Opts {}
}
