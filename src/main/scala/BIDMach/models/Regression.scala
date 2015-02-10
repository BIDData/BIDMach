package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._

abstract class RegressionModel(override val opts:RegressionModel.Opts) extends Model {
  var targmap:Mat = null
  var targets:Mat = null
  var mask:Mat = null
  var sp:Mat = null
   
  override def copyTo(mod:Model) = {
    super.copyTo(mod);
    val rmod = mod.asInstanceOf[RegressionModel];
    rmod.targmap = targmap;
    rmod.targets = targets;
    rmod.mask = mask;
    rmod.sp = sp;    
  }
  
  def init() = {
    useGPU = opts.useGPU && Mat.hasCUDA > 0
    val data0 = mats(0)
    val m = size(data0, 1)
    val targetData = mats.length > 1
    val d = if (opts.targmap.asInstanceOf[AnyRef] != null) {
      opts.targmap.nrows 
    } else if (opts.targets.asInstanceOf[AnyRef] != null) {
      opts.targets.nrows 
    } else {
      mats(1).nrows  
    }
    val sdat = (sum(data0,2).t + 0.5f).asInstanceOf[FMat]
    sp = sdat / sum(sdat)
    println("corpus perplexity=%f" format (math.exp(-(sp ddot ln(sp)))))
    
    if (refresh) {
    	val mm = zeros(d,m);
    	setmodelmats(Array(if (useGPU) GMat(mm) else mm));
    }
    updatemats = Array(modelmats(0).zeros(modelmats(0).nrows, modelmats(0).ncols));
    targmap = if (useGPU && opts.targmap.asInstanceOf[AnyRef] != null) GMat(opts.targmap) else opts.targmap
    if (! targetData) {
      targets = if (useGPU && opts.targets.asInstanceOf[AnyRef] != null) GMat(opts.targets) else opts.targets
      mask =    if (useGPU && opts.rmask.asInstanceOf[AnyRef] != null) GMat(opts.rmask) else opts.rmask
    }
  } 
  
  def mupdate(data:Mat)
  
  def mupdate2(data:Mat, targ:Mat)
  
  def meval(data:Mat):FMat
  
  def meval2(data:Mat, targ:Mat):FMat
  
  def doblock(gmats:Array[Mat], ipass:Int, i:Long) = {
    if (gmats.length == 1) {
      mupdate(gmats(0))
    } else {
      mupdate2(gmats(0), gmats(1))
    }
  }
  
  def evalblock(mats:Array[Mat], ipass:Int, here:Long):FMat = {
    if (gmats.length == 1) {
      meval(gmats(0))
    } else {
      meval2(gmats(0), gmats(1))
    }
  }
}

object RegressionModel {
  trait Opts extends Model.Opts {
    var targets:FMat = null
    var targmap:FMat = null
    var rmask:FMat = null
  }
  
  class Options extends Opts {}
}
