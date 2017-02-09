package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._

/**
 * Abstract class with shared code for Regression Models
 */
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
    val m = data0.nrows;
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
      setmodelmats(Array(mm))
    }
    modelmats(0) = convertMat(modelmats(0));
    updatemats = Array(modelmats(0).zeros(modelmats(0).nrows, modelmats(0).ncols));
    targmap = if (opts.targmap.asInstanceOf[AnyRef] != null) convertMat(opts.targmap) else opts.targmap
    if (! targetData) {
      targets = if (opts.targets.asInstanceOf[AnyRef] != null) convertMat(opts.targets) else opts.targets
      mask =    if (opts.rmask.asInstanceOf[AnyRef] != null) convertMat(opts.rmask) else opts.rmask
    }
  } 
  
  def mupdate(data:Mat, ipass:Int, i:Long)
  
  def mupdate2(data:Mat, targ:Mat, ipass:Int, i:Long)
  
  def meval(data:Mat):FMat
  
  def meval2(data:Mat, targ:Mat):FMat
  
  def dobatch(gmats:Array[Mat], ipass:Int, i:Long) = {
    if (gmats.length == 1) {
      mupdate(gmats(0), ipass, i)
    } else {
      mupdate2(gmats(0), gmats(1), ipass, i)
    }
  }
  
  def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat = {
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
