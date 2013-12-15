package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

abstract class RegressionModel(override val opts:RegressionModel.Opts) extends Model {
  var targmap:Mat = null
  var targets:Mat = null
  var mask:Mat = null
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    useGPU = opts.useGPU && Mat.hasCUDA > 0
    val data0 = mats(0)
    val m = size(data0, 1)
    val d = opts.targmap.nrows
    val sdat = (sum(data0,2).t + 0.5f).asInstanceOf[FMat]
    val sp = sdat / sum(sdat)
    println("initial perplexity=%f" format (sp ddot ln(sp)) )
    
    val rmat = rand(d,m) 
    rmat ~ rmat *@ sdat
    val msum = sum(rmat, 2)
    rmat ~ rmat / msum
    val mm = rmat
    modelmats = Array[Mat](1)
    modelmats(0) = if (useGPU) GMat(mm) else mm 
    updatemats = new Array[Mat](1)
    updatemats(0) = modelmats(0).zeros(mm.nrows, mm.ncols)
    targets = if (useGPU) GMat(opts.targets) else opts.targets
    targmap = if (useGPU) GMat(opts.targmap) else opts.targmap
    mask = if (useGPU) GMat(opts.mask) else opts.mask
  } 
  
  def mupdate(data:Mat):FMat
  
  def doblock(gmats:Array[Mat], ipass:Int, i:Long) = {
    val sdata = gmats(0)
    mupdate(sdata)
  }
  
  def evalblock(mats:Array[Mat], ipass:Int):FMat = {
    val sdata = gmats(0)
    mupdate(sdata)
  }
}

object RegressionModel {
  trait Opts extends Model.Opts {
    var targets:FMat = null
    var targmap:FMat = null
    var mask:FMat = null
  }
  
  class Options extends Opts {}
}
