package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

abstract class FactorModel(override val opts:FactorModel.Opts) extends Model(opts) {
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    val data0 = mats(0)
    val m = size(data0, 1)
    val d = opts.dim
    val sdat = (sum(data0,2).t + 1.0f).asInstanceOf[FMat]
    val sp = sdat / sum(sdat)
    println("initial perplexity=%f" format math.exp(- (sp ddot ln(sp))) )
    
    val modelmat = rand(d,m) 
    modelmat ~ modelmat *@ sdat
    val msum = sum(modelmat, 2)
    modelmat ~ modelmat / msum
    modelmats = new Array[Mat](1)
    modelmats(0) = if (opts.useGPU && Mat.hasCUDA > 0) GMat(modelmat) else modelmat
    datasource.reset
    
    if (mats.size > 1) {
      while (datasource.hasNext) {
        mats = datasource.next
        val dmat = mats(1)
        dmat.set(1.0f/d)
        datasource.putBack(mats,1)
      }
    }
  } 
  
  def reuseuser(a:Mat):Mat = {
    val out = a match {
      case aa:SMat => FMat.newOrCheckFMat(opts.dim, a.ncols, null, a.GUID, "reuseuser".##)
      case aa:GSMat => GMat.newOrCheckGMat(opts.dim, a.ncols, null, a.GUID, "reuseuser".##)
    }
    out.set(1f)
    out
  }
  
  def uupdate(data:Mat, user:Mat)
  
  def mupdate(data:Mat, user:Mat)
  
  def mupdate2(data:Mat, user:Mat) = {}
  
  def evalfun(data:Mat, user:Mat):FMat
  
  def doblock(gmats:Array[Mat], i:Long) = {
    val sdata = gmats(0)
    val user = if (gmats.length > 1) gmats(1) else reuseuser(gmats(0))
    uupdate(sdata, user)
    mupdate(sdata, user)
  }
  
  def evalblock(mats:Array[Mat]):FMat = {
    val sdata = gmats(0)
    val user = if (gmats.length > 1) gmats(1) else reuseuser(gmats(0))
    uupdate(sdata, user)
    evalfun(sdata, user)
  } 
}

object FactorModel { 
  trait Opts extends Model.Opts { 
    var uiter = 8
    var weps = 1e-10f
    var minuser = 1e-8f
  }
  
  class Options extends Opts {}
} 


