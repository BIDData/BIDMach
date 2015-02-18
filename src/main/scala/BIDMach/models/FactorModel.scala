package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._

/**
 * An Abstract class with shared code for Factor Models
 */
abstract class FactorModel(override val opts:FactorModel.Opts) extends Model(opts) {
  
  def init() = {
    val data0 = mats(0)
    val m = size(data0, 1)
    val d = opts.dim
    val sdat = (sum(data0,2).t + 1.0f).asInstanceOf[FMat]
    val sp = sdat / sum(sdat)
    println("corpus perplexity=%f" format math.exp(- (sp ddot ln(sp))) )
    
    if (refresh) {
    	val modelmat = rand(d,m);
    	modelmat ~ modelmat *@ sdat;
    	val msum = sum(modelmat, 2);
    	modelmat ~ modelmat / msum;
    	setmodelmats(Array[Mat](1));
    	modelmats(0) = if (opts.useGPU && Mat.hasCUDA > 0) GMat(modelmat) else modelmat;
    }
    
    if (mats.size > 1) {
      while (datasource.hasNext) {
        mats = datasource.next
        val dmat = mats(1)
        dmat.set(1.0f/d)
        datasource.putBack(mats,1)
      }
    }
  } 
  
  
  def uupdate(data:Mat, user:Mat, ipass:Int)
  
  def mupdate(data:Mat, user:Mat, ipass:Int)
  
  def mupdate2(data:Mat, user:Mat, ipass:Int) = {}
  
  def evalfun(data:Mat, user:Mat, ipass:Int):FMat
  
  def doblock(gmats:Array[Mat], ipass:Int, i:Long) = {
    val sdata = gmats(0)
    val user = if (gmats.length > 1) gmats(1) else FactorModel.reuseuser(gmats(0), opts.dim, 1f)
    uupdate(sdata, user, ipass)
    mupdate(sdata, user, ipass)
  }
  
  def evalblock(mats:Array[Mat], ipass:Int, here:Long):FMat = {
    val sdata = gmats(0)
    val user = if (gmats.length > 1) gmats(1) else FactorModel.reuseuser(gmats(0), opts.dim, 1f)
    uupdate(sdata, user, ipass)
    evalfun(sdata, user, ipass)
  } 
}

object FactorModel { 
  trait Opts extends Model.Opts { 
    var uiter = 5
    var weps = 1e-10f
    var minuser = 1e-8f
  }
  
  def reuseuser(a:Mat, dim:Int, ival:Float):Mat = {
    val out = a match {
      case aa:SMat => FMat.newOrCheckFMat(dim, a.ncols, null, a.GUID, "SMat.reuseuser".##)
      case aa:FMat => FMat.newOrCheckFMat(dim, a.ncols, null, a.GUID, "FMat.reuseuser".##)
      case aa:GSMat => GMat.newOrCheckGMat(dim, a.ncols, null, a.GUID, "GSMat.reuseuser".##)
      case aa:GMat => GMat.newOrCheckGMat(dim, a.ncols, null, a.GUID, "GMat.reuseuser".##)
    }
    out.set(ival)
    out
  }
  
  class Options extends Opts {}
} 


