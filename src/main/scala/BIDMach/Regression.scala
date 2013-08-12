package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

abstract class RegressionModel(override val opts:RegressionModel.Options) extends Model {
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    val data0 = mats(0)
    val m = size(data0, 1)
    val d = opts.targets.nrows
    val sdat = (sum(data0,2).t + 0.5f).asInstanceOf[FMat]
    val sp = sdat / sum(sdat)
    println("initial perplexity=%f" format (sp ddot ln(sp)) )
    
    val rmat = rand(d,m) 
    rmat ~ rmat *@ sdat
    val msum = sum(rmat, 2)
    rmat ~ rmat / msum
    val modelmat = opts.targets on rmat
    modelmats = Array[Mat](1)
    modelmats(0) = if (opts.useGPU) GMat(modelmat) else modelmat    
  } 
  
  def mupdate(data:Mat):FMat
  
  def doblock(gmats:Array[Mat], i:Long) = {
    val sdata = gmats(0)
    mupdate(sdata)
  }
  
  def evalblock(mats:Array[Mat]):FMat = {
    val sdata = gmats(0)
    mupdate(sdata)
  }
}

object RegressionModel {
  class Options extends Model.Options {
    var targets:FMat = null
    var nrows = 0
    var nmodels = 0
    var transpose:Boolean = false
  }
}
