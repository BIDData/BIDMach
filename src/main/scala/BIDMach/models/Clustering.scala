package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._

abstract class ClusteringModel(override val opts:ClusteringModel.Opts) extends Model {
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    useGPU = opts.useGPU && Mat.hasCUDA > 0
    val data0 = mats(0)
    val m = size(data0, 1)
    
    val mm = rand(opts.dim, m) 
    mm ~ mm / sqrt(mm dotr mm)
    modelmats = Array[Mat](1)
    modelmats(0) = if (useGPU) GMat(mm) else mm 
    updatemats = new Array[Mat](1)
    updatemats(0) = modelmats(0).zeros(mm.nrows, mm.ncols)
  } 
  
  def mupdate(data:Mat, ipass:Int):Unit
   
  def evalfun(data:Mat):FMat
  
  def doblock(gmats:Array[Mat], ipass:Int, i:Long) = {
    mupdate(gmats(0), ipass)
  }
  
  def evalblock(mats:Array[Mat], ipass:Int):FMat = {
    evalfun(gmats(0))
  }
}

object ClusteringModel {
  trait Opts extends Model.Opts {
  }
  
  class Options extends Opts {}
}
