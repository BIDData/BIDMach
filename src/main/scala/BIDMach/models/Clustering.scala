package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._

abstract class ClusteringModel(override val opts:ClusteringModel.Opts) extends Model {
  
  def init() = {

    useGPU = opts.useGPU && Mat.hasCUDA > 0
    val data0 = mats(0)
    val m = size(data0, 1)
    
    val nc = data0.ncols
    if (opts.dim > nc)
      throw new RuntimeException("Cluster initialization needs batchsize >= dim")

    val rp = randperm(nc)
    val mmi = full(data0(?,rp(0,0->opts.dim))).t
    
    modelmats = Array[Mat](1)
    modelmats(0) = if (useGPU) GMat(mmi) else mmi
    updatemats = new Array[Mat](1)
    updatemats(0) = modelmats(0).zeros(mmi.nrows, mmi.ncols)
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
