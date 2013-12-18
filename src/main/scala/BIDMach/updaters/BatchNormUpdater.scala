package BIDMach.updaters

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._


class BatchNormUpdater(override val opts:BatchNormUpdater.Opts = new BatchNormUpdater.Options) extends Updater {
  var accumulators:Array[Mat] = null
  
  override def init(model0:Model) = {
    super.init(model0)
    val modelmats = model.modelmats
    val updatemats = model.updatemats
    accumulators = new Array[Mat](updatemats.length)
    for (i <- 0 until accumulators.length) {
    	accumulators(i) = updatemats(i).zeros(updatemats(i).nrows, updatemats(i).ncols)
    }
  }
     
  def update(ipass:Int, step:Long) = {
  	val updatemats = model.updatemats
    for (i <- 0 until accumulators.length) {
    	accumulators(i) ~ accumulators(i) + updatemats(i) 
    }
  }
  
  override def clear() = {
	  for (i <- 0 until accumulators.length) {
	  	accumulators(i).clear
	  }
  }
  
  override def updateM(ipass:Int):Unit = {
    val mm = model.modelmats(0)
    mm ~ accumulators(0) / accumulators(1)
    mm ~ mm / sum(mm,2)
    clear
  }
}

object BatchNormUpdater {
  trait Opts extends Updater.Opts {
  }
  
  class Options extends Opts {}
}
