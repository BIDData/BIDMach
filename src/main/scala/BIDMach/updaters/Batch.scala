package BIDMach.updaters

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._


class Batch(override val opts:Batch.Opts = new Batch.Options) extends Updater {
  var mm:Mat = null
  var um:Mat = null
  
  override def init(model0:Model) = {
    super.init(model0)
    mm = model0.modelmats(0)
    um = model0.updatemats(0)
  }
     
  def update(ipass:Int, step:Long) = {}
  
  override def clear() = {
    um.clear
  }
  
  override def updateM(ipass:Int):Unit = {
    mm ~ um / max(sqrt(um dotr um), opts.beps)
    clear
  }
}

object Batch {
  trait Opts extends Updater.Opts {
    var beps = 1e-5f
  }
  
  class Options extends Opts {}
}
