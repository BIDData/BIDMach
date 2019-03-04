package BIDMach.updaters

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._

@SerialVersionUID(100L)
class Batch(override val opts:Batch.Opts = new Batch.Options) extends Updater {
  
  override def init(model0:Model) = {
    super.init(model0)
  }
     
  override def update(ipass:Int, step:Long) = {}
}

@SerialVersionUID(100L)
object Batch {
  trait Opts extends Updater.Opts {
    var beps = 1e-5f
  }
  
  class Options extends Opts {}
}
