package BIDMach.updaters
 
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._


abstract class Updater(val opts:Updater.Opts = new Updater.Options) extends Serializable {
  var model:Model = null;
  var runningtime = 0.0;
  
  def init(model0:Model) = {
    model = model0 
  }
  
  def clear():Unit = {}
  
  def update(ipass:Int, step:Long):Unit = {}
  
  def update(ipass:Int, step:Long, gprogress:Float):Unit = update(ipass, step)
  
  def updateM(ipass:Int):Unit = {
    model.updatePass(ipass)
  }
}


object Updater {
  trait Opts extends BIDMat.Opts {  
  }
  
  class Options extends Opts {}
}
