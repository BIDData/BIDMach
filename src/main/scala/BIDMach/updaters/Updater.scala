package BIDMach.updaters
 
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._


abstract class Updater(val opts:Updater.Opts = new Updater.Options) {
  var model:Model = null;
  var runningtime = 0.0;
  
  def init(model0:Model) = {
    model = model0 
  }
  
  def update(ipass:Int, step:Long):Unit
  def clear():Unit = {}
  
  def updateM(ipass:Int):Unit = {
    model.updatePass(ipass)
  }
}


object Updater {
  trait Opts {  
  }
  
  class Options extends Opts {}
}
