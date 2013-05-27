package BIDMach
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,GMat,GIMat,GSMat,HMat,IMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

abstract class Model {
  
  val options:Model.Options
  
  var modelmats:Array[Mat] = null
  
  var updatemats:Array[Mat] = null
  
  var datamats:Array[Mat] = null
  
  var regularizer:Regularizer = null
  
  var updater:Updater = null
  
  var sampler:Sampler = null
  
  def initmodel:Unit
  
  def doblock(istart:Int, iend:Int):Unit
  
  def evalfun(istart:Int, iend:Int):Mat

}


object Model {
	class Options {
	  var nzPerColumn:Int = 0
	  var startBlock = 8000
	  var useGPU = false
  }
	
}
