package BIDMach
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,GMat,GIMat,GSMat,HMat,IMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

abstract class Model {
  
  var datasource:DataSource = null
  
  var modelmats:Array[Mat] = null
  
  var updatemats:Array[Mat] = null
  
  var regularizer:Regularizer = null
  
  var updater:Updater = null
  
  var sampler:Sampler = null
  
  def init:Unit
  
  def doblock(i:Int):Unit                                // Calculate an update for the updater
  
//  def evalfun(datamats:Array[Mat]):Mat                                 // Scores (log likelihoods)

}


object Model {
	class Options {
	  var nzPerColumn:Int = 0
	  var startBlock = 8000
	  var useGPU = false
  }
	
}
