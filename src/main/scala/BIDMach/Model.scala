package BIDMach
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,GMat,GIMat,GSMat,HMat,IMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

abstract class Model {
  
  var modelmats:Array[Mat] = null
  
  var updatemats:Array[Mat] = null
  
  def init(datasource:DataSource):Unit
  
  def doblock(mats:Array[Mat], i:Long)                                    // Calculate an update for the updater
  
  def evalblock(mats:Array[Mat]):FMat                                        // Scores (log likelihoods)

}


object Model {
	class Options {
	  var nzPerColumn:Int = 0
	  var startBlock = 8000
	  var useGPU = false
  }
	
}
