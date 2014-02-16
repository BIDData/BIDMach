package BIDMach
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,GMat,GIMat,GSMat,HMat,IMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

abstract class Sampler {
  
  val options:Sampler.Options
  
  def insample(pos:Int, modelnum:Int):Int = 1
  
  def outsample(mat:Mat):Unit = {}


}


object Sampler {
	class Options {
	
  }
}
