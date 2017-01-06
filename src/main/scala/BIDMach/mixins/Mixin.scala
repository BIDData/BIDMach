package BIDMach.mixins
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,FND,IMat,HMat,GMat,GIMat,GSMat,GND,ND,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._

@SerialVersionUID(100L)
abstract class Mixin(val opts:Mixin.Opts = new Mixin.Options) extends Serializable { 
  val options = opts
  var modelmats:Array[ND] = null
  var updatemats:Array[ND] = null
  var counter = 0 

  def compute(mats:Array[ND], step:Float)
  
  def score(mats:Array[ND], step:Float):FMat
  
  def init(model:Model) = {
    modelmats = model.modelmats
    updatemats = model.updatemats
  }
}

object Mixin {
	trait Opts extends BIDMat.Opts {
        var mixinInterval = 1
    }
	
	class Options extends Opts {}
}
