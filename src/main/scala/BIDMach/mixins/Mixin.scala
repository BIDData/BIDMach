package BIDMach.mixins
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._

@SerialVersionUID(100L)
abstract class Mixin(val opts:Mixin.Opts = new Mixin.Options) extends Serializable { 
  val options = opts
  var modelmats:Array[Mat] = null
  var updatemats:Array[Mat] = null
  var counter = 0 

  def compute(mats:Array[Mat], step:Float)
  
  def score(mats:Array[Mat], step:Float):FMat
  
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
