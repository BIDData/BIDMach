package BIDMach.mixins
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._

abstract class Mixin(val opts:Mixin.Opts = new Mixin.Options) { 
  val options = opts
  var modelmats:Array[Mat] = null
  var updatemats:Array[Mat] = null
  
  def compute(mats:Array[Mat], step:Float)
  
  def score(mats:Array[Mat], step:Float):FMat
  
  def init(model:Model) = {
    modelmats = model.modelmats
    updatemats = model.updatemats
  }
}

object Mixin {
	trait Opts {}
	
	class Options extends Opts {}
}
