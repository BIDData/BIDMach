package BIDMach.datasinks
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,ND,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import java.io._

@SerialVersionUID(100L)
abstract class DataSink(val opts:DataSink.Opts = new DataSink.Options) extends Serializable {   
  private var _GUID = Mat.myrand.nextLong
  def setGUID(v:Long):Unit = {_GUID = v} 
  def GUID:Long = _GUID
  def put;
  def init:Unit = {}
  def close = {}
  private var _nmats = 0;
  def nmats = _nmats;
  def setnmats(k:Int) = {_nmats = k;}
  var omats:Array[ND] = null
}


object DataSink {
  trait Opts extends BIDMat.Opts {
  } 
  
  class Options extends Opts {}
}

