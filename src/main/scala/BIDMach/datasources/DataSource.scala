package BIDMach.datasources
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,ND,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import java.io._

@SerialVersionUID(100L)
abstract class DataSource(val opts:DataSource.Opts = new DataSource.Options) extends Serializable {   
  private var _GUID = Mat.myrand.nextLong
  def setGUID(v:Long):Unit = {_GUID = v} 
  def GUID:Long = _GUID
  def next:Array[ND]  
  def hasNext:Boolean
  def reset:Unit
  def nmats:Int
  def init:Unit
  def progress:Float
  def close = {}
  var omats:Array[ND] = null
  var endmats:Array[ND] = null
  var fullmats:Array[ND] = null
}


object DataSource {
  trait Opts extends BIDMat.Opts {
    var batchSize = 10000
    var sizeMargin = 3f
    var sample = 1f
    var addConstFeat:Boolean = false
    var featType:Int = 1                 // 0 = binary features, 1 = linear features, 2 = threshold features
    var featThreshold:ND = null
  } 
  
  class Options extends Opts {}
}

