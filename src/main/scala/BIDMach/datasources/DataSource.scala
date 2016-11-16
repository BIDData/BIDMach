package BIDMach.datasources
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import java.io._

@SerialVersionUID(100L)
abstract class DataSource(val opts:DataSource.Opts = new DataSource.Options) extends Serializable {   
  private var _GUID = Mat.myrand.nextLong
  def setGUID(v:Long):Unit = {_GUID = v} 
  def GUID:Long = _GUID
  def next:Array[Mat]  
  def hasNext:Boolean
  def reset:Unit
  def putBack(mats:Array[Mat],i:Int):Unit = {throw new RuntimeException("putBack not implemented")}
  def setupPutBack(n:Int,dim:Int):Unit = {throw new RuntimeException("putBack not implemented")}
  def nmats:Int
  def init:Unit
  def progress:Float
  def close = {}
  var omats:Array[Mat] = null
  var endmats:Array[Mat] = null
  var fullmats:Array[Mat] = null
}


object DataSource {
  trait Opts extends BIDMat.Opts {
    var batchSize = 10000
    var sizeMargin = 3f
    var sample = 1f
    var addConstFeat:Boolean = false
    var featType:Int = 1                 // 0 = binary features, 1 = linear features, 2 = threshold features
    var featThreshold:Mat = null
    var putBack = -1
  } 
  
  class Options extends Opts {}
}

