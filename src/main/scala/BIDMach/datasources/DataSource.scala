package BIDMach.datasources
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import scala.actors._
import java.io._

abstract class DataSource(val opts:DataSource.Opts = new DataSource.Options) {   
  def next:Array[Mat]  
  def hasNext:Boolean
  def reset:Unit
  def putBack(mats:Array[Mat],i:Int):Unit = {throw new RuntimeException("putBack not implemented")}
  def setupPutBack(n:Int,dim:Int):Unit = {throw new RuntimeException("putBack not implemented")}
  def nmats:Int
  def init:Unit
  def progress:Float
  var omats:Array[Mat] = null
  var endmats:Array[Mat] = null
  var fullmats:Array[Mat] = null
}


object DataSource {
  trait Opts {
    var blockSize = 100000
    var sizeMargin = 3f
    var sample = 1f
    var addConstFeat:Boolean = false
    var featType:Int = 1                 // 0 = binary features, 1 = linear features
  } 
  
  class Options extends Opts {}
}

