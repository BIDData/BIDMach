package BIDMach.datasources
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import scala.actors._
import java.io._


class MatDataSource(var mats:Array[Mat], override val opts:MatDataSource.Opts = new MatDataSource.Options) extends DataSource(opts) { 
  var sizeMargin = 0f 
  var here = 0
  var there = 0
  var blockSize = 0
  var totalSize = 0
  var umat:Mat = null
  
  def init = {
    sizeMargin = opts.sizeMargin
    blockSize = opts.blockSize
    if (opts.addConstFeat) {
      mats(0) = mats(0) on sparse(ones(1, mats(0).ncols)) 
    }
    if (opts.featType == 0) {
      mats(0).contents.set(1)     
    }
    here = -blockSize
    totalSize = mats(0).ncols
    omats = new Array[Mat](mats.length)
    endmats = new Array[Mat](mats.length)
    fullmats = new Array[Mat](mats.length)   
  }
  
  def nmats = omats.length
  
  def reset = {
    here = -blockSize
  }
  
  def next:Array[Mat] = {
    here = math.min(here+blockSize, mats(0).ncols)
    there = math.min(here+blockSize, mats(0).ncols)
  	for (i <- 0 until mats.length) {
  	  if (there - here == blockSize) {
  	    fullmats(i) = mats(i).colslice(here, there, fullmats(i))
  	    omats(i) = fullmats(i)
  	  } else {
  	    endmats(i) = mats(i).colslice(here, there, endmats(i))
  	    omats(i) = endmats(i) 	    
  	  }
  	}
  	omats
  }
  
  def hasNext:Boolean = {
    here + blockSize < mats(0).ncols
  }
  
  override def setupPutBack(n:Int, dim:Int) = {
    if (mats.length < n || mats(n-1).asInstanceOf[AnyRef] == null || mats(n-1).nrows != dim) {
      val newmats = new Array[Mat](n)
      for (i <- 0 until mats.length) {
        newmats(i) = mats(i)
      }
      for (i <- mats.length until n) {
      	newmats(i) = zeros(dim, mats(0).ncols)
      }
      mats = newmats
    } 
  }
  
  override def putBack(tmats:Array[Mat],n:Int):Unit = {
    for (i <- 1 to n) {
    	tmats(i).colslice(0, tmats(i).ncols, mats(i), here)
    }
  }
  
  def progress = {
    math.min((here+blockSize)*1f/totalSize, 1f)
  }

}

object MatDataSource {
  trait Opts extends DataSource.Opts {
  }
  
  class Options extends Opts {   
  }
}

