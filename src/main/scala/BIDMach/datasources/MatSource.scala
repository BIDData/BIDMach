package BIDMach.datasources
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,ND,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import java.io._


class MatSource(var mats:Array[ND], override val opts:MatSource.Opts = new MatSource.Options) extends DataSource(opts) { 
  var sizeMargin = 0f 
  var here = 0
  var there = 0
  var blockSize = 0
  var totalSize = 0
  var umat:ND = null;
  
  def init = {
    sizeMargin = opts.sizeMargin
    blockSize = opts.batchSize
    if (opts.addConstFeat) {
      mats(0) = mats(0).asMat on sparse(ones(1, mats(0).ncols)) 
    }
    if (opts.featType == 0) {
      mats(0).contents.set(1)     
    }
    here = -blockSize
    totalSize = mats(0).ncols
    omats = new Array[ND](mats.length)
    endmats = new Array[ND](mats.length)
    fullmats = new Array[ND](mats.length)   
  }
  
  def nmats = omats.length
  
  def reset = {
    here = -blockSize
  }
  
  def next:Array[ND] = {
    here = math.min(here+blockSize, mats(0).ncols)
    there = math.min(here+blockSize, mats(0).ncols)
  	for (i <- 0 until mats.length) {
  	  if (there - here == blockSize) {
  	    mats(i) match {
  	      case aa:Mat => fullmats(i) = mats(i).colslice(here, there, fullmats(i).asInstanceOf[Mat]);
  	      case _ => fullmats(i) = mats(i).colslice(here, there, fullmats(i));
  	    } 	    
  	    omats(i) = fullmats(i)
  	  } else {
  	  	mats(i) match {
  	  	  case aa:Mat => endmats(i) = mats(i).colslice(here, there, endmats(i).asInstanceOf[Mat]);
  	  	  case _ => endmats(i) = mats(i).colslice(here, there, endmats(i));
  	  	} 
  	    omats(i) = endmats(i) 	    
  	  }
  	}
  	omats
  }
  
  def hasNext:Boolean = {
    here + blockSize < mats(0).ncols
  }
  
  def progress = {
    math.min((here+blockSize)*1f/totalSize, 1f)
  }

}

object MatSource {
    trait Opts extends DataSource.Opts {
  }
  
  class Options extends Opts {   
  }
}

