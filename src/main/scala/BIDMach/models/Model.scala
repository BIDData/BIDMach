package BIDMach.models
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,GMat,GIMat,GSMat,HMat,IMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._

abstract class Model(val opts:Model.Opts = new Model.Options) {
  
  var datasource:DataSource = null
  
  var modelmats:Array[Mat] = null
  
  var updatemats:Array[Mat] = null
  
  var mats:Array[Mat] = null
  
  var gmats:Array[Mat] = null
  
  var useGPU = false
  
  var putBack = -1
  
  def bind(ds:DataSource):Unit = {
      datasource = ds;
	  mats = datasource.next
	  datasource.reset
	  putBack = datasource.opts.putBack
	  useGPU = opts.useGPU && Mat.hasCUDA > 0
	  if (useGPU) {
	    gmats = new Array[Mat](mats.length)
	  } else {
	    gmats = mats
	  }
  }
  
  def init():Unit
  
  def doblock(mats:Array[Mat], ipass:Int, i:Long)                                       // Calculate an update for the updater
  
  def evalblock(mats:Array[Mat], ipass:Int):FMat                                        // Scores (log likelihoods)
  
  def doblockg(amats:Array[Mat], ipass:Int, i:Long) = {
    if (useGPU) copyMats(amats, gmats)            		
    doblock(gmats, ipass, i)
    if (useGPU && putBack >= 0) {
    	for (i <- 1 to putBack) {
    		amats(i) <-- gmats(i)
    	}
    }
  }
  
  def evalblockg(amats:Array[Mat], ipass:Int):FMat = {
    if (useGPU) copyMats(amats, gmats)
    val v = evalblock(gmats, ipass)
    if (useGPU && putBack >= 0) {
      for (i <- 1 to putBack) {
        amats(i) <-- gmats(i)
      }
    }
	v
  }

  def copyMats(from:Array[Mat], to:Array[Mat]) = {
	  for (i <- 0 until from.length) {
	    if (useGPU) {
	    	to(i) = from(i) match {
	    	case aa:FMat => GMat(aa)
	    	case aa:SMat => GSMat(aa)
	    	}
	    }
	  }
  }
  
  def updatePass = {}
  
}


object Model {
	trait Opts {
	  var nzPerColumn:Int = 0
	  var startBlock = 8000
	  var useGPU = true
	  var doubleScore = false
	  var dim = 256
  }
	
	class Options extends Opts {}
}
