package BIDMach
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,GMat,GIMat,GSMat,HMat,IMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

abstract class Model(val opts:Model.Opts = new Model.Options) {
  
  var modelmats:Array[Mat] = null
  
  var updatemats:Array[Mat] = null
  
  var mats:Array[Mat] = null
  
  var gmats:Array[Mat] = null
  
  var useGPU = false
  
  def init(datasource:DataSource):Unit = {
	  mats = datasource.next
	  datasource.reset
	  useGPU = opts.useGPU && Mat.hasCUDA > 0
	  if (useGPU) {
	    gmats = new Array[Mat](mats.length)
	  } else {
	    gmats = mats
	  }
  }
  
  def doblock(mats:Array[Mat], ipass:Int, i:Long)                                       // Calculate an update for the updater
  
  def evalblock(mats:Array[Mat], ipass:Int):FMat                                        // Scores (log likelihoods)
  
  def doblockg(amats:Array[Mat], ipass:Int, i:Long) = {
    if (useGPU) copyMats(amats, gmats)            		
    doblock(gmats, ipass, i)
    if (useGPU && opts.putBack >= 0) {
    	for (i <- 1 to opts.putBack) {
    		amats(i) <-- gmats(i)
    	}
    }
  }
  
  def evalblockg(amats:Array[Mat], ipass:Int):FMat = {
	  if (useGPU) copyMats(amats, gmats)
	  val v = evalblock(gmats, ipass)
	  if (useGPU && opts.putBack >= 0) {
	    for (i <- 1 to opts.putBack) {
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
}


object Model {
	trait Opts {
	  var nzPerColumn:Int = 0
	  var startBlock = 8000
	  var useGPU = true
	  var putBack = -1
	  var doubleScore = false
	  var dim = 256
  }
	
	class Options extends Opts {}
}
