package BIDMach
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,GMat,GIMat,GSMat,HMat,IMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

abstract class Model(val opts:Model.Options = new Model.Options) {
  
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
	    for (i <- 0 until mats.length) {
	      gmats(i) = mats(i) match {
	        case aa:FMat => GMat(aa)
	        case aa:SMat => GSMat(aa)
	      }
	    }
	  } else {
	    gmats = mats
	  }
  }
  
  def doblock(mats:Array[Mat], i:Long)                                       // Calculate an update for the updater
  
  def evalblock(mats:Array[Mat]):FMat                                        // Scores (log likelihoods)
  
  def doblockg(amats:Array[Mat], i:Long) = {
    if (useGPU) copyMats(amats, gmats)            		
    doblock(gmats, i)
    if (useGPU && opts.putBack >= 0) amats(opts.putBack) <-- gmats(opts.putBack)
  }
  
  def evalblockg(amats:Array[Mat]):FMat = {
	  if (useGPU) copyMats(amats, gmats)
	  val v = evalblock(gmats)
	  if (useGPU && opts.putBack >= 0) amats(opts.putBack) <-- gmats(opts.putBack)
	  v
  }

  def copyMats(from:Array[Mat], to:Array[Mat]) = {
	  for (i <- 0 until from.length) {
		  to(i) = to(i) <-- from(i)
	  }
}
}


object Model {
	class Options {
	  var nzPerColumn:Int = 0
	  var startBlock = 8000
	  var useGPU = true
	  var putBack = -1
	  var doubleScore = false
	  var dim = 256
  }
}
