package BIDMach.models
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,GMat,GDMat,GIMat,GSMat,GSDMat,HMat,IMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._

/**
 * Abstract class with shared code for all models
 */
abstract class Model(val opts:Model.Opts = new Model.Options) {
  
  var datasource:DataSource = null
  
  var _modelmats:Array[Mat] = null
  
  var parent_model:Model = null
  
  def modelmats:Array[Mat] = {
    if (_modelmats != null) {
      _modelmats
    } else {
      parent_model._modelmats
    }
  }
  
  def setmodelmats(a:Array[Mat]) = {
    _modelmats = a;
  }
  
  var updatemats:Array[Mat] = null
  
  var mats:Array[Mat] = null
  
  var gmats:Array[Mat] = null
  
  var useGPU = false
  
  var useDouble = false
  
  var putBack = -1
  
  var refresh = true
  
  def copyTo(mod:Model) = {
    mod.datasource = datasource;
    mod._modelmats = modelmats;
    mod.updatemats = updatemats;
    mod.mats = mats;
    mod.gmats = gmats;
  }
  
  def bind(ds:DataSource):Unit = {
	  datasource = ds;
	  mats = datasource.next;
	  datasource.reset;
	  putBack = datasource.opts.putBack;
	  useGPU = opts.useGPU && Mat.hasCUDA > 0;
	  useDouble = opts.useDouble;
	  if (useGPU || useDouble) {
	  	gmats = new Array[Mat](mats.length);
	  } else {
	  	gmats = mats;
	  }
  }
  
  def init():Unit
  
  def doblock(mats:Array[Mat], ipass:Int, here:Long)                                       // Calculate an update for the updater
  
  def evalblock(mats:Array[Mat], ipass:Int, here:Long):FMat                                        // Scores (log likelihoods)
  
  def doblockg(amats:Array[Mat], ipass:Int, here:Long) = {
    if (useGPU) copyMats(amats, gmats)            		
    doblock(gmats, ipass, here)
    if ((useGPU || useDouble) && putBack >= 0) {
    	for (i <- 1 to putBack) {
    		amats(i) <-- gmats(i)
    	}
    }
  }
  
  def evalblockg(amats:Array[Mat], ipass:Int, here:Long):FMat = {
    if (useGPU) copyMats(amats, gmats)
    val v = evalblock(gmats, ipass, here)
    if ((useGPU || useDouble) && putBack >= 0) {
      for (i <- 1 to putBack) {
        amats(i) <-- gmats(i)
      }
    }
	v
  }

  def copyMats(from:Array[Mat], to:Array[Mat]) = {
    for (i <- 0 until from.length) {
      if (useGPU) {
        if (useDouble) {
         	to(i) = from(i) match {
        	case aa:FMat => GDMat(aa)
        	case aa:IMat => GIMat(aa)
        	case aa:SMat => GSDMat(aa)
        	}         
        } else {
        	to(i) = from(i) match {
        	case aa:FMat => GMat(aa)
        	case aa:IMat => GIMat(aa)        	
        	case aa:SMat => GSMat(aa)
        	}
        }
      } else {
      	if (useDouble) {
         	to(i) = from(i) match {
        	case aa:FMat => DMat(aa)
        	case aa:SMat => SDMat(aa)
        	}
      	}
      }
    }
  }
  
  def updatePass(ipass:Int) = {}
  
}


object Model {
	trait Opts {
	  var nzPerColumn:Int = 0
	  var startBlock = 8000
	  var useGPU = true
	  var useDouble = false
	  var doubleScore = false
	  var dim = 256
  }
	
	class Options extends Opts {}
}
