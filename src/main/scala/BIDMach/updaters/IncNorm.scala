package BIDMach.updaters

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._

class IncNorm(override val opts:IncNorm.Opts = new IncNorm.Options) extends Updater(opts) {
  
  var firstStep = 0f
  var rm:Mat = null
  var restart:Mat = null
  var started:Int = 0
  
  override def init(model0:Model) = {
  	super.init(model0)
    val modelmats = model0.modelmats
    val updatemats = model0.updatemats
    restart = modelmats(0) + 1f
    rm = model0.modelmats(0).zeros(1,1)
    firstStep = 0f
  }
      
  def update(ipass:Int, step:Long) = {
  	val modelmats = model.modelmats
  	val updatemats = model.updatemats
  	val mm = modelmats(0)
  	val um = updatemats(0)
  	val rr = if (step == 0) 0.99f else {
  	  if (firstStep == 0f) {
  	    firstStep = step
  	    0.99f
  	  } else {
  	    math.pow(firstStep / step, opts.power).toFloat
  	  }
  	}
  	if (modelmats.length > 1) {
  		val ms = modelmats(1)
  		val ums = updatemats(1)
  		ums ~ ums *@ rm.set(rr)
  		ms ~ ms *@ rm.set(1-rr)
  		ms ~ ms + ums
  		um ~ um / ms
  	}
  	um ~ um *@ rm.set(rr)
  	mm ~ mm *@ rm.set(1-rr)
    mm ~ mm + um 
    mm ~ mm / sum(mm,2)
    if (opts.warmup > 0) {
      if (started == 0 && step > opts.warmup) {
        restart <-- mm
        started = 1
      }
      if (started == 1 && step > 2*opts.warmup) {
        mm ~ mm - restart
        max(mm, 0f, mm)
        mm ~ mm / sum(mm,2)
        started = 2
      }
    }
  }
  
  override def clear() = {
	  firstStep = 0f
  }
}

object IncNorm {
  trait Opts extends Updater.Opts {
    var warmup = 0L 
    var power = 0.3f
  }
  
  class Options extends Opts {}
}
