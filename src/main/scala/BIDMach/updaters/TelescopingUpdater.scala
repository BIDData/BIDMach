package BIDMach.updaters
 
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._

class TelescopingUpdater(override val opts:TelescopingUpdater.Opts = new TelescopingUpdater.Options) extends Updater {
	var accumulators:Array[Mat] = null
  var firstStep = 0L
  var nextStep = 10L
  var nextCount = 0L
  var rm:Mat = null
  
  override def init(model0:Model) = {
  	super.init(model0)
    val modelmats = model0.modelmats
    val updatemats = model0.updatemats
    rm = model0.modelmats(0).zeros(1,1)
    accumulators = new Array[Mat](updatemats.length)
    for (i <- 0 until updatemats.length) {
    	accumulators(i) = updatemats(i).zeros(updatemats(i).nrows, updatemats(i).ncols)
    }
  	firstStep = 0L
    nextStep = 10L
    nextCount = 0L
  }
	
	def update(ipass:Int, step:Long) = {
	  if (firstStep == 0 && step > 0) {
	    firstStep = step
	  }
	  val updatemats = model.updatemats
    for (i <- 0 until updatemats.length) {
	    accumulators(i) ~ accumulators(i) + updatemats(i) 
    }
	  if (step >= nextCount) {
	    model.modelmats(0) ~ accumulators(0) / accumulators(1)
	    nextStep = (nextStep * opts.factor).toLong
	    nextCount = step + nextStep
	  }
  }
  
  override def clear() = {
	  for (i <- 0 until accumulators.length) {
     	accumulators(i).clear
	  }
  }
}

object TelescopingUpdater {
  trait Opts extends Updater.Opts {
    val factor = 1.5f
  }
  
  class Options extends Opts {}
}
