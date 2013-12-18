package BIDMach.updaters
 
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._

class IncMultUpdater(override val opts:IncMultUpdater.Opts = new IncMultUpdater.Options) extends Updater {
  
  var firstStep = 0f
  var rm:Mat = null
  
  override def init(model0:Model) = {
    super.init(model0)
    rm = model0.modelmats(0).zeros(1,1)
  }
      
  def update(ipass:Int, step:Long) = {
    val modelmats = model.modelmats
    val updatemats = model.updatemats
    val mm = modelmats(0)
    val ms = modelmats(1)
    val um = updatemats(0)
    val ums = updatemats(1)
    val rr = if (step == 0) 1f else {
	    if (firstStep == 0f) {
	    	firstStep = step
	    	1f
	    } else {
	    	math.pow(firstStep / step, opts.power).toFloat
	    }
  	}
//    println("rr=%g, %g %g" format (rr, mini(mini(um,1),2).dv, maxi(maxi(um,1),2).dv))
    um ~ um *@ rm.set(rr)
//    println("rr=%g, %g %g" format (rr, mini(mini(um,1),2).dv, maxi(maxi(um,1),2).dv))
    ln(mm, mm)
//    println("mm=%g %g" format (mini(mini(mm,1),2).dv, maxi(maxi(mm,1),2).dv))
    mm ~ mm *@ rm.set(1-rr)
//    println("mm=%g %g" format (mini(mini(mm,1),2).dv, maxi(maxi(mm,1),2).dv))
    mm ~ mm + um 
//    println("mm=%g %g" format (mini(mini(mm,1),2).dv, maxi(maxi(mm,1),2).dv))
    exp(mm, mm)
//    println("mm=%g %g" format (mini(mini(mm,1),2).dv, maxi(maxi(mm,1),2).dv))
    mm ~ mm / sum(mm,2)
  }
  
  override def clear() = {
	  firstStep = 0f
  }
}


object IncMultUpdater {
  trait Opts extends Updater.Opts {
    var warmup = 0L 
    var power = 0.3f
  }
  
  class Options extends Opts {}
}
