package BIDMach.updaters
 
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._

class Grad(override val opts:Grad.Opts = new Grad.Options) extends Updater {
  
  var firstStep = 0f
 
  var modelmats:Array[Mat] = null
  var updatemats:Array[Mat] = null
  var sumSq:Mat = null 
  var stepn:Mat = null
  var mask:Mat = null
  var ve:Mat = null
	var te:Mat = null
	var lrate:Mat = null

  override def init(model0:Model) = {
    model = model0
	  modelmats = model.modelmats
	  updatemats = model.updatemats 
	  mask = opts.mask
    stepn = modelmats(0).zeros(1,1)
    te = modelmats(0).zeros(opts.texp.nrows, opts.texp.ncols)
    lrate = modelmats(0).zeros(opts.lrate.nrows, 1)
    te <-- opts.texp
  } 
  
	def update(ipass:Int, step:Long):Unit = {
	  val nsteps = if (step == 0) 1f else {
  	  if (firstStep == 0f) {
  	    firstStep = step
  	    1f
  	  } else {
  	    step / firstStep
  	  }
  	}
	  stepn.set(1f/nsteps);
	  val nmats = modelmats.length;
	  //	println("u2 sumsq %g" format mini(sumSq(0)).dv)
	  for (i <- 0 until nmats) {
	  	if (opts.lrate.ncols > 1) {
	  		lrate <-- opts.lrate(?,i);
	  	} else {
	  		lrate <-- opts.lrate;
	  	}
	  	if (opts.waitsteps < nsteps) {
	  		val tmp = updatemats(i) *@ (lrate *@ (stepn ^ te));
	  		modelmats(i) ~ modelmats(i) + tmp;
	  		if (mask != null) modelmats(i) ~ modelmats(i) *@ mask;
	  	}
	  }
	}
}


object Grad {
  trait Opts extends Updater.Opts {
    var lrate:FMat = 1f
    var texp:FMat = 0.5f
    var waitsteps = 2
    var mask:FMat = null
  }
  
  class Options extends Opts {}
}

