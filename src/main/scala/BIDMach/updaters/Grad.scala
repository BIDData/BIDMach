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
  var momentum:Array[Mat] = null;
  var stepn:Mat = null
  var mask:Mat = null
  var ve:Mat = null
	var te:Mat = null
	var pe:Mat = null
	var lrate:Mat = null
	var mu:Mat = null
	var randmat:Array[Mat] = null

  override def init(model0:Model) = {
    model = model0;
	  modelmats = model.modelmats;
	  updatemats = model.updatemats;
	  mask = opts.mask;
	  val mm = modelmats(0);
    stepn = mm.zeros(1,1);
    val nmats = modelmats.length;
    val hasmomentum = (opts.momentum.asInstanceOf[AnyRef] != null || opts.nesterov.asInstanceOf[AnyRef] != null);
    if (hasmomentum) {
      momentum = new Array[Mat](nmats);
      for (i <- 0 until nmats) {
    	  momentum(i) = modelmats(i).zeros(modelmats(i).nrows, modelmats(i).ncols);
      }
    }
    if (opts.langevin > 0) {
      randmat = new Array[Mat](nmats);
      for (i <- 0 until nmats) {
        randmat(i) = modelmats(i).zeros(modelmats(i).nrows, modelmats(i).ncols);
      }
    }
    if (opts.texp.asInstanceOf[AnyRef] != null) {
      te = mm.zeros(opts.texp.nrows, opts.texp.ncols);
      te <-- opts.texp;
    }
    if (opts.pexp.asInstanceOf[AnyRef] != null) {
      pe = mm.zeros(opts.pexp.nrows, opts.pexp.ncols);
      pe <-- opts.pexp;
    }
    lrate = mm.zeros(opts.lrate.nrows, 1);
    mu = mm.zeros(1,1);
  } 
  
	override def update(ipass:Int, step:Long, gprogress:Float):Unit = {
  	val nsteps = if (step == 0) 1f else {
  		if (firstStep == 0f) {
  			firstStep = step;
  			1f;
  		} else {
  			step / firstStep;
  		}
  	}
  	val nmats = updatemats.length;
	  //	println("u2 sumsq %g" format mini(sumSq(0)).dv)
	  for (i <- 0 until nmats) {
		  val mm = modelmats(i);
      val tscale = if (te.asInstanceOf[AnyRef] != null) {
        stepn.set(1f/nsteps);
        stepn ^ te;
      } else {
        stepn.set(1f/(ipass+1));
        stepn ^ pe;
      }
      if (opts.policies.asInstanceOf[AnyRef] != null) {
			  if (opts.policies.length > 1) {
				  tscale.set(opts.policies(i)(nsteps, gprogress));
			  } else {
				  tscale.set(opts.policies(0)(nsteps, gprogress));
			  }
		  }
	  	if (opts.lrate.ncols > 1) {
	  		lrate <-- opts.lrate(?,i);
	  	} else {
	  		lrate <-- opts.lrate;
	  	}

	  	if (opts.waitsteps < nsteps) {
        val grad = updatemats(i);
        if (opts.langevin > 0) {                              // Add Langevin random permutations
        	normrnd(0, opts.langevin, randmat(i));
        	grad ~ grad + randmat(i);
        }
	  		grad ~ grad *@ (lrate *@ tscale);
	  		if (opts.momentum.asInstanceOf[AnyRef] != null) {
	  			val i0 = if (opts.momentum.length > 1) i else 0;
	  			mu <-- opts.momentum(i0);                           // Get the momentum decay rate
	  			grad ~ grad + momentum(i);                          // Add momentum to the gradient
	  			momentum(i) ~ grad *@ mu;                           // update momentum using the new gradient
	  		}
	  		if (opts.nesterov.asInstanceOf[AnyRef] != null) {
	  			val i0 = if (opts.nesterov.length > 1) i else 0;
	  			mu <-- opts.nesterov(i0);                           // Get the momentum decay rate
	  			grad ~ grad + momentum(i);                          // Add momentum to the gradient
	  			mm ~ mm - momentum(i);                              // A bit of algebra, remove old momentum from the model
	  			momentum(i) ~ grad *@ mu;                           // Update the momentum
	  			mm ~ mm + momentum(i);                              // Add the new momentum to the model;
	  		}
	  		modelmats(i) ~ modelmats(i) + grad;
	  		if (mask != null) modelmats(i) ~ modelmats(i) *@ mask;
	  	}
	  }
	}
}


object Grad {
  trait Opts extends Updater.Opts {
    var lrate:FMat = 1f
    var texp:FMat = 0.5f
    var pexp:FMat = 0.5f
    var waitsteps = 3
    var mask:FMat = null
    var policies:Array[(Float, Float)=>Float] = null
    var momentum:FMat = null
    var nesterov:FMat = null
    var langevin = 0f;
  }
  
  class Options extends Opts {}
  
  def PWlinear(segments:FMat):(Float, Float) => Float = {
    (nsteps:Float, gprogress:Float) => {
      var i = 1;
      while (i < segments.nrows && gprogress > segments(i, 0)) {
        i += 1;
      }
      val frac = (gprogress - segments(i-1,0)) / (segments(i,0) - segments(i-1,0));
      frac * segments(i,1) + (1-frac) * segments(i-1,1);
    }
  }
  
   def PWexp(segments:FMat):(Float, Float) => Float = {
    (nsteps:Float, gprogress:Float) => {
      var i = 1;
      while (i < segments.nrows && gprogress > segments(i, 0)) {
        i += 1;
      }
      val frac = (gprogress - segments(i-1,0)) / (segments(i,0) - segments(i-1,0));
      math.exp(frac * math.log(segments(i,1)) + (1-frac) * math.log(segments(i-1,1))).toFloat;
    }
  }
}

