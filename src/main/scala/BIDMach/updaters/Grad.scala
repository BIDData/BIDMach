package BIDMach.updaters
 
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,FND,IMat,HMat,GMat,GIMat,GSMat,GND,ND,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._
import edu.berkeley.bid.CUMACH

class Grad(override val opts:Grad.Opts = new Grad.Options) extends Updater {
  
  var firstStep = 0f
 
  var modelmats:Array[ND] = null
  var updatemats:Array[ND] = null
  var sumSq:ND = null 
  var momentum:Array[ND] = null;
  var stepn:ND = null
  var mask:ND = null
  var ve:ND = null
	var te:ND = null
	var pe:ND = null
	var lrate:ND = null
	var mu:ND = null
	var randmat:Array[ND] = null
	var norm_scaling:ND = null

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
      momentum = new Array[ND](nmats);
      for (i <- 0 until nmats) {
    	  momentum(i) = modelmats(i).zeros(modelmats(i).nrows, modelmats(i).ncols);
      }
    }
    if (opts.langevin > 0) {
      randmat = new Array[ND](nmats);
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
  
  def clipping() {
      if (opts.clipByValue>0f) {
          var i = 0
          while (i < updatemats.length){
              min(updatemats(i),opts.clipByValue,updatemats(i));
              max(updatemats(i),-opts.clipByValue,updatemats(i));
              i+=1
          }
      }
      if (opts.max_grad_norm>0f){
        var i=0;
        var tot = 0.0
        while(i<updatemats.length){
            tot+=sum(sum(updatemats(i).asMat*@updatemats(i).asMat)).dv
            i+=1
        }
        val scale=opts.max_grad_norm/max(sqrt(tot),opts.max_grad_norm).dv
        if (norm_scaling==null) norm_scaling = updatemats(0).zeros(1,1)
        norm_scaling.set(scale.toFloat);
        i=0;
        while(i<updatemats.length){
            updatemats(i)~updatemats(i)*@norm_scaling
            i+=1
        }     
      }
  }
  
	override def update(ipass:Int, step:Long, gprogress:Float):Unit = {
	    clipping()
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
	  			mu.set(opts.momentum(i0));                           // Get the momentum decay rate
	  			grad ~ grad + momentum(i);                          // Add momentum to the gradient
	  			momentum(i) ~ grad *@ mu;                           // update momentum using the new gradient
	  		}
	  		if (opts.nesterov.asInstanceOf[AnyRef] != null) {
	  			val i0 = if (opts.nesterov.length > 1) i else 0;
	  			mu.set(opts.nesterov(i0));                           // Get the momentum decay rate
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
    var texp:FND = 0.5f
    var pexp:FMat = 0.5f
    var waitsteps = 3
    var mask:FMat = null
    var policies:Array[(Float, Float)=>Float] = null
    var momentum:FMat = null
    var nesterov:FMat = null
    var langevin = 0f;
    var clipByValue = -1f
    var max_grad_norm = -1f
  }
  
  class Options extends Opts {}
  
  
  def multUpdate(a:Mat, b:Mat, mm:Mat, mask:Mat, lrate:Mat, texp:Mat, step:Float, limit:Float):Unit = 
    multUpdate(a, b, mm, mask, lrate, texp, step, limit, false);
  
  def multUpdate(a:Mat, b:Mat, mm:Mat, mask:Mat, lrate:Mat, texp:Mat, step:Float, limit:Float, hasBias:Boolean):Unit = {
  		val istep = 1f/step;
  		val nr = a.nrows;
  		val nc = b.ncols;
  		val nbr = b.nrows;
  		val biasv = if (hasBias) 1 else 0;
  		val te = texp + 0f;
  		te.set(istep);
  		te ~ te ^ texp;
  		val lr = lrate ∘ te;
  		(a, b, mm, lr) match {
  		case (ga:GMat, gb:GSMat, gmm:GMat, glr:GMat) => {
  			val maskdata = if (mask != null) mask.asInstanceOf[GMat].data else null;
  			val masknr = if (mask != null) mask.nrows else 0; 
  			CUMACH.multGradTile(nr, nc, 0, 0, b.nnz, ga.data, a.nrows, gb.data, gb.ir, gb.ic, 
  			    gmm.data, maskdata, masknr, glr.data, lr.length, limit, biasv, nbr);
  		}
  		case (ga:GMat, gb:GSMat, tmm:TMat, glr:GMat) => {
  			for (i <- 0 until tmm.tiles.length) {
  				val tile = tmm.tiles(i).asInstanceOf[GMat];
  				val maskmat = if (mask != null) mask.asInstanceOf[TMat].tiles(i).asInstanceOf[GMat] else null;
  				val masknr = if (mask != null) maskmat.nrows else 0;
  				val maskdata = if (mask != null) maskmat.data else null;
  				CUMACH.multGradTile(tile.nrows, tile.ncols, tmm.y(i), tmm.x(i), b.nnz, ga.data, a.nrows, gb.data, gb.ir, gb.ic, 
  						tile.data, maskdata, masknr, glr.data, lr.length, limit, biasv, nbr);
  			}
  		}
  		case _ => {
  			val grad0 = mm + 0;
  			a.madd(b, grad0, false, true);
  			val grad = if (hasBias) grad0 \ sum(a,2) else grad0;
  			if (limit > 0) {
  			  min(grad, limit, grad);
  			  max(grad, -limit, grad);
  			}
  			grad ~ grad ∘ lr;
  			mm ~ mm + grad;
  		}
  		}
  }
 
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

