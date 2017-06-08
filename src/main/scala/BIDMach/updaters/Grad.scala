package BIDMach.updaters
 
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._
import edu.berkeley.bid.CUMACH

class Grad(override val opts:Grad.Opts = new Grad.Options) extends Updater {
  
  var firstStep = 0f
 
  var modelmats:Array[Mat] = null
  var updatemats:Array[Mat] = null
//  var sumSq:Array[Mat] = null 
  var momentum:Array[Mat] = null;
  var stepn:Mat = null
  var mask:Mat = null
	var te:Mat = null
	var pe:Mat = null
	var lrate:Mat = null
	var mu:Mat = null
	var randmat:Array[Mat] = null
	var norm_scaling:Mat = null
	var tscale:Mat = null

  def initGrad(model0:Model) = {
    firstStep = 0f;
    model = model0;
	  modelmats = model.modelmats;
	  updatemats = model.updatemats;
	  mask = opts.mask;
	  val mm = modelmats(0);
    stepn = mm.zeros(1,1);
    val nmats = modelmats.length;
    val hasvel_decay = (opts.vel_decay.asInstanceOf[AnyRef] != null || opts.nesterov_vel_decay.asInstanceOf[AnyRef] != null);
    if (hasvel_decay) {
      momentum = new Array[Mat](nmats);
      for (i <- 0 until nmats) {
    	  momentum(i) = modelmats(i).zeros(modelmats(i).dims);
      }
    }
    if (opts.langevin > 0) {
      randmat = new Array[Mat](nmats);
      for (i <- 0 until nmats) {
        randmat(i) = modelmats(i).zeros(modelmats(i).dims);
      }
    }
    if (opts.texp.asInstanceOf[AnyRef] != null) {
      te = mm.zeros(opts.texp.nrows, opts.texp.ncols);
    }
    if (opts.pexp.asInstanceOf[AnyRef] != null) {
      pe = mm.zeros(opts.pexp.nrows, opts.pexp.ncols);
    }
    lrate = mm.zeros(opts.lrate.nrows, 1);
    mu = mm.zeros(1,1);
  } 
  
  override def init(model0:Model) = initGrad(model0);
  
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
            tot+=sum(updatemats(i) dot updatemats(i)).dv
            i+=1
        }
        val scale=opts.max_grad_norm/max(sqrt(tot),opts.max_grad_norm).dv
        if (norm_scaling==null) norm_scaling = updatemats(0).zeros(1,1)
        norm_scaling(0,0) = scale.toFloat
        i=0;
        while(i<updatemats.length){
            updatemats(i)~updatemats(i)*@norm_scaling
            i+=1
        }     
      }
  }
  
	override def update(ipass:Int, step:Long, gprogress:Float):Unit = {
		val start = toc;
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
  	val lr0 = if (opts.policies.asInstanceOf[AnyRef] != null) opts.policies(0)(ipass, nsteps, gprogress) else 0;
	  for (i <- 0 until nmats) {
		  val mm = modelmats(i);
      tscale = if (te.asInstanceOf[AnyRef] != null) {
      	te <-- opts.texp;
        stepn.set(1f/nsteps);
        stepn ^ te;
      } else {
      	pe <-- opts.pexp;
        stepn.set(1f/(ipass+1));
        stepn ^ pe;
      }
      if (opts.policies.asInstanceOf[AnyRef] != null) {
			  if (opts.policies.length > 1) {
				  lrate.set(opts.policies(i)(ipass, nsteps, gprogress));
			  } else {
				  lrate.set(lr0);
			  }
		  } else {
		  	if (opts.lrate.ncols > 1) {
		  		lrate <-- opts.lrate(?,i);
		  	} else {
		  		lrate <-- opts.lrate;
		  	}
		  }
    	val lr_scales = model.lr_scales;
    	if (lr_scales.asInstanceOf[AnyRef] != null) {
    	  lrate ~ lrate *@ lr_scales(i);
    	}
	  	if (opts.waitsteps < nsteps) {
        val grad = updatemats(i);
        if (opts.l2reg.asInstanceOf[AnyRef] != null) {
        	val i0 = if (opts.l2reg.length > 1) i else 0;
        	grad ~ grad - (mm *@ opts.l2reg(i0));
        }
        if (opts.langevin > 0) {                              // Add Langevin random permutations
        	normrnd(0, opts.langevin, randmat(i));
        	grad ~ grad + randmat(i);
        }
	  		grad ~ grad *@ (lrate *@ tscale);
	  		if (opts.vel_decay.asInstanceOf[AnyRef] != null) {
	  			val i0 = if (opts.vel_decay.length > 1) i else 0;
	  			mu <-- opts.vel_decay(i0);                          // Get the momentum decay rate      
	  			momentum(i) ~ momentum(i) *@ mu;                    // update momentum using the new gradient p = mu p + grad
	  			momentum(i) ~ momentum(i) + grad;
	  			grad <-- momentum(i);
	  		}
	  		if (opts.nesterov_vel_decay.asInstanceOf[AnyRef] != null) {
	  			val i0 = if (opts.nesterov_vel_decay.length > 1) i else 0;
	  			mu <-- opts.nesterov_vel_decay(i0);                 // Get the momentum decay rate
	  			mm ~ mm - momentum(i);                              // A bit of algebra, remove old momentum from the model
	  			momentum(i) ~ momentum(i) + grad;                   // Memory-efficient version of p = mu * p + mu grad
	  			momentum(i) ~ momentum(i) *@ mu;
	  			mm ~ mm + momentum(i);                              // Add the new momentum to the model;
	  			grad ~ momentum(i) / mu;                              
	  		}
	  		modelmats(i) ~ modelmats(i) + grad;
	  		if (mask != null) modelmats(i) ~ modelmats(i) *@ mask;
	  	}
	  	updatemats(i).clear;
	  }
	  runningtime += toc - start;
	}
}


object Grad {
  trait Opts extends Updater.Opts {
  	var lrate:FMat = 1f;
    var texp:FMat = 0.5f;
    var pexp:FMat = 0f;
    var waitsteps = 3;
    var mask:FMat = null;
    var policies:Array[(Float, Float, Float)=>Float] = null;
    var vel_decay:FMat = null;
    var nesterov_vel_decay:FMat = null;
    var l2reg:FMat = null;
    var langevin = 0f;
    var clipByValue = -1f;
    var max_grad_norm = -1f;
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
  			val maskdata = if (mask != null) mask.asInstanceOf[GMat].pdata else null;
  			val masknr = if (mask != null) mask.nrows else 0; 
  			CUMACH.multGradTile(nr, nc, 0, 0, b.nnz, ga.pdata, a.nrows, gb.pdata, gb.pir, gb.pic, 
  			    gmm.pdata, maskdata, masknr, glr.pdata, lr.length, limit, biasv, nbr);
  		}
  		case (ga:GMat, gb:GSMat, tmm:TMat, glr:GMat) => {
  			for (i <- 0 until tmm.tiles.length) {
  				val tile = tmm.tiles(i).asInstanceOf[GMat];
  				val maskmat = if (mask != null) mask.asInstanceOf[TMat].tiles(i).asInstanceOf[GMat] else null;
  				val masknr = if (mask != null) maskmat.nrows else 0;
  				val maskdata = if (mask != null) maskmat.pdata else null;
  				CUMACH.multGradTile(tile.nrows, tile.ncols, tmm.y(i), tmm.x(i), b.nnz, ga.pdata, a.nrows, gb.pdata, gb.pir, gb.pic, 
  						tile.pdata, maskdata, masknr, glr.pdata, lr.length, limit, biasv, nbr);
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

