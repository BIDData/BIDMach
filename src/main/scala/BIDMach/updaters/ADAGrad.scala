package BIDMach.updaters
 
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._
import edu.berkeley.bid.CUMACH
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

class ADAGrad(override val opts:ADAGrad.Opts = new ADAGrad.Options) extends Updater {
  
  var firstStep = 0f
  var modelmats:Array[Mat] = null
  var updatemats:Array[Mat] = null  
  var sumSq:Array[Mat] = null 
  var stepn:Mat = null
  var mask:Mat = null
  var ve:Mat = null
  var pe:Mat = null
  var te:Mat = null
  var lrate:Mat = null
  var one:Mat = null

  override def init(model0:Model) = {
    model = model0
    modelmats = model.modelmats;
    updatemats = model.updatemats;
    val mm = modelmats(0);
    mask = opts.mask;
    val nmats = modelmats.length;
    sumSq = new Array[Mat](nmats);
    for (i <- 0 until nmats) {
    	sumSq(i) = modelmats(i).ones(modelmats(i).nrows, modelmats(i).ncols) *@ opts.initsumsq
    }
    stepn = mm.zeros(1,1);
    one = mm.ones(1,1);
    ve = mm.zeros(opts.vexp.nrows, opts.vexp.ncols);
    if (opts.texp.asInstanceOf[AnyRef] != null) te = mm.zeros(opts.texp.nrows, opts.texp.ncols);
    if (opts.pexp.asInstanceOf[AnyRef] != null) pe = mm.zeros(opts.pexp.nrows, opts.pexp.ncols);
    lrate = mm.zeros(opts.lrate.nrows, 1);
    ve <-- opts.vexp;
    te <-- opts.texp;
  } 
  
  def update2(ipass:Int, step:Long):Unit = {
  	modelmats = model.modelmats;
  	updatemats = model.updatemats;
  	val nsteps = if (step == 0) 1f else {
  		if (firstStep == 0f) {
  			firstStep = step;
  			1f;
  		} else {
  			step / firstStep;
  		}
  	}
  	stepn.set(nsteps+1);
  	val nw = one / stepn;
  	val nmats = math.min(modelmats.length, updatemats.length)
  	//	println("u2 sumsq %g" format mini(sumSq(0)).dv)
  	for (i <- 0 until nmats) {
  		val um = updatemats(i);
  		val mm = modelmats(i);
  		val ss = sumSq(i);
  		if (opts.lrate.ncols > 1) {
  			lrate <-- opts.lrate(?,i);
  		} else {
  			lrate <-- opts.lrate;
  		}
  		val newsquares = um *@ um;
  		newsquares ~ newsquares *@ nw;
  		ss  ~ ss *@ (one - nw);
  		ss ~ ss + newsquares;
  		if (opts.waitsteps < nsteps) {
  			val tmp = ss ^ ve;
  			if (java.lang.Double.isNaN(sum(sum(tmp)).dv)) throw new RuntimeException("ADA0 1 "+i);
  			tmp ~ tmp *@ (stepn ^ te);
  			if (java.lang.Double.isNaN(sum(sum(tmp)).dv)) throw new RuntimeException("ADA0 2 "+i);
  			tmp ~ tmp + opts.epsilon;
  			mm ~ mm + ((um / tmp) *@ lrate);
  			if (java.lang.Double.isNaN(sum(sum(mm)).dv)) throw new RuntimeException("ADA0 3 "+i);
  			if (mask != null) mm ~ mm *@ mask;
  		}
  	}
  }
	
  def update(ipass:Int, step:Long):Unit = { 
    modelmats = model.modelmats
    updatemats = model.updatemats
    val nsteps = if (step == 0) 1f else {
      if (firstStep == 0f) {
        firstStep = step;
        1f;
      } else {
      	step / firstStep;
      }
    }
    val tscale = if (opts.texp.asInstanceOf[AnyRef] != 0) {
      stepn.set(nsteps+1);
      stepn ^ te;
    } else {
      stepn.set(ipass+1);
      stepn ^ pe;
    }
    val nw = one / stepn;
    val nmats = math.min(modelmats.length, updatemats.length)
//    println("u sumsq %g" format mini(sumSq(0)).dv)
    for (i <- 0 until nmats) {
    	val mm = modelmats(i);
    	val um = updatemats(i);
    	val ss = sumSq(i);
    	if (opts.lrate.ncols > 1) {
    		lrate <-- opts.lrate(?,i);
    	} else {
    		lrate <-- opts.lrate;
    	}
    	val newsquares = um *@ um;
    	newsquares ~ newsquares *@ nw;
    	ss ~ ss *@ (one - nw);
    	ss ~ ss + newsquares;
    	if (opts.waitsteps < nsteps) {
    		if (java.lang.Double.isNaN(sum(sum(ss)).dv)) throw new RuntimeException("ADAGrad NaN in sumsquares matrix "+i);
    		val tmp = ss ^ ve;
    		if (java.lang.Double.isNaN(sum(sum(tmp)).dv)) throw new RuntimeException("ADAGrad NaN in powered sumsquares matrix "+i);
    		tmp ~ tmp *@ tscale;
//    		if (java.lang.Double.isNaN(sum(sum(tmp)).dv)) throw new RuntimeException("ADAGrad NaN in scaled sumsquares matrix "+i);
    		tmp ~ tmp + opts.epsilon;
    		tmp ~ um / tmp;
    		if (java.lang.Double.isNaN(sum(sum(tmp)).dv)) throw new RuntimeException("ADAGrad NaN in gradient quotient in derivative "+i);
    		tmp ~ tmp *@ lrate;
    		mm ~ mm + tmp;
    		if (mask != null) mm ~ mm *@ mask;
    	}
    }
  }
}


object ADAGrad {
  trait Opts extends Grad.Opts {
    var vexp:FMat = 0.5f
    var epsilon = 1e-5f
    var initsumsq = 1e-5f
  }
  
  class Options extends Opts {}
  
  
  def multUpdateHelperT(a:FMat, b:SMat, mm:FMat, ssq:FMat, mask:FMat, lrate:FMat, vexp:FMat, texp:FMat, 
      istep:Float, addgrad:Int, epsilon:Float, ithread:Int, numThreads:Int) = {
  	val nr = a.nrows;
  	val lrdim = lrate.length;
  	val vedim = vexp.length;
  	val tedim = texp.length;
  	val istart = (1L*ithread*nr/numThreads).toInt;
  	val iend = (1L*(ithread+1)*nr/numThreads).toInt;
  	val ioff = Mat.ioneBased;
  	var i = 0;
    while (i < b.ncols) {
    	var j = b.jc(i) - ioff;
    	while (j < b.jc(i+1)-ioff) {
    		val dval = b.data(j);
    		val ival = b.ir(j) - ioff;
    		var k = istart;
    		while (k < iend) {
    			val grad = a.data(k+i*nr)*dval;
    			ssq.data(k+ival*nr) += grad*grad + epsilon;
    			if (addgrad > 0) {
    				val lr = if (lrdim > 1) lrate.data(k) else lrate.data(0);
    				val ve = if (vedim > 1) vexp.data(k) else vexp.data(0);
    				val te = if (tedim > 1) texp.data(k) else texp.data(0);
    				val pve = if (ve == 0) 1f else math.pow(ssq.data(k+ival*nr) * istep, ve).toFloat;
    				val ste = math.pow(istep, te).toFloat;
    				val ngrad = grad * lr * ste / pve;
    				mm.data(k+ival*nr) += ngrad;
    			}
    			k += 1;
    		}
    		if (mask.asInstanceOf[AnyRef] != null) {
    		  k = istart;
    		  if (mask.nrows == 1) {
    		  	while (k < iend) {
    		  		mm.data(k+ival*nr) *= mask.data(ival);
    		  		k += 1;
    		  	} 
    		  } else {
    		  	while (k < iend) {
    		  		mm.data(k+ival*nr) *= mask.data(k+ival*nr);
    		  		k += 1;
    		  	}
    		  }
    		}
    		j += 1;
    	}
    	i += 1;
    }
  }
  
  /**
   * Integrate the last stage of a gradient update (sparse, transposed multiply) with ADAGRAD. 
   * Supports both CPU and GPU implementation.
   */
  def multUpdate(a:Mat, b:Mat, mm:Mat, sumSq:Mat, mask:Mat, lrate:Mat, texp:Mat, vexp:Mat, eps:Float, step:Float, waitsteps:Int) = {
    val istep = 1f/step;
    val addgrad = if (step > waitsteps - 0.5f) 1 else 0;
    val nr = a.nrows;
    val nc = b.ncols;
    Mat.nflops += 2L * nr * b.nnz;
    (a, b, mm, sumSq, lrate, texp, vexp) match {
      case (fa:FMat, sb:SMat, fmm:FMat, fssq:FMat, flrate:FMat, ftexp:FMat, fvexp:FMat) => {
        val fmask = mask.asInstanceOf[FMat];
      	if (1L*nr*b.nnz > 100000L && Mat.numThreads > 1) {
    			(0 until Mat.numThreads).par.map((ithread:Int) => 
    			  multUpdateHelperT(fa, sb, fmm, fssq, fmask, flrate, ftexp, fvexp, istep, addgrad, eps, ithread, Mat.numThreads));
    		} else {
    			multUpdateHelperT(fa, sb, fmm, fssq, fmask, flrate, ftexp, fvexp, istep, addgrad, eps, 0, 1);
    		}
      }
      case (ga:GMat, gsb:GSMat, gmm:GMat, gssq:GMat, glrate:GMat, gtexp:GMat, gvexp:GMat) => {
        val gmask0 = mask.asInstanceOf[GMat];
        val gmaskdata = if (gmask0.asInstanceOf[AnyRef] != null) gmask0.data else new jcuda.Pointer();
        val masknr = if (gmask0.asInstanceOf[AnyRef] != null) gmask0.nrows else 0;
        CUMACH.multADAGrad(nr, nc, b.nnz, ga.data, gsb.data, gsb.ir, gsb.ic, gmm.data, gssq.data, gmaskdata, masknr,
            glrate.data, lrate.nrows, gvexp.data, vexp.nrows, gtexp.data, texp.nrows, istep, addgrad, eps)
      }
      case _ => {
        val grad = a *^ b;
        sumSq ~ sumSq + (grad ∘ grad);
        sumSq ~ sumSq + eps;
        val ssq = sumSq + 0f;
        ssq ~ ssq * istep;
        ssq ~ ssq ^ vexp;
        val te = texp + 0f;
        te.set(istep);
        te ~ te ^ texp;
        grad ~ grad ∘ lrate;
        grad ~ grad ∘ te;
        grad ~ grad / ssq;
        mm ~ mm + grad;
      }
    }    
  }
  
  
  /**
   * Integrate the last stage of a gradient update (sparse, transposed multiply) with ADAGRAD. 
   * Supports both CPU and GPU implementation.
   */
  def hashmultUpdate(a:Mat, b:Mat, nfeats:Int, bound1:Int, bound2:Int, transpose:Int,
  		mm:Mat, sumSq:Mat, mask:Mat, lrate:Mat, texp:Mat, vexp:Mat, eps:Float, step:Float, waitsteps:Int) = {
    val istep = 1f/step;
    val addgrad = if (step > waitsteps - 0.5f) 1 else 0;
    val nr = a.nrows;
    val nc = b.ncols;
    val npc = b.nnz / b.ncols;
    Mat.nflops += 2L * nr * b.nnz * npc;
    (a, b, mm, sumSq, lrate, texp, vexp) match {
      case (fa:FMat, sb:SMat, fmm:FMat, fssq:FMat, flrate:FMat, ftexp:FMat, fvexp:FMat) => {
        val fmask = mask.asInstanceOf[FMat];
      	if (1L*nr*b.nnz > 100000L && Mat.numThreads > 1) {
    			(0 until Mat.numThreads).par.map((ithread:Int) => 
    			  multUpdateHelperT(fa, sb, fmm, fssq, fmask, flrate, ftexp, fvexp, istep, addgrad, eps, ithread, Mat.numThreads));
    		} else {
    			multUpdateHelperT(fa, sb, fmm, fssq, fmask, flrate, ftexp, fvexp, istep, addgrad, eps, 0, 1);
    		}
      }
      case (ga:GMat, gsb:GSMat, gmm:GMat, gssq:GMat, glrate:GMat, gtexp:GMat, gvexp:GMat) => {
        val gmask0 = mask.asInstanceOf[GMat];
        val gmaskdata = if (gmask0.asInstanceOf[AnyRef] != null) gmask0.data else new jcuda.Pointer();
        val masknr = if (gmask0.asInstanceOf[AnyRef] != null) gmask0.nrows else 0;
        val err = CUMACH.hashmultADAGrad(nr, nfeats, nc, bound1,  bound2, ga.data, gsb.data, gsb.ir, gsb.jc, transpose,
            gmm.data, gssq.data, gmaskdata, masknr, glrate.data, lrate.nrows, gvexp.data, vexp.nrows, gtexp.data, texp.nrows, istep, addgrad, eps)
	if (err != 0) {
	    throw new RuntimeException("hashMultUpdate error " + jcuda.runtime.JCuda.cudaGetErrorString(err));
	}
      }
    }    
  }
}

