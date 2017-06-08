package BIDMach.updaters
 
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._
import edu.berkeley.bid.CUMACH
import edu.berkeley.bid.CPUMACH
import jcuda.runtime.JCuda._
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

class ADAGrad(override val opts:ADAGrad.Opts = new ADAGrad.Options) extends Grad {
  
	var sumSq:Array[Mat] = null;
  var ve:Mat = null;
  var one:Mat = null;

  override def init(model0:Model) = {
    initGrad(model0);
    val nmats = modelmats.length;
    val mm = modelmats(0);
    sumSq = new Array[Mat](nmats);
    for (i <- 0 until nmats) {
    	sumSq(i) = modelmats(i).ones(modelmats(i).dims) *@ opts.initsumsq
    }
    ve = mm.zeros(opts.vexp.nrows, opts.vexp.ncols);
    one = mm.ones(1,1);
  } 

	
  override def update(ipass:Int, step:Long, gprogress:Float):Unit = { 
    val start = toc;
    clipping()
    ve <-- opts.vexp;
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
    tscale = if (opts.texp.asInstanceOf[AnyRef] != null) {
    	te <-- opts.texp;
    	stepn.set(1/(nsteps+1));
    	stepn ^ te;
    } else if (opts.pexp.asInstanceOf[AnyRef] != null) {
    	pe <-- opts.pexp;
    	stepn.set(1f/(ipass+1));
    	stepn ^ pe;
    } else {
      stepn.set(1f)
    }
    if (opts.gsq_decay >= 0){
    	stepn.set(1f - opts.gsq_decay);
    } else {
    	stepn.set(1/(nsteps+1));
    }
    val nw = stepn;
    val nmats = math.min(modelmats.length, updatemats.length);
//    println("u sumsq %g" format mini(sumSq(0)).dv)
    val lr0 = if (opts.policies.asInstanceOf[AnyRef] != null) opts.policies(0)(ipass, nsteps, gprogress) else 0;
    for (i <- 0 until nmats) {
    	val mm = modelmats(i);
    	val um = updatemats(i);
    	if (opts.l2reg.asInstanceOf[AnyRef] != null) {
    		val i0 = if (opts.l2reg.length > 1) i else 0;
    		um ~ um - (mm *@ opts.l2reg(i0));
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
    	val ss = sumSq(i);
    	val lr_scales = model.lr_scales;
    	if (lr_scales.asInstanceOf[AnyRef] != null) {
    	  lrate ~ lrate *@ lr_scales(i);
    	}
    	(mm, um, ss, ve, tscale, lrate) match {
    	  case (gmm:GMat, gum:GMat, gss:GMat, gve:GMat, gts:GMat, glrate:GMat) => {
          if (opts.vel_decay.asInstanceOf[AnyRef] != null) {
            val mu = if (opts.vel_decay.length > 1) opts.vel_decay(i) else opts.vel_decay(0);
            ADAGrad.ADAGradm(gmm, gum, gss, momentum(i).asInstanceOf[GMat], mu, mask.asInstanceOf[GMat], nw.dv.toFloat, gve, gts, glrate, opts.langevin, opts.epsilon, (opts.waitsteps < nsteps));
          } else if (opts.nesterov_vel_decay.asInstanceOf[AnyRef] != null) {
            val mu = if (opts.nesterov_vel_decay.length > 1) opts.nesterov_vel_decay(i) else opts.nesterov_vel_decay(0);
            ADAGrad.ADAGradn(gmm, gum, gss, momentum(i).asInstanceOf[GMat], mu, mask.asInstanceOf[GMat], nw.dv.toFloat, gve, gts, glrate, opts.langevin, opts.epsilon, (opts.waitsteps < nsteps));
          } else {
        	  ADAGrad.ADAGradx(gmm, gum, gss, mask.asInstanceOf[GMat], nw.dv.toFloat, gve, gts, glrate, opts.langevin, opts.epsilon, (opts.waitsteps < nsteps));
          }
    	  }
    	  case _ => {
    	  	val newsquares = um *@ um;
    	  	newsquares ~ newsquares *@ nw;
    	  	ss ~ ss *@ (one - nw);
    	  	ss ~ ss + newsquares;
    	  	if (opts.waitsteps < nsteps) {
    	  		// if (java.lang.Double.isNaN(sum(sum(ss)).dv)) throw new RuntimeException("ADAGrad NaN in sumsquares matrix "+i);
    	  		val grad = ss ^ ve;
    	  		// if (java.lang.Double.isNaN(sum(sum(grad)).dv)) throw new RuntimeException("ADAGrad NaN in scaled sumsquares matrix "+i);
    	  		grad ~ grad + opts.epsilon;
    	  		grad ~ um / grad;                                      // Normalized gradient
            if (opts.langevin > 0) {                               // Add Langevin random permutations
              normrnd(0, opts.langevin, randmat(i));
              grad ~ grad + randmat(i);
            }
    	  		// if (java.lang.Double.isNaN(sum(sum(grad)).dv)) throw new RuntimeException("ADAGrad NaN in gradient quotient in derivative "+i);
    	  		grad ~ grad *@ (tscale *@ lrate);                                   // Basic scaled gradient
            if (opts.vel_decay.asInstanceOf[AnyRef] != null) {
              val i0 = if (opts.vel_decay.length > 1) i else 0;
              mu <-- opts.vel_decay(i0);                           // Get the momentum decay rate      
            	momentum(i) ~ momentum(i) *@ mu;                     // Memory-efficient version of p = mu * p + grad
            	momentum(i) ~ momentum(i) + grad;
            	grad <-- momentum(i);
            }
            if (opts.nesterov_vel_decay.asInstanceOf[AnyRef] != null) {
              val i0 = if (opts.nesterov_vel_decay.length > 1) i else 0;
              mu <-- opts.nesterov_vel_decay(i0);                  // Implement x_t = x_t-1 + p_t + mu * (p_t - p_t-1)
              momentum(i) ~ momentum(i) *@ mu;                     // Compute mu * p_t-1
              mm ~ mm - momentum(i);                               // Subtract mu * p_t-1 from the model
              momentum(i) ~ momentum(i) + grad;        	           // p_t = mu * p_t-1 + g
            	mm ~ mm + momentum(i);                               // Add p_t to the model;
            	grad ~ momentum(i) *@ mu;                            // grad = mu p_t is ready to be added. 
            }
            mm ~ mm + grad;                                        // Add full gradient to the model
    	  		if (mask != null) mm ~ mm *@ mask;
    	  	}
    	  }
    	}
    	um.clear
    }
    runningtime += toc - start;
  }
}


object ADAGrad {
  trait Opts extends Grad.Opts {
    var vexp:FMat = 0.5f;
    var gsq_decay = -1f;
    var epsilon = 1e-5f;
    var initsumsq = 1e-5f;
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
  
  def multUpdate(a:Mat, b:Mat, mm:Mat, sumSq:Mat, mask:Mat, lrate:Mat, vexp:Mat, texp:Mat, eps:Float, step:Float, waitsteps:Int):Unit = 
    multUpdate(a, b, mm, sumSq, mask, lrate, vexp, texp, eps, step, waitsteps, false);
  
  def multUpdate(a:Mat, b:Mat, mm:Mat, sumSq:Mat, mask:Mat, lrate:Mat, vexp:Mat, texp:Mat, eps:Float, step:Float, waitsteps:Int, hasBias:Boolean):Unit = {
    val istep = 1f/step;
    val addgrad = if (step > waitsteps - 0.5f) 1 else 0;
    val nr = a.nrows;
    val nc = b.ncols;
    val nbr = b.nrows;
    val biasv = if (hasBias) 1 else 0;
    (a, b, mm, sumSq, lrate, vexp, texp) match {
      case (ga:GMat, gsb:GSMat, gmm:GMat, gssq:GMat, glrate:GMat, gvexp:GMat, gtexp:GMat) => {
      	Mat.nflops += 20L * nr * b.nnz;
        val gmask0 = mask.asInstanceOf[GMat];
        val gmaskdata = if (gmask0.asInstanceOf[AnyRef] != null) gmask0.pdata else new jcuda.Pointer();
        val masknr = if (gmask0.asInstanceOf[AnyRef] != null) gmask0.nrows else 0;
        val err = CUMACH.multADAGrad(nr, nc, b.nnz, ga.pdata, gsb.pdata, gsb.pir, gsb.pic, gmm.pdata, gssq.pdata, gmaskdata, masknr,
            glrate.pdata, lrate.nrows, gvexp.pdata, vexp.nrows, gtexp.pdata, texp.nrows, istep, addgrad, eps, biasv, nbr);
        if (err > 0) {
          throw new RuntimeException("multADAgrad error " + cudaGetErrorString(err))
        }
      }
      case (ga:GMat, gsb:GSMat, gmm:TMat, gssq:TMat, glrate:GMat, gvexp:GMat, gtexp:GMat) => {
      	Mat.nflops += 20L * nr * b.nnz;
//	println("istep=%f" format istep);
        val gmask0 = mask.asInstanceOf[GMat];
        val gmaskdata = if (gmask0.asInstanceOf[AnyRef] != null) gmask0.pdata else new jcuda.Pointer();
        val masknr = if (gmask0.asInstanceOf[AnyRef] != null) gmask0.nrows else 0;
        for (i <- 0 until gmm.tiles.length) {
          val mmtile = gmm.tiles(i).asInstanceOf[GMat];
          val ssqtile = gssq.tiles(i).asInstanceOf[GMat];
          val nr = mmtile.nrows;
          val nc = mmtile.ncols;
          val y = gmm.y(i);
          val x = gmm.x(i);
          val err = CUMACH.multADAGradTile(nr, nc, y, x, gsb.nnz, ga.pdata, ga.nrows, gsb.pdata, gsb.pir, gsb.pic, mmtile.pdata, ssqtile.pdata, gmaskdata, masknr,
          		glrate.pdata, lrate.nrows, gvexp.pdata, vexp.nrows, gtexp.pdata, texp.nrows, istep, addgrad, eps, biasv, nbr);
          if (err > 0) {
          throw new RuntimeException("multADAgrad error " + cudaGetErrorString(err))
        }
        }
      } 
      case (fa:FMat, sb:SMat, fmm:FMat, fssq:FMat, flrate:FMat, fvexp:FMat, ftexp:FMat) => {
      	Mat.nflops += 20L * nr * b.nnz;
        val fmask = mask.asInstanceOf[FMat];
        val masknr = if (fmask.asInstanceOf[AnyRef] != null) fmask.nrows else 0;
        CPUMACH.multADAGrad(nr, nc, b.nnz, fa.data, sb.data, sb.ir, sb.jc, fmm.data, fssq.data, if (fmask != null) fmask.data else null, masknr, 
            flrate.data, flrate.nrows, fvexp.data, fvexp.nrows, ftexp.data, ftexp.nrows, istep, addgrad, eps, biasv, nbr);
      }
      case (fa:FMat, sb:SMat, fmm:TMat, fssq:TMat, flrate:FMat, fvexp:FMat, ftexp:FMat) => {
      	Mat.nflops += 20L * nr * b.nnz;
        val fmask = mask.asInstanceOf[FMat];
        val masknr = if (fmask.asInstanceOf[AnyRef] != null) fmask.nrows else 0;
        for (i <- 0 until fmm.tiles.length) {
        	val mmtile = fmm.tiles(i).asInstanceOf[FMat];
        	val ssqtile = fssq.tiles(i).asInstanceOf[FMat];
        	val nr = mmtile.nrows;
        	val nc = mmtile.ncols;
        	val y = fmm.y(i);
        	val x = fmm.x(i);
        	CPUMACH.multADAGradTile(nr, nc, y, x, b.nnz, fa.data, fa.nrows, sb.data, sb.ir, sb.jc, mmtile.data, ssqtile.data, if (fmask != null) fmask.data else null, masknr, 
        			flrate.data, flrate.nrows, fvexp.data, fvexp.nrows, ftexp.data, ftexp.nrows, istep, addgrad, eps, biasv, nbr);
        }
      }
      case _ => {
        val grad0 = mm match {
          case tmm:TMat => mm + 0f;
          case _ => mm.view(mm.nrows, mm.ncols - (if (hasBias) 1 else 0)) + 0;
        }
        grad0.clear;
        a.madd(b, grad0, false, true);
        val grad = if (hasBias) grad0 \ sum(a,2) else grad0;
        val ssq = grad ∘ grad;
        ssq ~ ssq ∘ istep;
        sumSq ~ sumSq ∘ (1f - istep);
        sumSq ~ sumSq + ssq;
        ssq ~ sumSq ^ vexp;
        grad ~ grad / ssq;
        val te = texp + 0f;
        te.set(istep);
        te ~ te ^ texp;
        grad ~ grad ∘ (lrate ∘ te);
        mm ~ mm + grad;
      }
    }    
  }
  
  def pairMultUpdate(a:Mat, b:Mat, mm:Mat, sumSq:Mat, mask:Mat, lrate:Mat, vexp:Mat, texp:Mat, eps:Float, step:Float, waitsteps:Int, hasBias:Boolean):Unit = {
    val istep = 1f/step;
    val addgrad = if (step > waitsteps - 0.5f) 1 else 0;
    val biasv = if (hasBias) 1 else 0;
    (a, b, mm, sumSq, lrate, vexp, texp) match {
    case (ga:GMat, gsb:GSMat, gmm:GMat, gssq:GMat, glrate:GMat, gvexp:GMat, gtexp:GMat) => {
      val nr = a.nrows;   // = mm.nrows
      val nc = a.ncols;   // = b.ncols
      val nbr = b.nrows;
      val nfeats = mm.ncols/2;
    	Mat.nflops += 20L * nr * b.nnz;
    	if (nr != mm.nrows || nc != b.ncols) {
    	  throw new RuntimeException("pairMultUpdate: dimensions mismatch");
    	}
    	val (gmdata, masklen) = if (mask.asInstanceOf[AnyRef] != null) (mask.asInstanceOf[GMat].pdata, mask.length) else (null, 0);
    	val err = CUMACH.pairMultADAGradTile(nr, nc, nfeats, nfeats, ga.pdata, nr, 0, 0, gsb.pdata, gsb.pir, gsb.pjc, 0, 0, 1, gmm.pdata, mm.nrows, 
    	    gssq.pdata, gmdata, masklen, glrate.pdata, lrate.length, gvexp.pdata, vexp.length, gtexp.pdata, texp.length,
    	    istep, 1, eps);
    	if (err > 0) {
          throw new RuntimeException("pairMultADAgrad error " + cudaGetErrorString(err))
    	}
    }
    case (ga:GMat, gsb:GSMat, gmm:TMat, gssq:TMat, glrate:GMat, gvexp:GMat, gtexp:GMat) => {
    	Mat.nflops += 20L * a.nrows * b.nnz;
    	for (i <- 0 until gmm.tiles.length) {
    		val mmtile = gmm.tiles(i).asInstanceOf[GMat];
    		val ssqtile = gssq.tiles(i).asInstanceOf[GMat];
    		val nr = mmtile.nrows;
    		val nc = b.ncols;
    		val nfeats = mmtile.ncols/2;
    		val y = gmm.y(i);
    		val x = gmm.x(i);
    		if (y < 0 || y + nr > a.nrows || x < 0 || nc > b.ncols) {
    		  throw new RuntimeException("pairMultUpdate: tile strays outside matrix dimensions");
    		}
    		val (gmdata, masklen) = if (mask.asInstanceOf[AnyRef] != null) (mask.asInstanceOf[GMat].pdata, mask.length) else (null, 0);
    		val err = CUMACH.pairMultADAGradTile(nr, nc, nfeats, nfeats, ga.pdata, a.nrows, y, 0, gsb.pdata, gsb.pir, gsb.pjc, x, 0, 1, 
    		    mmtile.pdata, mmtile.nrows, ssqtile.pdata, gmdata, masklen, glrate.pdata, lrate.length, 
    		    gvexp.pdata, vexp.length, gtexp.pdata, texp.length,	istep, 1, eps);
    		if (err > 0) {
          throw new RuntimeException("pairMultADAgrad error " + cudaGetErrorString(err))
    	}
    	}
    }
    case (fa:FMat, fsb:SMat, fmm:FMat, fssq:FMat, flrate:FMat, fvexp:FMat, ftexp:FMat) => {
      val nr = a.nrows;
      val nc = b.ncols;
      val nbr = b.nrows;
      val nfeats = mm.ncols/2;
    	Mat.nflops += 20L * nr * b.nnz;
    	if (nr != mm.nrows || nc != a.ncols) {
    	  throw new RuntimeException("pairMultUpdate: dimensions mismatch");
    	}
    	val (fmdata, masklen) = if (mask.asInstanceOf[AnyRef] != null) (mask.asInstanceOf[FMat].data, mask.length) else (null, 0);
    	CPUMACH.pairMultADAGradTile(nr, nc, nfeats, nfeats, fa.data, nr, 0, 0, fsb.data, fsb.ir, fsb.jc, 0, 0, fmm.data, mm.nrows, 
    	    fssq.data, fmdata, masklen, flrate.data, lrate.length, fvexp.data, vexp.length, ftexp.data, texp.length,
    	    istep, 1, eps, biasv, 0);
    }
    case (fa:FMat, fsb:SMat, fmm:TMat, fssq:TMat, flrate:FMat, fvexp:FMat, ftexp:FMat) => {
    	Mat.nflops += 20L * a.nrows * b.nnz;
    	for (i <- 0 until fmm.tiles.length) {
    		val mmtile = fmm.tiles(i).asInstanceOf[FMat];
    		val ssqtile = fssq.tiles(i).asInstanceOf[FMat];
    		val nr = mmtile.nrows;
    		val nc = a.ncols;
    		val nfeats = mmtile.ncols/2;
    		val y = fmm.y(i);
    		val x = fmm.x(i);
    		if (y < 0 || y + nr > a.nrows || x < 0 || nc > b.ncols) {
    		  throw new RuntimeException("pairMultUpdate: tile strays outside matrix dimensions");
    		}
    		val (gmdata, masklen) = if (mask.asInstanceOf[AnyRef] != null) (mask.asInstanceOf[FMat].data, mask.length) else (null, 0);
    		CPUMACH.pairMultADAGradTile(nr, nc, nfeats, nfeats, fa.data, y, 0, nr, fsb.data, fsb.ir, fsb.jc, x, 0, 
    		    mmtile.data, mm.nrows, ssqtile.data, gmdata, masklen, flrate.data, lrate.length, 
    		    fvexp.data, vexp.length, ftexp.data, texp.length,	istep, 1, eps, biasv, 0);
    	}
    }
    }
  }
  
  /**
   * Integrate the last stage of a gradient update (sparse, transposed multiply) with ADAGRAD. 
   * Supports both CPU and GPU implementation.
   */
  def hashmultUpdate(a:Mat, b:Mat, nfeats:Int, bound1:Int, bound2:Int, transpose:Int,
  		mm:Mat, sumSq:Mat, mask:Mat, lrate:Mat, vexp:Mat, texp:Mat, eps:Float, step:Float, waitsteps:Int) = {
    val istep = 1f/step;
    val addgrad = if (step > waitsteps - 0.5f) 1 else 0;
    val nr = a.nrows;
    val nc = b.ncols;
    val npc = b.nnz / b.ncols;
    Mat.nflops += 2L * nr * b.nnz * npc;
    (a, b, mm, sumSq, lrate, vexp, texp) match {
      case (ga:GMat, gsb:GSMat, gmm:GMat, gssq:GMat, glrate:GMat, gvexp:GMat, gtexp:GMat) => {
        val gmask0 = mask.asInstanceOf[GMat];
        val gmaskdata = if (gmask0.asInstanceOf[AnyRef] != null) gmask0.pdata else new jcuda.Pointer();
        val masknr = if (gmask0.asInstanceOf[AnyRef] != null) gmask0.nrows else 0;
        val err = CUMACH.hashmultADAGrad(nr, nfeats, nc, bound1,  bound2, ga.pdata, gsb.pdata, gsb.pir, gsb.pjc, transpose,
        		gmm.pdata, gssq.pdata, gmaskdata, masknr, glrate.pdata, lrate.nrows, gvexp.pdata, vexp.nrows, gtexp.pdata, texp.nrows, istep, addgrad, eps);
        if (err != 0) {
        	throw new RuntimeException("hashMultUpdate error " + jcuda.runtime.JCuda.cudaGetErrorString(err));
        }
      }
      case (fa:FMat, sb:SMat, fmm:FMat, fssq:FMat, flrate:FMat, fvexp:FMat, ftexp:FMat) => {
        val fmask = mask.asInstanceOf[FMat];
      	if (1L*nr*b.nnz > 100000L && Mat.numThreads > 1) {
    			(0 until Mat.numThreads).par.map((ithread:Int) => 
    			  multUpdateHelperT(fa, sb, fmm, fssq, fmask, flrate, fvexp, ftexp, istep, addgrad, eps, ithread, Mat.numThreads));
    		} else {
    			multUpdateHelperT(fa, sb, fmm, fssq, fmask, flrate, fvexp, ftexp, istep, addgrad, eps, 0, 1);
    		}
      }
    }    
  }
  
  def ADAGradx(mm:GMat, um:GMat, ss:GMat, mask:GMat, nw:Float, ve:GMat, ts:GMat, lrate:GMat, langevin:Float, epsilon:Float, doupdate:Boolean) = {
  	val (gmask, maskr) = if (mask.asInstanceOf[AnyRef] == null) (null, 0) else (mask.pdata, mask.nrows);
  	CUMACH.ADAGrad(mm.nrows, mm.ncols, mm.pdata, um.pdata, ss.pdata, gmask, maskr, nw, ve.pdata, ve.nrows,
  			ts.pdata, ts.nrows, lrate.pdata, lrate.nrows, langevin, epsilon, if (doupdate) 1 else 0);
  }
  
  def ADAGradm(mm:GMat, um:GMat, ss:GMat, vel_decay:GMat, mu:Float, mask:GMat, nw:Float, ve:GMat, ts:GMat, lrate:GMat, langevin:Float, epsilon:Float, doupdate:Boolean) = {
    val (gmask, maskr) = if (mask.asInstanceOf[AnyRef] == null) (null, 0) else (mask.pdata, mask.nrows);
    CUMACH.ADAGradm(mm.nrows, mm.ncols, mm.pdata, um.pdata, ss.pdata, vel_decay.pdata, mu, gmask, maskr, nw, ve.pdata, ve.nrows,
        ts.pdata, ts.nrows, lrate.pdata, lrate.nrows, langevin, epsilon, if (doupdate) 1 else 0);
  }
    
  def ADAGradn(mm:GMat, um:GMat, ss:GMat, vel_decay:GMat, mu:Float, mask:GMat, nw:Float, ve:GMat, ts:GMat, lrate:GMat, langevin:Float, epsilon:Float, doupdate:Boolean) = {
    val (gmask, maskr) = if (mask.asInstanceOf[AnyRef] == null) (null, 0) else (mask.pdata, mask.nrows);
    CUMACH.ADAGradn(mm.nrows, mm.ncols, mm.pdata, um.pdata, ss.pdata, vel_decay.pdata, mu, gmask, maskr, nw, ve.pdata, ve.nrows,
        ts.pdata, ts.nrows, lrate.pdata, lrate.nrows, langevin, epsilon, if (doupdate) 1 else 0);
  }
}

