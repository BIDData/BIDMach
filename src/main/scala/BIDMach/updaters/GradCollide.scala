package BIDMach.updaters;
 
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GIMat,GDMat,GLMat,GSMat,ND,SMat,SDMat,TMat};
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._
import edu.berkeley.bid.CUMACH

class GradCollide(override val opts:GradCollide.Opts = new GradCollide.Options) extends Updater {
  
  var firstStep = 0.0;
 
  var modelmats:Array[Mat] = null;
  var updatemats:Array[Mat] = null;
  var momentum:Array[Mat] = null;
  var randmat:Array[Mat] = null;

  var modelmatsSave:Array[Mat] = null;
  var momentumSave:Array[Mat] = null;

  var swap:Mat = null;
  var x:Mat = null;
  var y:Mat = null;
  var u:Mat = null;
  var v:Mat = null;
  var pbar:Mat = null;
  var qbar:Mat = null;
  var c:Mat = null;
  var tmp:Mat = null;
  var tmat:Mat = null;
  var aelem:Mat = null;

  var stepn:Mat = null;
  var mask:Mat = null;
  var te:Mat = null;
  var pe:Mat = null;
  var lrate:Mat = null;
  var mu:Mat = null;
  var norm_scaling:Mat = null;
  var tscale:Mat = null;

  def initGrad(model0:Model) = {
    firstStep = 0;
    model = model0;
    modelmats = model.modelmats;
    updatemats = model.updatemats;
    mask = opts.mask;
    val mm = modelmats(0);
    stepn = mm.zeros(1,1);
    val nmats = modelmats.length;
    modelmatsSave = new Array[Mat](nmats);
    val hasvel_decay = (opts.vel_decay.asInstanceOf[AnyRef] != null || opts.nesterov_vel_decay.asInstanceOf[AnyRef] != null);
    if (hasvel_decay) {
      momentum = new Array[Mat](nmats);
      momentumSave = new Array[Mat](nmats);		
      for (i <- 0 until nmats) {
        if (modelmats(i).asInstanceOf[AnyRef] != null) {
	  momentum(i) = modelmats(i).zeros(modelmats(i).dims);
        }
      }
    }
    if (opts.langevin > 0) {
      randmat = new Array[Mat](nmats);
      for (i <- 0 until nmats) {
      	if (modelmats(i).asInstanceOf[AnyRef] != null) {
	  randmat(i) = modelmats(i).zeros(modelmats(i).dims);
      	}
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
      for (i <- 0 until updatemats.length){
	if (updatemats(i).asInstanceOf[AnyRef] != null) {
	  min(updatemats(i),opts.clipByValue,updatemats(i));
	  max(updatemats(i),-opts.clipByValue,updatemats(i));
	}
      }
    }
    if (opts.max_grad_norm>0f){
      var tot = 0.0;
      for (i <- 0 until updatemats.length){
	if (updatemats(i).asInstanceOf[AnyRef] != null) {
	  tot+=sum(updatemats(i) dot updatemats(i)).dv
	    }
      }
      val scale=opts.max_grad_norm/max(sqrt(tot),opts.max_grad_norm).dv;
      if (norm_scaling==null) norm_scaling = updatemats(0).zeros(1,1);
      norm_scaling(0,0) = scale.toFloat;
      for (i <- 0 until updatemats.length){
	if (updatemats(i).asInstanceOf[AnyRef] != null) {
	  updatemats(i)~updatemats(i)*@norm_scaling;
	}     
      }
    }
  }

  def getLargestMat(mat:Mat, umat:Mat) = {
    var maxlen = 0;
    val oldmat = if (mat.asInstanceOf[AnyRef] != null) {
      mat;
    } else {
      for (i <- 0 until updatemats.length) {
	if (updatemats(i).asInstanceOf[AnyRef] != null) {
	  maxlen = math.max(maxlen, updatemats(i).length);
	}
      }
      umat.zeros(1,maxlen);
    }
    oldmat match {
      case a:GMat => new GMat(umat.dims.data, a.pdata, a.realsize);
      case a:GIMat => new GIMat(umat.dims.data, a.pdata, a.realsize);
      case a:GDMat => new GDMat(umat.dims.data, a.pdata, a.realsize);
      case a:GLMat => new GLMat(umat.dims.data, a.pdata, a.realsize);

      case a:FMat => new FMat(umat.dims.data, a.data);
      case a:IMat => new IMat(umat.dims.data, a.data);
      case a:DMat => new DMat(umat.dims.data, a.data);
      case a:LMat => new LMat(umat.dims.data, a.data);
    }
  };

  def getLargestCol(mat:Mat, umat:Mat) = {
    var maxlen = 0;
    val oldmat = if (mat.asInstanceOf[AnyRef] != null) {
      mat;
    } else {
      for (i <- 0 until updatemats.length) {
	if (updatemats(i).asInstanceOf[AnyRef] != null) {
	  maxlen = math.max(maxlen, updatemats(i).nrows);
	}
      }
      umat.zeros(maxlen,1);
    }
    val newdims = umat.dims.data.clone();
    newdims(newdims.length-1) = 1;
    oldmat match {
      case a:GMat => new GMat(newdims, a.pdata, a.realsize);
      case a:GIMat => new GIMat(newdims, a.pdata, a.realsize);
      case a:GDMat => new GDMat(newdims, a.pdata, a.realsize);
      case a:GLMat => new GLMat(newdims, a.pdata, a.realsize);

      case a:FMat => new FMat(newdims, a.data);
      case a:IMat => new IMat(newdims, a.data);
      case a:DMat => new DMat(newdims, a.data);
      case a:LMat => new LMat(newdims, a.data);
    }
  }

  def swapMats(i:Int) = {
    if (modelmats(i).asInstanceOf[AnyRef] != null) {
      swap = getLargestMat(swap, modelmats(i));

      if (opts.logSwap) Mat.logger.info("before: i=%d, mm=%f, mms=%f" format (i, dotprod(modelmats(i), modelmats(i)), dotprod(modelmatsSave(i), modelmatsSave(i))));
      swap <-- modelmats(i);
      modelmats(i) <-- modelmatsSave(i);
      modelmatsSave(i) <-- swap;
      if (opts.logSwap) Mat.logger.info("after : i=%d, mm=%f, mms=%f" format (i, dotprod(modelmats(i), modelmats(i)), dotprod(modelmatsSave(i), modelmatsSave(i))));
    }    

    if (momentum(i).asInstanceOf[AnyRef] != null) {
	if (opts.logSwap) Mat.logger.info("before: i=%d, mo=%f, mos=%f" format (i, dotprod(momentum(i), momentum(i)), dotprod(momentumSave(i), momentumSave(i))));
      swap <-- momentum(i);
      momentum(i) <-- momentumSave(i);
      momentumSave(i) <-- swap;
      if (opts.logSwap) Mat.logger.info("after : i=%d, mo=%f, mos=%f" format (i, dotprod(momentum(i), momentum(i)), dotprod(momentumSave(i), momentumSave(i))));
    }
  };

  def dotprod(a:Mat, b:Mat):Float = {
    aelem ~ a.contents dot b.contents
    aelem.dv.toFloat;
  };

  // This version conserves energy individually for p and q

  def collide1(p:Mat, q:Mat, i:Int) = {
    x = getLargestMat(x, p);
    tmp = getLargestMat(tmp, p);
    c = getLargestMat(c, p);
//    y = getLargestMat(y, p);
//    u = getLargestMat(u, p);
//    v = getLargestMat(v, p);
//    pbar = getLargestMat(pbar, p);
//    qbar = getLargestMat(qbar, p);
//    tmat = getLargestMat(tmat, p);
    val epsilon = 1e-36f;
    
    if (opts.logCollide) {
      val lp = math.sqrt(dotprod(p, p) / p.length).toFloat;
      val lq = math.sqrt(dotprod(q, q) / q.length).toFloat;
      val dp = dotprod(p, q);
      tmp ~ p *@ p;
      val meansqp = dotprod(tmp, tmp) / p.length;
      tmp ~ q *@ q;
      val meansqq = dotprod(tmp, tmp) / p.length;
      val meanp = lp * lp;
      val meanq = lq * lq;
      val cosp = dp / (p.length * lp * lq + epsilon);
      Mat.logger.info("before: i=%d, cos(p,q)=%g, meanp=%g, meanq=%g, varp=%g, varq=%g" format (i, cosp, meanp, meanq, meansqp - meanp * meanp, meansqq - meanq * meanq));
    }

    normrnd(0, 1, x);

    val pp = dotprod(p, p);
    val qq = dotprod(q, q);
    val pq = dotprod(p, q);
    val rp = dotprod(p, x);
    val rq = dotprod(q, x);

    // solve for coefficients alpha, beta s.t. (r + alpha p + beta q) is orthogonal to p and q.
    val det = pp * qq - pq * pq;
    val alpha = (pq * rq - qq * rp) / (det + epsilon);
    val beta = (pq * rp - pp * rq) / (det + epsilon);

    tmp ~ p * aelem.set(alpha);
    x ~ x + tmp;

    tmp ~ q * aelem.set(beta);
    x ~ x + tmp;

    val pq2 = pp + 2*pq + qq;

    // First find a vector c to reduce the energy of p and q.
    // This c is the projection of p (or -q) normal to p+q, scaled by hardness. 
    c ~ p * aelem.set(opts.hardness * (pq + qq) / (pq2 + epsilon));
    tmp ~ q * aelem.set(- opts.hardness * (pq + pp) / (pq2 + epsilon));
    c ~ c + tmp;

    // Compute new p and q, and the reduction in energy.
    p ~ p - c;
    q ~ q + c;
    val energy = pp - dotprod(p, p);

    // Scale the random vector to have the same energy. 
    if (energy > 0) { 
      c ~ x * aelem.set(math.sqrt(energy / dotprod(x, x)).toFloat);
    } else { 
      c.set(0);
    }
    // Add/subtract the random vector.
    p ~ p - c;
    q ~ q + c;

    if (opts.logCollide) {
	//	Mat.logger.info("after: nx=%g, ny1=%g, ny2=%g, nu=%g, nv=%g, nc=%g" format (dotprod(x,x),ny1,dotprod(y,y),dotprod(u,u), dotprod(v,v), dotprod(c,c)));
      val lp = math.sqrt(dotprod(p, p) / p.length).toFloat;
      val lq = math.sqrt(dotprod(q, q) / q.length).toFloat;
      val dp = dotprod(p, q);
      tmp ~ p *@ p;
      val meansqp = dotprod(tmp, tmp) / p.length;
      tmp ~ q *@ q;
      val meansqq = dotprod(tmp, tmp) / p.length;
      val meanp = lp * lp;
      val meanq = lq * lq;
      val cosp = dp / (p.length * lp * lq + epsilon);
      Mat.logger.info("after:  i=%d, cos(p,q)=%g, meanp=%g, meanq=%g, varp=%g, varq=%g" format (i, cosp, meanp, meanq, meansqp - meanp * meanp, meansqq - meanq * meanq));
    }

  };

  def collide1x(p:Mat, q:Mat, i:Int) = {
    x = getLargestMat(x, p);
    y = getLargestMat(y, p);
    u = getLargestMat(u, p);
    v = getLargestMat(v, p);
    tmp = getLargestMat(tmp, p);
    pbar = getLargestMat(pbar, p);
    qbar = getLargestMat(qbar, p);
    c = getLargestMat(c, p);
    tmat = getLargestMat(tmat, p);
    val epsilon = 1e-36f;
    
    val lp = math.sqrt(dotprod(p, p) / p.length).toFloat;
    val lq = math.sqrt(dotprod(q, q) / q.length).toFloat;
    if (opts.logCollide) {
      val dp = dotprod(p, q);
      tmp ~ p *@ p;
      val meansqp = dotprod(tmp, tmp) / p.length;
      tmp ~ q *@ q;
      val meansqq = dotprod(tmp, tmp) / p.length;
      val meanp = lp * lp;
      val meanq = lq * lq;
      val cosp = dp / (p.length * lp * lq + epsilon);
      Mat.logger.info("before: i=%d, cos(p,q)=%g, meanp=%g, meanq=%g, varp=%g, varq=%g" format (i, cosp, meanp, meanq, meansqp - meanp * meanp, meansqq - meanq * meanq));
    }
    normrnd(0, lp, x);
    normrnd(0, lq, y);
/*    x ~ x + p;
    y ~ y + q;

    u ~ x * aelem.set(math.sqrt(1/(dotprod(x, x)+epsilon)).toFloat);
    v ~ u * aelem.set(dotprod(y, u));
    y ~ y - v;
    val ny1 = dotprod(y, y);
    v ~ y * aelem.set(math.sqrt(1/(ny1+epsilon)).toFloat);

    tmp ~ u * aelem.set(dotprod(p, u));
    pbar ~ v * aelem.set(dotprod(p, v));
    pbar ~ pbar + tmp;

    tmp ~ u * aelem.set(dotprod(q, u));
    qbar ~ v * aelem.set(dotprod(q, v));
    qbar ~ qbar + tmp; */

    pbar ~ x + p;
    qbar ~ y + q;


    tmat ~ pbar + qbar;
    tmp ~ pbar - qbar;
    val twt = dotprod(tmp, tmat)/(dotprod(tmat, tmat)+epsilon);
    c ~ tmat * aelem.set(twt);
    c ~ c - tmp;

    p ~ p + c;
    q ~ q - c;

    if (opts.logCollide) {
	//	Mat.logger.info("after: nx=%g, ny1=%g, ny2=%g, nu=%g, nv=%g, nc=%g" format (dotprod(x,x),ny1,dotprod(y,y),dotprod(u,u), dotprod(v,v), dotprod(c,c)));
      val lp = math.sqrt(dotprod(p, p) / p.length).toFloat;
      val lq = math.sqrt(dotprod(q, q) / q.length).toFloat;
      val dp = dotprod(p, q);
      tmp ~ p *@ p;
      val meansqp = dotprod(tmp, tmp) / p.length;
      tmp ~ q *@ q;
      val meansqq = dotprod(tmp, tmp) / p.length;
      val meanp = lp * lp;
      val meanq = lq * lq;
      val cosp = dp / (p.length * lp * lq + epsilon);
      Mat.logger.info("after:  i=%d, cos(p,q)=%g, meanp=%g, meanq=%g, varp=%g, varq=%g" format (i, cosp, meanp, meanq, meansqp - meanp * meanp, meansqq - meanq * meanq));
    }

  };

  // This version conserves total energy for p and q
    
  def collide2(p:Mat, q:Mat, i:Int) = {
    x = getLargestMat(x, p);
    y = getLargestMat(y, p);
    c = getLargestMat(c, p);
    tmp = getLargestMat(tmp, p);
    val epsilon = 1e-36f;

    if (opts.logCollide) {
      val meanp = dotprod(p, p) / p.length;
      val meanq = dotprod(q, q) / q.length;
      val dp = dotprod(p, q);
      tmp ~ p *@ p;
      val meansqp = dotprod(tmp, tmp) / p.length;
      tmp ~ q *@ q;
      val meansqq = dotprod(tmp, tmp) / p.length;
      val cosp = dotprod(p, q) / (p.length * math.sqrt(meanp * meanq).toFloat + epsilon);
      Mat.logger.info("before: i=%d, cos(p,q)=%g, tote=%g, meanp=%g, meanq=%g, varp=%g, varq=%g" format (i, cosp, meanp+meanq, meanp, meanq, meansqp - meanp * meanp, meansqq - meanq * meanq));
    }
    x ~ p - q;
    val lx = math.sqrt(dotprod(x, x) / x.length).toFloat;
    normrnd(0, lx, y);
    y ~ y + x;
	
    c ~ y * aelem.set((dotprod(x, y)/(dotprod(y, y)+epsilon)).toFloat);

    p ~ p - c;
    q ~ q + c;

    if (opts.logCollide) {
	//	Mat.logger.info("after: nx=%g, ny1=%g, ny2=%g, nu=%g, nv=%g, nc=%g" format (dotprod(x,x),ny1,dotprod(y,y),dotprod(u,u), dotprod(v,v), dotprod(c,c)));
      val meanp = dotprod(p, p) / p.length;
      val meanq = dotprod(q, q) / q.length;
      val dp = dotprod(p, q);
      tmp ~ p *@ p;
      val meansqp = dotprod(tmp, tmp) / p.length;
      tmp ~ q *@ q;
      val meansqq = dotprod(tmp, tmp) / p.length;
      val cosp = dotprod(p, q) / (p.length * math.sqrt(meanp * meanq).toFloat + epsilon);
      Mat.logger.info("after : i=%d, cos(p,q)=%g, tote=%g, meanp=%g, meanq=%g, varp=%g, varq=%g" format (i, cosp, meanp+meanq, meanp, meanq, meansqp - meanp * meanp, meansqq - meanq * meanq));
    }

  };


  // This version conserves total energy for p and q, allows variable collision "hardness"
    
  def collide3(p:Mat, q:Mat, i:Int) = {
    x = getLargestMat(x, p);
    y = getLargestMat(y, p);
    c = getLargestMat(c, p);
    tmp = getLargestMat(tmp, p);
    val epsilon = 1e-36f;

    if (opts.logCollide) {
      val meanp = dotprod(p, p) / p.length;
      val meanq = dotprod(q, q) / q.length;
      val dp = dotprod(p, q);
      tmp ~ p *@ p;
      val meansqp = dotprod(tmp, tmp) / p.length;
      tmp ~ q *@ q;
      val meansqq = dotprod(tmp, tmp) / p.length;
      val cosp = dotprod(p, q) / (p.length * math.sqrt(meanp * meanq).toFloat + epsilon);
      val varp = meansqp - meanp * meanp;
      val varq = meansqq - meanq * meanq;
      Mat.logger.info("before: i=%d, cos(p,q)=%g, tote=%g, meanp=%g, meanq=%g, totv=%g, varp=%g, varq=%g" format (i, cosp, meanp+meanq, meanp, meanq, meansqp+meansqq, varp, varq));
    }
    tmp ~ p - q;
    c ~ tmp * aelem.set(opts.hardness * 0.5f);
    x ~ p - c;
    y ~ q + c;
    normrnd(0, 1, c);
    tmp ~ x - y;
    // Quadratic coefficients for energy conservation
    val qa = 2 * dotprod(c, c);
    val qb = 2 * dotprod(c, tmp);
    val qc = dotprod(x,x) + dotprod(y,y) - dotprod(p,p) - dotprod(q,q);
    val discr = qb*qb - 4*qa*qc;
    if (discr >= 0) {
	// Quadratic is solvable (should be for any hardness in [0,1]) so solve it.
	val beta = if (Mat.myrand.nextFloat() < 0.5f) {
	    (-qb + math.sqrt(discr).toFloat)/(2*qa+epsilon);
	} else {
	    (-qb - math.sqrt(discr).toFloat)/(2*qa+epsilon);
	}	    
	c ~ c * aelem.set(beta);
	p ~ x + c;
	q ~ y - c;

	if (opts.logCollide) {
	    //	Mat.logger.info("after: nx=%g, ny1=%g, ny2=%g, nu=%g, nv=%g, nc=%g" format (dotprod(x,x),ny1,dotprod(y,y),dotprod(u,u), dotprod(v,v), dotprod(c,c)));
	    val meanp = dotprod(p, p) / p.length;
	    val meanq = dotprod(q, q) / q.length;
	    val dp = dotprod(p, q);
	    tmp ~ p *@ p;
	    val meansqp = dotprod(tmp, tmp) / p.length;
	    tmp ~ q *@ q;
	    val meansqq = dotprod(tmp, tmp) / p.length;
	    val cosp = dotprod(p, q) / (p.length * math.sqrt(meanp * meanq).toFloat + epsilon);
	    val varp = meansqp - meanp * meanp;
	    val varq = meansqq - meanq * meanq;
	    Mat.logger.info("after:  i=%d, cos(p,q)=%g, tote=%g, meanp=%g, meanq=%g, totv=%g, varp=%g, varq=%g" format (i, cosp, meanp+meanq, meanp, meanq, meansqp+meansqq, varp, varq));
	}
    }

  };

  def collide(p:Mat, q:Mat, i:Int) = {
      if (opts.perParticleMomentum) {
	  collide1(p, q, i);
      } else {
	  collide3(p, q, i);
      }
  };      

  def attract(p:Mat, q:Mat, afactor:Float, i:Int) = {
    u = getLargestMat(u, p);
    u ~ p - q;
    val pm = if (opts.logAttract) dotprod(p,p) else 0f;
    val qm = if (opts.logAttract) dotprod(q,q) else 0f;
    val um = if (opts.logAttract) dotprod(u,u) else 0f;
    u ~ u * aelem.set(0.5f * afactor);
    p ~ p - u;
    q ~ q + u;
    if (opts.logAttract) {
	val pm2 = dotprod(p,p);
	val qm2 = dotprod(q,q);
	Mat.logger.info("attract %d pm %g, qm %g, um %g, pm %g, qm %g" format (i, pm, qm, um, pm2, qm2));
    }	    
  }

  def checkSwapMats(i:Int) = {
      if (modelmats(i).asInstanceOf[AnyRef] != null) {
	  if (modelmatsSave(i).asInstanceOf[AnyRef] == null) {
	      modelmatsSave(i) = modelmats(i).zeros(modelmats(i).dims);
	      modelmatsSave(i) <-- modelmats(i);
	  }
	  if (momentum(i).asInstanceOf[AnyRef] != null && momentumSave(i).asInstanceOf[AnyRef] == null) {
	      momentumSave(i) = momentum(i).zeros(momentum(i).dims);
	      momentumSave(i) <-- momentum(i);
	  }
    }
    if (aelem.asInstanceOf[AnyRef] == null) aelem = modelmats(i).zeros(1,1);
    swapMats(i);
  }

  override def update(ipass:Int, step:Long, gprogress:Float):Unit = {
    val start = toc;
    clipping();
    val nsteps = if (step == 0) 1.0 else {
	if (firstStep == 0.0) {
	  firstStep = step;
	  1f;
	} else {
	  step / firstStep;
	}
      }
    val batchSize = model.gmats(0).ncols;
    val nmats = updatemats.length;
    val lr0 = if (opts.lr_policy.asInstanceOf[AnyRef] != null) opts.lr_policy(ipass, nsteps.toFloat, gprogress) else 0;
    for (i <- 0 until nmats) {
      if (updatemats(i).asInstanceOf[AnyRef] != null) {
	val mm = modelmats(i);
	tscale = if (te.asInstanceOf[AnyRef] != null) {
	  te <-- opts.texp;
	  stepn.set(1f/nsteps.toFloat);
	  stepn ^ te;
	} else {
	  pe <-- opts.pexp;
	  stepn.set(1f/(ipass+1));
	  stepn ^ pe;
	}
	if (opts.lr_policy.asInstanceOf[AnyRef] != null) {
	  lrate.set(lr0);
	} else {
	  if (opts.lrate.ncols > 1) {
	    lrate <-- opts.lrate(?,i);
	  } else {
	    lrate <-- opts.lrate;
	  }
	}
	lrate ~ lrate / batchSize;
	val lr_scales = model.lr_scales;
	if (lr_scales.asInstanceOf[AnyRef] != null) {
	  lrate ~ lrate *@ lr_scales(i);
	} 
	if (opts.waitsteps < nsteps) {
	  val grad = updatemats(i);
	  if (opts.l2reg.asInstanceOf[AnyRef] != null) {
	    val i0 = if (opts.l2reg.length > 1) i else 0;
	    (grad, mm) match {
	      case (ggrad:GMat, gmm:GMat) => Grad.linComb(ggrad, 1f, gmm, -opts.l2reg(i0) * batchSize, ggrad);
	      case _ => grad ~ grad - (mm *@ (opts.l2reg(i0) * batchSize));
	    }
	  }
	  if (opts.langevin > 0) {                              // Add Langevin random permutations
	    normrnd(0, opts.langevin, randmat(i));
	    grad ~ grad + randmat(i);
	  }
	  grad ~ grad *@ (lrate *@ tscale);
	  if (opts.vel_decay.asInstanceOf[AnyRef] != null) {
	    val i0 = if (opts.vel_decay.length > 1) i else 0;
	    (grad, momentum(i)) match {
	      case (ggrad:GMat, gmom:GMat) => Grad.linComb(ggrad, 1f, gmom, opts.vel_decay(i0), gmom);
	      case _ => {
		mu <-- opts.vel_decay(i0);                          // Get the momentum decay rate      
		momentum(i) ~ momentum(i) *@ mu;                    // update momentum using the new gradient p = mu p + grad
		momentum(i) ~ momentum(i) + grad;
	      }
	    }
	    grad <-- momentum(i);
	  }
	  if (opts.nesterov_vel_decay.asInstanceOf[AnyRef] != null) {
	    val i0 = if (opts.nesterov_vel_decay.length > 1) i else 0;
	    mu <-- opts.nesterov_vel_decay(i0);                    // Implement x_t = x_t-1 + p_t + mu * (p_t - p_t-1)
	    (grad, momentum(i)) match {
	      case (ggrad:GMat, gmom:GMat) => {
		Grad.linComb(ggrad, 1f, gmom, opts.vel_decay(i0), gmom);   // p_t = mu * p_t-1 + g
		Grad.linComb(ggrad, 1f, gmom, opts.vel_decay(i0), ggrad);  // g' = mu p_t + (p_t - mu*p_t-1) = mu *p_t + g
	      }
	      case _ => {      
		momentum(i) ~ momentum(i) *@ mu;                     // Compute mu * p_t-1
		momentum(i) ~ momentum(i) + grad;                    // Compute new momentum p_t = mu * p_t-1 + g
		mm ~ mm + grad;                                      // Add 2nd and 4th terms: (p_t - mu p_t-1) = g to the model;
		grad ~ momentum(i) *@ mu;                            // grad = mu p_t is ready to be added.
	      }
	    }       	                                  
	  }
	  modelmats(i) ~ modelmats(i) + grad;
	  if (mask != null) modelmats(i) ~ modelmats(i) *@ mask;
	}
	updatemats(i).clear;
	if (opts.collideEvery > 0 && (nsteps.toInt % opts.collideEvery) == 0) collide(momentum(i), momentumSave(i), i);
	if (opts.attractEvery > 0 && (nsteps.toInt % opts.attractEvery) == 0) attract(modelmats(i), modelmatsSave(i), opts.attraction, i);
      }
      if (opts.doSwaps) checkSwapMats(i);
    }
    runningtime += toc - start;
  }
}


object GradCollide {
  trait Opts extends Grad.Opts {
    var doSwaps = true;
    var collideEvery = 2;
    var attractEvery = 2;
    var attraction = 0.1f;
    var hardness = 1f;
    var perParticleMomentum = true;
    var logCollide = false;
    var logAttract = false;
    var logSwap = false;
  }
  
  class Options extends Opts {}
  
}

