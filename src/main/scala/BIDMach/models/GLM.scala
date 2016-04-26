package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GDMat,GMat,GIMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.CUMACH
import scala.concurrent.future
import scala.concurrent.ExecutionContext.Implicits.global
import java.util.concurrent.CountDownLatch
import BIDMach.datasources._
import BIDMach.datasinks._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach._

/**
 * Train a GLM model. The types of model are given by the values of opts.links (IMat). They are:
 - 0 = linear model (squared loss)
 - 1 = logistic model (logistic loss)
 - 2 = hinge logistic (hinge loss on logistic prediction)
 - 3 = SVM model (hinge loss)
 * 
 * Options are:
 -   links: an IMat whose nrows should equal the number of targets. Values as above. Can be different for different targets.
 -   iweight: an FMat typically used to select a weight row from the input. i.e. iweight = 0,1,0,0,0 uses the second
 *            row of input data as weights to be applied to input samples. The iweight field should be 0 in mask. 
 *            
 * Inherited from Regression Model:
 -   rmask: FMat, optional, 0-1-valued. Used to ignore certain input rows (which are targets or weights). 
 *          Zero value in an element will ignore the corresponding row. 
 -   targets: FMat, optional, 0-1-valued. ntargs x nfeats. Used to specify which input features corresponding to targets. 
 -   targmap: FMat, optional, 0-1-valued. nntargs x ntargs. Used to replicate actual targets, e.g. to train multiple models
 *            (usually with different parameters) for the same target. 
 *    
 * Some convenience functions for training:
 * {{{
 * val (mm, opts) = GLM.learner(a, d)    // On an input matrix a including targets (set opts.targets to specify them), 
 *                                       // learns a GLM model of type d. 
 *                                       // returns the model (nn) and the options class (opts). 
 * val (mm, opts) = GLM.learner(a, c, d) // On an input matrix a and target matrix c, learns a GLM model of type d. 
 *                                       // returns the model (nn) and the options class (opts). 
 * val (nn, nopts) = predictor(model, ta, pc, d) // constructs a prediction learner from an existing model. returns the learner and options. 
 *                                       // pc should be the same dims as the test label matrix, and will contain results after nn.predict
 * val (mm, mopts, nn, nopts) = GLM.learner(a, c, ta, pc, d) // a = training data, c = training labels, ta = test data, pc = prediction matrix, d = type.
 *                                       // returns a training learner mm, with options mopts. Also returns a prediction model nn with its own options.
 *                                       // typically set options, then do mm.train; nn.predict with results in pc.  
 * val (mm, opts) = learner(ds)          // Build a learner for a general datasource ds (e.g. a files data source). 
 * }}}                                   
 */

class GLM(opts:GLM.Opts) extends RegressionModel(opts) {
  
  val linkArray = GLM.linkArray
  
  var mylinks:Mat = null;
  var iweight:Mat = null;
  var ulim:Mat = null;
  var llim:Mat = null;
  var totflops = 0L;
  var hashFeatures = 0;
  // For integrated ADAGrad updater
  var vexp:Mat = null;
  var texp:Mat = null;
  var lrate:Mat = null;
  var sumsq:Mat = null;
  var firststep = -1f;
  var waitsteps = 0;
  var epsilon = 0f;
  
  override def copyTo(mod:Model) = {
    super.copyTo(mod);
    val rmod = mod.asInstanceOf[GLM];
    rmod.mylinks = mylinks;
    rmod.iweight = iweight;    
  }
  
  override def init() = {
  	useGPU = opts.useGPU && Mat.hasCUDA > 0
    val data0 = mats(0)
    val m = if (opts.hashFeatures > 0) opts.hashFeatures else size(data0, 1)
    val targetData = mats.length > 1
    val d = if (opts.targmap.asInstanceOf[AnyRef] != null) {
      opts.targmap.nrows 
    } else if (opts.targets.asInstanceOf[AnyRef] != null) {
      opts.targets.nrows 
    } else if (mats.length > 1) {
      mats(1).nrows  
    } else {
      modelmats(0).nrows;
    }
    val sdat = (sum(data0,2).t + 0.5f).asInstanceOf[FMat]
    sp = sdat / sum(sdat)
    println("corpus perplexity=%f" format (math.exp(-(sp ddot ln(sp)))))
    
    if (refresh) {
    	val mm = zeros(d,m);
      setmodelmats(Array(mm))
    }
    modelmats(0) = convertMat(modelmats(0));
    updatemats = Array(modelmats(0).zeros(modelmats(0).nrows, modelmats(0).ncols));
    targmap = if (opts.targmap.asInstanceOf[AnyRef] != null) convertMat(opts.targmap) else opts.targmap
    if (! targetData) {
      targets = if (opts.targets.asInstanceOf[AnyRef] != null) convertMat(opts.targets) else opts.targets
      mask =    if (opts.rmask.asInstanceOf[AnyRef] != null) convertMat(opts.rmask) else opts.rmask
    } 
    mylinks = if (useGPU) GIMat(opts.links) else opts.links;
    iweight = opts.iweight;
    if (iweight.asInstanceOf[AnyRef] != null && useGPU) iweight = convertMat(iweight);
    if (mask.asInstanceOf[AnyRef] != null) modelmats(0) ~ modelmats(0) ∘ mask;
    totflops = 0L;
    for (i <- 0 until opts.links.length) {
      totflops += linkArray(opts.links(i)).fnflops;
    }
    ulim = convertMat(opts.lim)
    llim = - ulim;
    hashFeatures = opts.hashFeatures;
    if (opts.aopts != null) {
      initADAGrad(d, m);
    } else {
    	vexp = null;
    	texp = null;
    	lrate = null;
    	sumsq = null;
    }
  }
  
  def initADAGrad(d:Int, m:Int) = {
    val aopts = opts.asInstanceOf[ADAGrad.Opts];
    firststep = -1f;
    lrate = convertMat(aopts.lrate);
    texp = convertMat(aopts.texp);
    vexp = convertMat(aopts.vexp);
    sumsq = convertMat(zeros(d, m));
    sumsq.set(aopts.initsumsq);
    waitsteps = aopts.waitsteps;
    epsilon = aopts.epsilon;
  }
    
  def mupdate(in:Mat, ipass:Int, pos:Long) = {
    val targs = targets * in
    min(targs, 1f, targs)
    val dweights = if (iweight.asInstanceOf[AnyRef] != null) iweight * in else null
    mupdate3(in, targs, dweights, ipass, pos)
  }
  
  def mupdate2(in:Mat, targ:Mat, ipass:Int, pos:Long) = mupdate3(in, targ, null, ipass, pos)
  
  def mupdate3(in:Mat, targ:Mat, dweights:Mat, ipass:Int, pos:Long) = {        
    val ftarg = full(targ);
    val targs = if (targmap.asInstanceOf[AnyRef] != null) targmap * ftarg else ftarg;
    val eta = if (hashFeatures > 0) GLM.hashMult(modelmats(0), in, opts.hashBound1, opts.hashBound2) else modelmats(0) * in 
    if (opts.lim > 0) {
      max(eta, llim, eta);
      min(eta, ulim, eta);
    }
    GLM.preds(eta, eta, mylinks, totflops);
    GLM.derivs(eta, targs, eta, mylinks, totflops);
    if (dweights.asInstanceOf[AnyRef] != null) eta ~ eta ∘ dweights;
    if (opts.aopts != null) {
      if (firststep <= 0) firststep = pos.toFloat;
      val step = (pos + firststep)/firststep;
      if (hashFeatures == 0) {
      	ADAGrad.multUpdate(eta, in, modelmats(0), sumsq, mask, lrate, texp, vexp, epsilon, step, waitsteps);
      } else {
        ADAGrad.hashmultUpdate(eta, in, hashFeatures, opts.hashBound1, opts.hashBound2, 1,
            modelmats(0), sumsq, mask, lrate, texp, vexp, epsilon, step, waitsteps);
      }
    } else {
    	if (hashFeatures > 0) {
    		updatemats(0) <-- GLM.hashMultT(eta, in, modelmats(0).ncols, opts.hashBound1, opts.hashBound2);
    	} else {
    		updatemats(0) ~ eta *^ in;
    	}
    	if (mask.asInstanceOf[AnyRef] != null) {
    		updatemats(0) ~ updatemats(0) ∘ mask
    	}
    }
  }
  
  def meval(in:Mat):FMat = {
    val targs = if (targets.asInstanceOf[AnyRef] != null) {val targs0 = targets * in; min(targs0, 1f, targs0); targs0} else null
    val dweights = if (iweight.asInstanceOf[AnyRef] != null) iweight * in else null;
    meval3(in, targs, dweights);
  }
  
  def meval2(in:Mat, targ:Mat):FMat = meval3(in, targ, null)
  
  def meval3(in:Mat, targ:Mat, dweights:Mat):FMat = {
    val ftarg = if (targ.asInstanceOf[AnyRef] != null) full(targ) else null;
    val targs = if (targmap.asInstanceOf[AnyRef] != null && ftarg.asInstanceOf[AnyRef] != null) targmap * ftarg else ftarg;
    val eta = if (hashFeatures > 0) GLM.hashMult(modelmats(0), in, opts.hashBound1, opts.hashBound2) else modelmats(0) * in;
    GLM.preds(eta, eta, mylinks, totflops);
    if (ogmats != null) {ogmats(0) = eta;}
    if (targs.asInstanceOf[AnyRef] != null) {
    	val v = GLM.llfun(eta, targs, mylinks, totflops);
    	if (dweights.asInstanceOf[AnyRef] != null) {
    		FMat(sum(v ∘  dweights, 2) / sum(dweights))
    	} else {
    		FMat(mean(v, 2))
    	}
    } else {
      row(0)
    }
  }
  
}


object GLM {
  trait Opts extends RegressionModel.Opts {
    var links:IMat = null;
    var iweight:FMat = null;
    var lim = 0f;
    var hashFeatures = 0;
    var hashBound1:Int = 1000000;
    var hashBound2:Int = 1000000;
    var aopts:ADAGrad.Opts = null;
  }
  
  val linear = 0;
  val logistic = 1;
  val maxp = 2;
  val svm = 3;

  object LinearLink extends GLMlink {
  	def link(in:Float) = {
  		in
  	}

  	def mean(in:Float) = {
  		in
  	}

  	def derivlink(in:Float, targ:Float) = {
  		targ - in;
  	}

  	def likelihood(pred:Float, targ:Float) = {
  		val diff = targ - pred;
  		- diff * diff;
  	}

  	override val linkfn = link _;

  	override val derivfn = derivlink _;

  	override val meanfn = mean _;

  	override val likelihoodfn = likelihood _;

  	val fnflops = 2;
  }

  object LogisticLink extends GLMlink {
  	def link(in:Float) = {
  		math.log(in / (1.0f - in)).toFloat;
  	}

  	def mean(in:Float) = {
  		if (in > 0) {
  			val tmp = math.exp(-in);
  			(1.0 / (1.0 + tmp)).toFloat;
  		} else {
  			val tmp = math.exp(in);
  			(tmp / (1.0 + tmp)).toFloat;
  		}
  	}

  	def derivlink(in:Float, targ:Float) = {
  		targ - in;
  	}

  	def likelihood(pred:Float, targ:Float) = {
  		math.log(targ * pred + (1.0f - targ) * (1.0f - pred) + 1e-20).toFloat
  	}

  	override val linkfn = link _;

  	override val derivfn = derivlink _;

  	override val meanfn = mean _;

  	override val likelihoodfn = likelihood _;

  	val fnflops = 20;
  }


  object MaxpLink extends GLMlink {
  	def link(in:Float) = {
  		math.log(in / (1.0f - in)).toFloat;
  	}

  	def mean(in:Float) = {
  		if (in > 0) {
  			val tmp = math.exp(-in);
  			(1.0 / (1.0 + tmp)).toFloat;
  		} else {
  			val tmp = math.exp(in);
  			(tmp / (1.0 + tmp)).toFloat;
  		}
  	}

  	def derivlink(p:Float, targ:Float) = {
  		(2.0f * targ - 1.0f) * p * (1.0f - p);
  	}

  	def likelihood(pred:Float, targ:Float) = {
  		targ * pred + (1.0f - targ) * (1.0f - pred) -1.0f;
  	}

  	override val linkfn = link _;

  	override val derivfn = derivlink _;

  	override val meanfn = mean _;

  	override val likelihoodfn = likelihood _;

  	val fnflops = 20;
  }

  object SVMLink extends GLMlink {
  	def link(in:Float) = {
  		in
  	}

  	def mean(in:Float) = {
  		in
  	}

  	def derivlink(pred:Float, targ:Float) = {
  		val ttarg = 2 * targ - 1;
  		if (pred * ttarg < 1f) ttarg else 0f;
  	}

  	def likelihood(pred:Float, targ:Float) = {
  		val ttarg = 2 * targ - 1;
  		scala.math.min(0f, ttarg * pred - 1f);
  	}

  	override val linkfn = link _;

  	override val derivfn = derivlink _;

  	override val meanfn = mean _;

  	override val likelihoodfn = likelihood _;

  	val fnflops = 2;
  }

  object LinkEnum extends Enumeration {
  	type LinkEnum = Value;
  	val Linear, Logistic, Maxp, SVMLink = Value
  }

  abstract class GLMlink {
  	val linkfn:(Float => Float)
  	val derivfn:((Float,Float) => Float)
  	val meanfn:(Float => Float)
  	val likelihoodfn:((Float,Float) => Float)
  	val fnflops:Int
  }
  
  val linkArray = Array[GLMlink](LinearLink, LogisticLink, MaxpLink, SVMLink)
  
  class Options extends Opts {}
  
  def meanHelper(feta:FMat, fout:FMat, ilinks:IMat, istart:Int, iend:Int) {
    var i = istart
    while (i < iend) {
      var j = 0
      while (j < feta.nrows) { 
        val fun = GLM.linkArray(ilinks(j)).meanfn
        fout.data(j + i * fout.nrows) = fun(feta.data(j + i * feta.nrows))
        j += 1 
      }
      i += 1
    }     
  }
  
  def preds(eta:Mat, out:Mat, links:Mat, totflops:Long):Mat = {
    (eta, links, out) match {
      case (feta:FMat, ilinks:IMat, fout:FMat) => {
        Mat.nflops += totflops * feta.ncols
        meanHelper(feta, fout, ilinks, 0, feta.ncols)
        out
      }
      case (geta:GMat, gilinks:GIMat, gout:GMat) => {
        Mat.nflops += totflops * geta.ncols
        CUMACH.applypreds(geta.data, gilinks.data, gout.data, geta.nrows, geta.ncols)
        out
      }
      case (geta:GDMat, gilinks:GIMat, gout:GDMat) => {
        Mat.nflops += totflops * geta.ncols
        CUMACH.applydpreds(geta.data, gilinks.data, gout.data, geta.nrows, geta.ncols)
        out
      }
    }
  }
  
  def preds(eta:Mat, links:Mat, totflops:Long):Mat = {
    (eta, links) match {
      case (feta:FMat, ilinks:IMat) => {
        val fout = FMat.newOrCheckFMat(eta.nrows, eta.ncols, null, eta.GUID, links.GUID, "GLM.preds".##)
        Mat.nflops += totflops * feta.ncols
        meanHelper(feta, fout, ilinks, 0, feta.ncols)
        fout
      }
      case (geta:GMat, gilinks:GIMat) => {
        val gout = GMat.newOrCheckGMat(eta.nrows, eta.ncols, null, eta.GUID, links.GUID, "GLM.preds".##)
        Mat.nflops += totflops * geta.ncols
        CUMACH.applypreds(geta.data, gilinks.data, gout.data, geta.nrows, geta.ncols)
        gout
      }
      case (geta:GDMat, gilinks:GIMat) => {
      	val gout = GDMat.newOrCheckGDMat(eta.nrows, eta.ncols, null, eta.GUID, links.GUID, "GLM.preds".##)
        Mat.nflops += totflops * geta.ncols
        CUMACH.applydpreds(geta.data, gilinks.data, gout.data, geta.nrows, geta.ncols)
        gout
      }
    }
  }
  
  def llfun(pred:Mat, targ:Mat, links:Mat, totflops:Long):Mat = {
    (pred, targ, links) match {
      case (fpred:FMat, ftarg:FMat, ilinks:IMat) => {
        Mat.nflops += 10L * ftarg.length
            var i = 0
            val out = (ftarg + 5f)
            while (i < ftarg.ncols) {
                var j = 0
                while (j < ftarg.nrows) {
                    val fun = GLM.linkArray(ilinks(j)).likelihoodfn
                    out.data(j + i * out.nrows) = fun(fpred.data(j + i * ftarg.nrows),  ftarg.data(j + i * ftarg.nrows))
                    j += 1
                }
                i += 1
            }
            out
      }
      case (gpred:GMat, gtarg:GMat, gilinks:GIMat) => {
        Mat.nflops += totflops * gpred.ncols
        val out = (gpred + 3f)
        CUMACH.applylls(gpred.data, gtarg.data, gilinks.data, out.data, gpred.nrows, gpred.ncols)
        out
      }
      case (gpred:GDMat, gtarg:GDMat, gilinks:GIMat) => {
        Mat.nflops += totflops * gpred.ncols
        val out = (gpred + 3f)
        CUMACH.applydlls(gpred.data, gtarg.data, gilinks.data, out.data, gpred.nrows, gpred.ncols)
        out
      }
    }
  }
  
  def derivs(pred:Mat, targ:Mat, out:Mat, links:Mat, totflops:Long) = {
    (pred, targ, out, links) match {
      case (fpred:FMat, ftarg:FMat, fout:FMat, ilinks:IMat) => {
      	Mat.nflops += 10L * ftarg.length;
      	var i = 0;
      	while (i < ftarg.ncols) {
      		var j = 0;
      		while (j < ftarg.nrows) {
      			val fun = GLM.linkArray(ilinks(j)).derivfn;
      			fout.data(j + i * out.nrows) = fun(fpred.data(j + i * ftarg.nrows),  ftarg.data(j + i * ftarg.nrows));
      			j += 1;
      		}
      		i += 1;
      	}
      	fout;
      }
      case (gpred:GMat, gtarg:GMat, gout:GMat, gilinks:GIMat) => {
        Mat.nflops += totflops * gpred.ncols
        CUMACH.applyderivs(gpred.data, gtarg.data, gilinks.data, gout.data, gpred.nrows, gpred.ncols)
        gout
      }
      case (gpred:GDMat, gtarg:GDMat, gout:GDMat, gilinks:GIMat) => {
        Mat.nflops += totflops * gpred.ncols
        CUMACH.applydderivs(gpred.data, gtarg.data, gilinks.data, gout.data, gpred.nrows, gpred.ncols)
        gout
      }
    }
  }
   
  def derivs(pred:Mat, targ:Mat, links:Mat, totflops:Long) = {
    (pred, targ, links) match {
      case (fpred:FMat, ftarg:FMat, ilinks:IMat) => {
      	val fout = FMat.newOrCheckFMat(pred.nrows, pred.ncols, null, pred.GUID, targ.GUID, links.GUID, "GLM.derivs".##)
        Mat.nflops += 10L * ftarg.length;
      	var i = 0;
      	while (i < ftarg.ncols) {
      		var j = 0
      				while (j < ftarg.nrows) {
      					val fun = GLM.linkArray(ilinks(j)).derivfn;
      					fout.data(j + i * fout.nrows) = fun(fpred.data(j + i * ftarg.nrows),  ftarg.data(j + i * ftarg.nrows));
      					j += 1;
      				}
      		i += 1;
      	}
      	fout;
      }
      case (gpred:GMat, gtarg:GMat, gilinks:GIMat) => {
      	val gout = GMat.newOrCheckGMat(pred.nrows, pred.ncols, null, pred.GUID, targ.GUID, links.GUID, "GLM.derivs".##)
        Mat.nflops += totflops * gpred.ncols
        CUMACH.applyderivs(gpred.data, gtarg.data, gilinks.data, gout.data, gpred.nrows, gpred.ncols)
        gout
      }
      case (gpred:GDMat, gtarg:GDMat, gilinks:GIMat) => {
        val gout = GDMat.newOrCheckGDMat(pred.nrows, pred.ncols, null, pred.GUID, targ.GUID, links.GUID, "GLM.derivs".##)
        Mat.nflops += totflops * gpred.ncols
        CUMACH.applydderivs(gpred.data, gtarg.data, gilinks.data, gout.data, gpred.nrows, gpred.ncols)
        gout
      }
    }
  }
   
  def hashMult(a:GMat, b:GSMat, bound1:Int, bound2:Int):GMat = {
    val c = GMat.newOrCheckGMat(a.nrows, b.ncols, null, a.GUID, b.GUID, "hashMult".##);
    c.clear;
    val npercol = b.nnz / b.ncols;
    Mat.nflops += 1L * a.nrows * npercol * b.nnz;
    CUMACH.hashMult(a.nrows, a.ncols, b.ncols, bound1, bound2, a.data, b.data, b.ir, b.jc, c.data, 0);
    c
  }
  
  def hashMult(a:Mat, b:Mat, bound1:Int, bound2:Int):Mat = {
  	(a, b) match {
  	  case (ga:GMat, gb:GSMat) => hashMult(ga, gb, bound1, bound2)
  	}
  }

  
  def hashMultT(a:GMat, b:GSMat, nfeats:Int, bound1:Int, bound2:Int):GMat = {
    val c = GMat.newOrCheckGMat(a.nrows, nfeats, null, a.GUID, b.GUID, nfeats, "hashMultT".##);
    c.clear;
    val npercol = b.nnz / b.ncols;
    Mat.nflops += 1L * a.nrows * npercol * b.nnz;
    CUMACH.hashMult(a.nrows, nfeats, b.ncols, bound1, bound2, a.data, b.data, b.ir, b.jc, c.data, 1);
    c
  }
  
  def hashMultT(a:Mat, b:Mat, nfeats:Int, bound1:Int, bound2:Int):Mat = {
  	(a, b) match {
  	  case (ga:GMat, gb:GSMat) => hashMultT(ga, gb, nfeats, bound1, bound2)
  	}
  }

  def hashCross(a:GMat, b:GSMat, c:GSMat):GMat = {
    val d = GMat.newOrCheckGMat(a.nrows, b.ncols, null, a.GUID, b.GUID, "hashCross".##);
    val npercol = b.nnz / b.ncols;
    Mat.nflops += 1L * a.nrows * npercol * b.nnz;
    d.clear;
    CUMACH.hashCross(a.nrows, a.ncols, b.ncols, a.data, b.data, b.ir, b.jc, c.data, c.ir, c.jc, d.data, 0);
    d
  }
  
  def hashCross(a:Mat, b:Mat, c:Mat):Mat = {
  	(a, b, c) match {
  	  case (ga:GMat, gb:GSMat, gc:GSMat) => hashCross(ga, gb, gc)
  	}
  }
  
  def hashCrossT(a:GMat, b:GSMat, c:GSMat, nfeats:Int):GMat = {
    val d = GMat.newOrCheckGMat(a.nrows, nfeats, null, a.GUID, b.GUID, "hashCrossT".##);
    val npercol = b.nnz / b.ncols;
    Mat.nflops += 1L * a.nrows * npercol * b.nnz;
    d.clear;
    CUMACH.hashCross(a.nrows, nfeats, b.ncols, a.data, b.data, b.ir, b.jc, c.data, c.ir, c.jc, d.data, 1);
    d
  }
  
  def hashCrossT(a:Mat, b:Mat, c:Mat, nfeats:Int):Mat = {
  	(a, b, c) match {
  	  case (ga:GMat, gb:GSMat, gc:GSMat) => hashCrossT(ga, gb, gc, nfeats)
  	}
  }
  
  def mkGLMModel(fopts:Model.Opts) = {
  	new GLM(fopts.asInstanceOf[GLM.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
  	new ADAGrad(nopts.asInstanceOf[ADAGrad.Opts])
  } 
  
  def mkRegularizer(nopts:Mixin.Opts):Array[Mixin] = {
    Array(new L1Regularizer(nopts.asInstanceOf[L1Regularizer.Opts]))
  }
  
  def mkL2Regularizer(nopts:Mixin.Opts):Array[Mixin] = {
    Array(new L2Regularizer(nopts.asInstanceOf[L2Regularizer.Opts]))
  }

  def mkL1L2Regularizers(nopts:Mixin.Opts):Array[Mixin] = {
    Array(new L1Regularizer(nopts.asInstanceOf[L1Regularizer.Opts]),
	  new L2Regularizer(nopts.asInstanceOf[L2Regularizer.Opts]))
  }
  
  class LearnOptions extends Learner.Options with GLM.Opts with MatSource.Opts with ADAGrad.Opts with L1Regularizer.Opts
  class Learn12Options extends Learner.Options with GLM.Opts with MatSource.Opts with ADAGrad.Opts with L1Regularizer.Opts with L2Regularizer.Opts
     
  // Basic in-memory learner with generated target
  def learner(mat0:Mat, d:Int = 0) = { 
    val opts = new LearnOptions
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    opts.lrate = 1f
  	val nn = new Learner(
  	    new MatSource(Array(mat0:Mat), opts), 
  	    new GLM(opts), 
  	    mkRegularizer(opts),
  	    new ADAGrad(opts), 
  	    null,
  	    opts)
    (nn, opts)
  } 
    
  def learner(mat0:Mat):(Learner, LearnOptions) = learner(mat0, 0)
  
    // Basic in-memory learner with generated target
  def learnerX(mat0:Mat, d:Int = 0) = { 
    val opts = new LearnOptions
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    opts.lrate = 1f
    opts.aopts = opts
  	val nn = new Learner(
  	    new MatSource(Array(mat0:Mat), opts), 
  	    new GLM(opts), 
  	    mkRegularizer(opts),
  	    null, 
  	    null,
  	    opts)
    (nn, opts)
  } 
    
  def learnerX(mat0:Mat):(Learner, LearnOptions) = learnerX(mat0, 0)
  
  // Basic in-memory learner with explicit target
  def learner(mat0:Mat, targ:Mat, d:Int):(Learner, LearnOptions) = {
    val mopts = new LearnOptions;
    mopts.lrate = 1f
    mopts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    if (mopts.links == null) mopts.links = izeros(1,targ.nrows)
    mopts.links.set(d)
    val model = new GLM(mopts)
    val mm = new Learner(
        new MatSource(Array(mat0, targ), mopts), 
        model, 
        mkRegularizer(mopts),
        new ADAGrad(mopts), 
        null,
        mopts)
    (mm, mopts)
  }

  
  // Basic in-memory learner with explicit target
  def learnerX(mat0:Mat, targ:Mat, d:Int):(Learner, LearnOptions) = {
    val mopts = new LearnOptions;
    mopts.lrate = 1f
    mopts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    if (mopts.links == null) mopts.links = izeros(1,targ.nrows)
    mopts.links.set(d)
    val model = new GLM(mopts)
    mopts.aopts = mopts;
    val mm = new Learner(
        new MatSource(Array(mat0, targ), mopts), 
        model, 
        mkRegularizer(mopts),
        null, 
        null,
        mopts)
    (mm, mopts)
  }
  
  def LinLearner(mat0:Mat, targ:Mat):(Learner, LearnOptions) = learner(mat0, targ, 0)
  
  def LogLearner(mat0:Mat, targ:Mat):(Learner, LearnOptions) = learner(mat0, targ, 2)
  
  // This function constructs a learner and a predictor. 
  def learner(mat0:Mat, targ:Mat, mat1:Mat, preds:Mat, d:Int):(Learner, LearnOptions, Learner, LearnOptions) = {
    val mopts = new LearnOptions;
    val nopts = new LearnOptions;
    mopts.lrate = 1f
    mopts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    mopts.autoReset = false
    if (mopts.links == null) mopts.links = izeros(targ.nrows,1)
    nopts.links = mopts.links
    mopts.links.set(d)
    nopts.batchSize = mopts.batchSize
    nopts.putBack = 1
    val model = new GLM(mopts)
    val mm = new Learner(
        new MatSource(Array(mat0, targ), mopts), 
        model, 
        mkRegularizer(mopts),
        new ADAGrad(mopts), 
        null,
        mopts)
    val nn = new Learner(
        new MatSource(Array(mat1, preds), nopts), 
        model, 
        null,
        null, 
        null,
        nopts)
    (mm, mopts, nn, nopts)
  }
  
  class GOptions extends Learner.Options with GLM.Opts with ADAGrad.Opts with L1Regularizer.Opts 

  // A learner that uses a general data source (e.g. a files data source). 
  // The datasource options (like batchSize) need to be set externally. 
  def learner(ds:DataSource):(Learner, GOptions) = {
    val mopts = new GOptions;
    mopts.lrate = 1f
    val model = new GLM(mopts)
    val mm = new Learner(
        ds, 
        model, 
        mkRegularizer(mopts),
        new ADAGrad(mopts),
        null,
        mopts)
    (mm, mopts)
  }
  
  def learnerX(ds:DataSource):(Learner, GOptions) = {
    val mopts = new GOptions;
    mopts.lrate = 1f
    mopts.aopts = mopts;
    val model = new GLM(mopts)
    val mm = new Learner(
        ds, 
        model, 
        mkRegularizer(mopts),
        null,
        null,
        mopts);
    (mm, mopts)
  }
  
  class FGOptions extends Learner.Options with GLM.Opts with ADAGrad.Opts with L1Regularizer.Opts with FileSource.Opts
  
  // A learner that uses a files data source specified by a list of strings.  
  def learner(fnames:List[String]):(Learner, FGOptions) = {
    val mopts = new FGOptions;
    mopts.lrate = 1f;
    val model = new GLM(mopts);
    mopts.fnames = fnames.map((a:String) => FileSource.simpleEnum(a,1,0));
    val ds = new FileSource(mopts);    
    val mm = new Learner(
        ds, 
        model, 
        mkRegularizer(mopts),
        new ADAGrad(mopts), 
        null,
        mopts)
    (mm, mopts)
  }
  
    // A learner that uses a files data source specified by a list of strings.  
  def learnerX(fnames:List[String]):(Learner, FGOptions) = {
    val mopts = new FGOptions;
    mopts.lrate = 1f;
    mopts.aopts = mopts;
    val model = new GLM(mopts);
    mopts.fnames = fnames.map((a:String) => FileSource.simpleEnum(a,1,0));
    val ds = new FileSource(mopts);    
    val mm = new Learner(
        ds, 
        model, 
        mkRegularizer(mopts),
        null,
        null,
        mopts)
    (mm, mopts)
  }
  
  class PredOptions extends Learner.Options with GLM.Opts with MatSource.Opts with MatSink.Opts
  
  // This function constructs a predictor from an existing model 
  def predictor(model0:Model, mat1:Mat):(Learner, PredOptions) = {
    val model = model0.asInstanceOf[GLM]
    val nopts = new PredOptions;
    nopts.batchSize = math.min(10000, mat1.ncols/30 + 1)
    nopts.putBack = 0
    val newmod = new GLM(nopts);
    newmod.refresh = false
    newmod.copyFrom(model);
    val mopts = model.opts.asInstanceOf[GLM.Opts];
    nopts.targmap = mopts.targmap;
    nopts.links = mopts.links;
    nopts.targets = mopts.targets;
    nopts.iweight = mopts.iweight;
    nopts.lim = mopts.lim;
    nopts.hashFeatures = mopts.hashFeatures;
    nopts.hashBound1 = mopts.hashBound1;
    nopts.hashBound2 = mopts.hashBound2;   
    val nn = new Learner(
        new MatSource(Array(mat1), nopts), 
        newmod, 
        null,
        null,
        new MatSink(nopts),
        nopts)
    (nn, nopts)
  }
  
   // Basic in-memory SVM learner with explicit target
  def SVMlearner(mat0:Mat, targ:Mat):(Learner, Learn12Options) = {
    val mopts = new Learn12Options;
    mopts.lrate = 1f
    mopts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    if (mopts.links == null) mopts.links = izeros(targ.nrows,1)
    mopts.links.set(3)
    mopts.reg2weight = 1f
    val model = new GLM(mopts)
    val mm = new Learner(
        new MatSource(Array(mat0, targ), mopts), 
        model, 
        mkL1L2Regularizers(mopts),
        new ADAGrad(mopts), 
        null,
        mopts)
    (mm, mopts)
  }
  
  // This function constructs a learner and a predictor. 
  def SVMlearner(mat0:Mat, targ:Mat, mat1:Mat, preds:Mat):(Learner, Learn12Options, Learner, Learn12Options) = {
    val mopts = new Learn12Options;
    val nopts = new Learn12Options;
    mopts.lrate = 1f
    mopts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    if (mopts.links == null) mopts.links = izeros(targ.nrows,1)
    mopts.links.set(3)
    mopts.reg2weight = 1f
    nopts.links = mopts.links
    nopts.batchSize = mopts.batchSize
    nopts.putBack = 1
    val model = new GLM(mopts)
    val mm = new Learner(
        new MatSource(Array(mat0, targ), mopts), 
        model, 
        mkL1L2Regularizers(mopts),
        new ADAGrad(mopts), 
        null,
        mopts)
    val nn = new Learner(
        new MatSource(Array(mat1, preds), nopts), 
        model, 
        null,
        null,
        null,
        nopts)
    (mm, mopts, nn, nopts)
  }
  
   // This function constructs a predictor from an existing model 
  def SVMpredictor(model:Model, mat1:Mat, preds:Mat):(Learner, LearnOptions) = {
    val nopts = new LearnOptions;
    nopts.batchSize = math.min(10000, mat1.ncols/30 + 1)
    if (nopts.links == null) nopts.links = izeros(preds.nrows,1)
    nopts.links.set(3)
    nopts.putBack = 1
    val nn = new Learner(
        new MatSource(Array(mat1, preds), nopts), 
        model.asInstanceOf[GLM], 
        null,
        null,
        null,
        nopts)
    (nn, nopts)
  }
     
  def learnBatch(mat0:Mat, targ:Mat, d:Int) = {
    val opts = new LearnOptions
    opts.lrate = 1f
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    if (opts.links == null) opts.links = izeros(targ.nrows,1)
    val nn = new Learner(
        new MatSource(Array(mat0, targ), opts), 
        new GLM(opts), 
        mkRegularizer(opts), 
        new ADAGrad(opts),
        null,
        opts)
    (nn, opts)
  }
  
  class LearnParOptions extends ParLearner.Options with GLM.Opts with MatSource.Opts with ADAGrad.Opts with L1Regularizer.Opts
  
  def learnPar(mat0:Mat, d:Int) = {
    val opts = new LearnParOptions
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    opts.lrate = 1f
  	val nn = new ParLearnerF(
  	    new MatSource(Array(mat0), opts), 
  	    opts, mkGLMModel _,
  	    opts, mkRegularizer _,
  	    opts, mkUpdater _, 
  	    null, null,
  	    opts)
    (nn, opts)
  }
  
  def learnPar(mat0:Mat):(ParLearnerF, LearnParOptions) = learnPar(mat0, 0)
  
  def learnPar(mat0:Mat, targ:Mat, d:Int) = {
    val opts = new LearnParOptions
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    opts.lrate = 1f
    if (opts.links == null) opts.links = izeros(targ.nrows,1)
    opts.links.set(d)
    val nn = new ParLearnerF(
        new MatSource(Array(mat0, targ), opts), 
        opts, mkGLMModel _,
        opts, mkRegularizer _,
        opts, mkUpdater _, 
        null, null,
        opts)
    (nn, opts)
  }
  
  def learnPar(mat0:Mat, targ:Mat):(ParLearnerF, LearnParOptions) = learnPar(mat0, targ, 0)
  
  class LearnFParOptions extends ParLearner.Options with GLM.Opts with SFileSource.Opts with ADAGrad.Opts with L1Regularizer.Opts
  
  def learnFParx(
  		nstart:Int=FileSource.encodeDate(2012,3,1,0), 
  		nend:Int=FileSource.encodeDate(2012,12,1,0), 
  		d:Int = 0
  		) = {
  	val opts = new LearnFParOptions;
  	opts.lrate = 1f;
  	val nn = new ParLearnerxF(
  			null,
  			(dopts:DataSource.Opts, i:Int) => Experiments.Twitter.twitterWords(nstart, nend, opts.nthreads, i),
  			opts, mkGLMModel _,
  			opts, mkRegularizer _,
  			opts, mkUpdater _,
  			null, null,
  			opts
  			)
  	(nn, opts)
  }
  
  def learnFPar(
  		nstart:Int=FileSource.encodeDate(2012,3,1,0), 
  		nend:Int=FileSource.encodeDate(2012,12,1,0), 
  		d:Int = 0
  		) = {	
  	val opts = new LearnFParOptions;
  	opts.lrate = 1f;
  	val nn = new ParLearnerF(
  			Experiments.Twitter.twitterWords(nstart, nend),
  			opts, mkGLMModel _, 
  			opts, mkRegularizer _,
  			opts, mkUpdater _,
  			null, null,
  			opts
  			)
  	(nn, opts)
  }
}

