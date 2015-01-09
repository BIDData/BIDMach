package BIDMach.updaters
 
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._

class ADAGrad(override val opts:ADAGrad.Opts = new ADAGrad.Options) extends Updater {
  
  var firstStep = 0f
  var modelmats:Array[Mat] = null
  var updatemats:Array[Mat] = null  
  var sumSq:Array[Mat] = null 
  var stepn:Mat = null
  var mask:Mat = null
  var ve:Mat = null
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
    te = mm.zeros(opts.texp.nrows, opts.texp.ncols);
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
    stepn.set(nsteps+1);
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
    		if (java.lang.Double.isNaN(sum(sum(ss)).dv)) throw new RuntimeException("ADA 0 "+i);
    		val tmp = ss ^ ve;
    		if (java.lang.Double.isNaN(sum(sum(tmp)).dv)) throw new RuntimeException("ADA 1 "+i);
    		tmp ~ tmp *@ (stepn ^ te);
    		if (java.lang.Double.isNaN(sum(sum(tmp)).dv)) throw new RuntimeException("ADA 2 "+i);
    		tmp ~ tmp + opts.epsilon;
    		tmp ~ um / tmp;
    		if (java.lang.Double.isNaN(sum(sum(tmp)).dv)) throw new RuntimeException("ADA 3 "+i);
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
}

