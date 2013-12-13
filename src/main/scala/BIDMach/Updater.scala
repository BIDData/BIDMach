package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._


abstract class Updater(val opts:Updater.Opts = new Updater.Options) {
  var model:Model = null
  
  def init(model0:Model) = {
    model = model0 
  }
  
  def update(step:Long):Unit
  def updateM():Unit = {}
  def clear():Unit = {}
}

class IncNormUpdater(override val opts:IncNormUpdater.Opts = new IncNormUpdater.Options) extends Updater(opts) {
  
  var firstStep = 0f
  var rm:Mat = null
  var restart:Mat = null
  var started:Int = 0
  
  override def init(model0:Model) = {
  	super.init(model0)
    val modelmats = model0.modelmats
    val updatemats = model0.updatemats
    restart = modelmats(0) + 1f
    rm = model0.modelmats(0).zeros(1,1)
    firstStep = 0f
  }
      
  def update(step:Long) = {
  	val modelmats = model.modelmats
  	val updatemats = model.updatemats
  	val mm = modelmats(0)
  	val um = updatemats(0)
  	val rr = if (step == 0) 0.99f else {
  	  if (firstStep == 0f) {
  	    firstStep = step
  	    0.99f
  	  } else {
  	    math.pow(firstStep / step, opts.power).toFloat
  	  }
  	}
  	if (modelmats.length > 1) {
  		val ms = modelmats(1)
  		val ums = updatemats(1)
//  		println("ums0 %g %g %g" format (rr, mini(mini(ums,1),2).dv, maxi(maxi(ums,1),2).dv))
  		ums ~ ums *@ rm.set(rr)
//  		println("ums1 %g %g %g" format (rr, mini(mini(ums,1),2).dv, maxi(maxi(ums,1),2).dv))
  		ms ~ ms *@ rm.set(1-rr)
//  		println("ums2 %g %g %g" format (rr, mini(mini(ums,1),2).dv, maxi(maxi(ums,1),2).dv))
  		ms ~ ms + ums
//  		println("ums3 %g %g %g" format (rr, mini(mini(ums,1),2).dv, maxi(maxi(ums,1),2).dv))
  		um ~ um / ms
//  		println("um %g %g" format (mini(mini(um,1),2).dv, maxi(maxi(um,1),2).dv))
  	}
  	um ~ um *@ rm.set(rr)
  	mm ~ mm *@ rm.set(1-rr)
    mm ~ mm + um 
    mm ~ mm / sum(mm,2)
    if (opts.warmup > 0) {
      if (started == 0 && step > opts.warmup) {
        restart <-- mm
        started = 1
      }
      if (started == 1 && step > 2*opts.warmup) {
        mm ~ mm - restart
        max(mm, 0f, mm)
        mm ~ mm / sum(mm,2)
        started = 2
      }
    }
  }
  
  override def clear() = {
	  firstStep = 0f
  }
}

class BatchNormUpdater(override val opts:BatchNormUpdater.Opts = new BatchNormUpdater.Options) extends Updater {
  var accumulators:Array[Mat] = null
  
  override def init(model0:Model) = {
    super.init(model0)
    val modelmats = model.modelmats
    val updatemats = model.updatemats
    accumulators = new Array[Mat](updatemats.length)
    for (i <- 0 until accumulators.length) {
    	accumulators(i) = updatemats(i).zeros(updatemats(i).nrows, updatemats(i).ncols)
    }
  }
     
  def update(step:Long) = {
  	val updatemats = model.updatemats
    for (i <- 0 until accumulators.length) {
    	accumulators(i) ~ accumulators(i) + updatemats(i) 
    }
  }
  
  override def clear() = {
	  for (i <- 0 until accumulators.length) {
	  	accumulators(i).clear
	  }
  }
  
  override def updateM():Unit = {
    val mm = model.modelmats(0)
    mm ~ accumulators(0) / accumulators(1)
    mm ~ mm / sum(mm,2)
    clear
  }
}


class IncMultUpdater(override val opts:IncMultUpdater.Opts = new IncMultUpdater.Options) extends Updater {
  
  var firstStep = 0f
  var rm:Mat = null
  
  override def init(model0:Model) = {
    super.init(model0)
    rm = model0.modelmats(0).zeros(1,1)
  }
      
  def update(step:Long) = {
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

class TelescopingUpdater(override val opts:TelescopingUpdater.Opts = new TelescopingUpdater.Options) extends Updater {
	var accumulators:Array[Mat] = null
  var firstStep = 0L
  var nextStep = 10L
  var nextCount = 0L
  var rm:Mat = null
  
  override def init(model0:Model) = {
  	super.init(model0)
    val modelmats = model0.modelmats
    val updatemats = model0.updatemats
    rm = model0.modelmats(0).zeros(1,1)
    accumulators = new Array[Mat](updatemats.length)
    for (i <- 0 until updatemats.length) {
    	accumulators(i) = updatemats(i).zeros(updatemats(i).nrows, updatemats(i).ncols)
    }
  	firstStep = 0L
    nextStep = 10L
    nextCount = 0L
  }
	
	def update(step:Long) = {
	  if (firstStep == 0 && step > 0) {
	    firstStep = step
	  }
	  val updatemats = model.updatemats
    for (i <- 0 until updatemats.length) {
	    accumulators(i) ~ accumulators(i) + updatemats(i) 
    }
	  if (step >= nextCount) {
	    model.modelmats(0) ~ accumulators(0) / accumulators(1)
	    nextStep = (nextStep * opts.factor).toLong
	    nextCount = step + nextStep
	  }
  }
  
  override def clear() = {
	  for (i <- 0 until accumulators.length) {
     	accumulators(i).clear
	  }
  }
}


class GradUpdater(override val opts:GradUpdater.Opts = new GradUpdater.Options) extends Updater {
  
  var firstStep = 0f
  var modelmat:Mat = null
  var updatemat:Mat = null  
  var sumSq:Mat = null 
  var stepn:Mat = null
  var mask:Mat = null
  var ve:Mat = null
	var te:Mat = null
	var alpha:Mat = null

  override def init(model0:Model) = {
    model = model0
	  modelmat = model.modelmats(0)
	  updatemat = model.updatemats(0) 
	  mask = opts.mask
    stepn = modelmat.zeros(1,1)
    te = modelmat.zeros(opts.timeExponent.nrows, opts.timeExponent.ncols)
    alpha = modelmat.zeros(opts.alpha.nrows, opts.alpha.ncols)
    te <-- opts.timeExponent
    alpha <-- opts.alpha
  } 
  
	def update(step:Long):Unit = {
	  val nsteps = if (step == 0) 1f else {
  	  if (firstStep == 0f) {
  	    firstStep = step
  	    1f
  	  } else {
  	    step / firstStep
  	  }
  	}
	  stepn.set(1f/nsteps)
	  if (opts.waitsteps < nsteps) {
	  	val tmp = updatemat *@ (alpha *@ (stepn ^ te))
 	  	modelmat ~ modelmat + tmp
	  	if (mask != null) modelmat ~ modelmat *@ mask
	  }
	}
}


class ADAGradUpdater(override val opts:ADAGradUpdater.Opts = new ADAGradUpdater.Options) extends Updater {
  
  var firstStep = 0f
  var modelmat:Mat = null
  var updatemat:Mat = null  
  var sumSq:Mat = null 
  var stepn:Mat = null
  var mask:Mat = null
  var ve:Mat = null
	var te:Mat = null
	var alpha:Mat = null

  override def init(model0:Model) = {
    model = model0
	  modelmat = model.modelmats(0)
	  updatemat = model.updatemats(0) 
	  mask = opts.mask
    if (sumSq.asInstanceOf[AnyRef] == null) {
      sumSq = modelmat.ones(size(modelmat,1), size(modelmat,2)) *@ opts.initsumsq
    } else {
    	sumSq.set(opts.initsumsq)
    }
    stepn = modelmat.zeros(1,1)
    ve = modelmat.zeros(opts.vecExponent.nrows, opts.vecExponent.ncols)
    te = modelmat.zeros(opts.timeExponent.nrows, opts.timeExponent.ncols)
    alpha = modelmat.zeros(opts.alpha.nrows, opts.alpha.ncols)
    ve <-- opts.vecExponent
    te <-- opts.timeExponent
    alpha <-- opts.alpha
  } 
  
	def update2(step:Long):Unit = {
	  val nsteps = if (step == 0) 1f else {
  	  if (firstStep == 0f) {
  	    firstStep = step
  	    1f
  	  } else {
  	    step / firstStep
  	  }
  	}
	  stepn.set(nsteps)
	  val nw = 1f / stepn
	  val newsquares = updatemat *@ updatemat
	  newsquares ~ newsquares *@ nw
	  sumSq  ~ sumSq *@ (1f - nw)
	  sumSq ~ sumSq + newsquares
	  if (opts.waitsteps < nsteps) {
	  	val tmp = sumSq ^ ve
	  	tmp ~ tmp *@ (stepn ^ te)
	  	tmp ~ tmp + opts.epsilon
	  	modelmat ~ modelmat + ((updatemat / tmp) *@ alpha)
	  	if (mask != null) modelmat ~ modelmat *@ mask
	  }
	}
	
	def update(step:Long):Unit = {
	  val nsteps = if (step == 0) 1f else {
  	  if (firstStep == 0f) {
  	    firstStep = step
  	    1f
  	  } else {
  	    step / firstStep
  	  }
  	}
	  stepn.set(nsteps)
	  val nw = 1f / stepn
	  val newsquares = updatemat *@ updatemat
	  newsquares ~ newsquares *@ nw
	  sumSq  ~ sumSq *@ (1f - nw)
	  sumSq ~ sumSq + newsquares
	  if (opts.waitsteps < nsteps) {
	  	val tmp = sumSq ^ ve
	  	tmp ~ tmp *@ (stepn ^ te)
	  	tmp ~ tmp + opts.epsilon
	  	tmp ~ updatemat / tmp
	  	tmp ~ tmp *@ alpha
	  	modelmat ~ modelmat + tmp
	  	if (mask != null) modelmat ~ modelmat *@ mask
	  }
	}
}



object IncNormUpdater {
  trait Opts extends Updater.Opts {
    var warmup = 0L 
    var power = 0.3f
  }
  
  class Options extends Opts {}
}

object IncMultUpdater {
  trait Opts extends Updater.Opts {
    var warmup = 0L 
    var power = 0.3f
  }
  
  class Options extends Opts {}
}

object BatchNormUpdater {
  trait Opts extends Updater.Opts {
  }
  
  class Options extends Opts {}
}

object BatchMultUpdater {
  trait Opts extends Updater.Opts {
    var eps = 1e-12   
  }
  
  class Options extends Opts {}
}

object TelescopingUpdater {
  trait Opts extends Updater.Opts {
    val factor = 1.5f
  }
  
  class Options extends Opts {}
}

object GradUpdater {
  trait Opts extends Updater.Opts {
    var alpha:FMat = 1f
    var timeExponent:FMat = 0.5f
    var waitsteps = 2
    var mask:FMat = null
  }
  
  class Options extends Opts {}
}


object ADAGradUpdater {
  trait Opts extends GradUpdater.Opts {
    var vecExponent:FMat = 0.5f
    var epsilon = 1e-15f
    var initsumsq = 1e-8f
  }
  
  class Options extends Opts {}
}

object Updater {
  trait Opts {  
  }
  
  class Options extends Opts {}
}
