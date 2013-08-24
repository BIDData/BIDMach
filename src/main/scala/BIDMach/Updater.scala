package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._


abstract class Updater(val opts:Updater.Options = new Updater.Options) {
  var model:Model = null
  
  def init(model0:Model) = {
    model = model0 
  }
  
  def update(step:Long):Unit
  def updateM():Unit = {}
  def clear():Unit = {}
}


class IncNormUpdater(override val opts:IncNormUpdater.Options = new IncNormUpdater.Options) extends Updater {
  
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

class BatchNormUpdater(override val opts:BatchNormUpdater.Options = new BatchNormUpdater.Options) extends Updater {
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


class IncMultUpdater(override val opts:IncMultUpdater.Options = new IncMultUpdater.Options) extends Updater {
  
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

class TelescopingUpdater(override val opts:TelescopingUpdater.Options = new TelescopingUpdater.Options) extends Updater {
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


class ADAGradUpdater(opts:ADAGradUpdater.Options = new ADAGradUpdater.Options) extends Updater {
  
  val options = opts
  var nsteps = 0f
  var firstStep = 0f
  var modelmat:Mat = null
  var updatemat:Mat = null  
  var sumSq:Mat = null 
  var stepn:Mat = null

  override def init(model0:Model) = {
    model = model0
	  modelmat = model.modelmats(0)
	  updatemat = model.updatemats(0) 
    if (sumSq.asInstanceOf[AnyRef] == null) {
      sumSq = modelmat.ones(size(modelmat,1), size(modelmat,2)) *@ options.initsumsq
    } else {
    	sumSq(?) = options.initsumsq
    }
    nsteps = options.initnsteps
    stepn = modelmat.zeros(1,1)
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
	  val ve = options.vecExponent
	  val te = options.timeExponent
	  val alpham = options.alpha
	  val nw = 1f / stepn
	  val newsquares = updatemat *@ updatemat
	  newsquares ~ newsquares *@ nw
	  sumSq  ~ sumSq *@ (1f - nw)
	  sumSq ~ sumSq + newsquares
	  if (options.waitsteps < nsteps) {
	  	val tmp = sumSq ^ ve
	  	tmp ~ tmp *@ (stepn ^ te)
	  	modelmat ~ modelmat + ((updatemat / tmp) *@ alpham)
	  }
	}
	
  def update2(step:Long):Unit =	{
	  val nwm = options.gradwindow/step
	  val alpham = options.alpha
	  val fmodel = modelmat.asInstanceOf[FMat]
	  val fsumSq = sumSq.asInstanceOf[FMat]
	  val fupdate = updatemat.asInstanceOf[FMat]
	  val ftmp0 = fmodel + 0
	  val ftmp1 = fmodel + 1

	  var i = 0 
	  while (i < modelmat.ncols) {
	  	var j = 0
	  	while (j < modelmat.nrows) {
	  	  val indx = j + i * modelmat.nrows
	  	  val nw = if (nwm.length > 1) nwm(i) else nwm(0)
	  		fsumSq.data(indx) = 1/nw*(fupdate.data(indx)*fupdate.data(indx) + (nw-1)*fsumSq.data(indx))
	  		ftmp0.data(indx) = fsumSq.data(indx) * nsteps
	  		j += 1
	  	}
	    i += 1
	  }	  
	  if (nsteps > options.waitsteps) {
	  	ftmp1 ~ ftmp0 ^ options.vecExponent
	  	i = 0 
	  	while (i < modelmat.ncols) {
	  		var j = 0
	  		while (j < modelmat.nrows) {
	  			val indx = j + i * modelmat.nrows
	  			val alpha = if (alpham.length > 1) alpham(i) else alpham(0)
	  			fmodel.data(indx) = fmodel.data(indx) + alpha*fupdate.data(indx)/ftmp1.data(indx)
	  			j += 1
	  		}
	  		i += 1
	  	}	
	  }
	  Mat.nflops += 8L * modelmat.length 
	  nsteps += step
	}
}

object IncNormUpdater {
  class Options extends Updater.Options {
    var warmup = 0L 
    var power = 0.98f
  }
}

object IncMultUpdater {
  class Options extends Updater.Options {
    var warmup = 0L 
    var power = 0.98f
  }
}

object BatchNormUpdater {
  class Options extends Updater.Options {
    
  }
}

object BatchMultUpdater {
  class Options extends Updater.Options {
    var eps = 1e-12
    
  }
}

object TelescopingUpdater {
  class Options extends Updater.Options {
    val factor = 1.5f
  }
}

object ADAGradUpdater {
  class Options extends Updater.Options {
    var gradwindow:FMat = 1e6f
    var alpha:FMat = 100f
    var vecExponent:FMat = 0.5f
    var timeExponent:FMat = 0.5f
    var initsumsq:FMat = 1e-8f
    var initnsteps = 1000f
    var waitsteps = 200000
  }
}

object Updater {
  class Options {
    
  }
}
