package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._


abstract class Updater {
  var model:Model = null
  var modelmats:Array[Mat] = null
  var updatemats:Array[Mat] = null

  
  def init(model0:Model) = {
    model = model0
    modelmats = model.modelmats
    updatemats = model.updatemats
  }
  
  def update(step:Long):Unit
  def updateM():Unit = {}
  def clear():Unit = {}
}


class IncNormUpdater(val opts:IncNormUpdater.Options = new IncNormUpdater.Options) extends Updater {
  
  var firstStep = 0f
  var rm:Mat = null
  var restart:Mat = null
  var started:Int = 0
  
  override def init(model0:Model) = {
    super.init(model0)
    restart = modelmats(0) + 1f
    rm = model0.modelmats(0).zeros(1,1)
  }
      
  def update(step:Long) = {
  	val mm = modelmats(0)
  	val um = updatemats(0)
  	val rr = if (step == 0) 0.9f else {
  	  if (firstStep == 0f) {
  	    firstStep = step
  	    0.9f
  	  } else {
  	    math.pow(firstStep / step, opts.power).toFloat
  	  }
  	}
  	if (modelmats.length > 1) {
  		val ms = modelmats(1)
  		val ums = updatemats(1)
  		ums ~ ums * rm.set(rr)
  		ms ~ ms * rm.set(1-rr)
  		ms ~ ms + ums
  		um ~ um / ms
  	}
  	um ~ um * rm.set(rr)
  	mm ~ mm * rm.set(1-rr)
    mm ~ mm + um 
    mm ~ mm / sum(mm,2)
    if (opts.warmup > 0) {
      if (started == 0 && step > opts.warmup) {
        restart <-- mm
        started = 1
      }
      if (started == 1 && step > 2*opts.warmup) {
        mm <-- mm - restart
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

class BatchNormUpdater(val opts:BatchNormUpdater.Options = new BatchNormUpdater.Options) extends Updater {
  var accumulators:Array[Mat] = null
  
  override def init(model0:Model) = {
    super.init(model0)
    modelmats = model.modelmats
    updatemats = model.updatemats
    accumulators = new Array[Mat](updatemats.length)
    for (i <- 0 until accumulators.length) {
    	accumulators(i) = updatemats(i).zeros(updatemats(i).nrows, updatemats(i).ncols)
    }
  }
     
  def update(step:Long) = {
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
    val mm = modelmats(0)
    mm ~ accumulators(0) / sum(accumulators(0),2)
  }
}


class IncMultUpdater(val opts:IncMultUpdater.Options = new IncMultUpdater.Options) extends Updater {
  
  var firstStep = 0f
  var rm:Mat = null
  
  override def init(model0:Model) = {
    super.init(model0)
    rm = model0.modelmats(0).zeros(1,1)
  }
      
  def update(step:Long) = {
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
    um ~ um * rm.set(rr)
//    println("rr=%g, %g %g" format (rr, mini(mini(um,1),2).dv, maxi(maxi(um,1),2).dv))
    ln(mm, mm)
//    println("mm=%g %g" format (mini(mini(mm,1),2).dv, maxi(maxi(mm,1),2).dv))
    mm ~ mm * rm.set(1-rr)
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

class BatchMultUpdater(val opts:BatchMultUpdater.Options = new BatchMultUpdater.Options) extends Updater {
  var accumulators:Array[Mat] = null
  
  override def init(model0:Model) = {
    super.init(model0)
    accumulators = new Array[Mat](updatemats.length)
    for (i <- 0 until updatemats.length)
    	accumulators(i) = updatemats(i).zeros(updatemats(i).nrows, updatemats(i).ncols)
  }
     
  def update(step:Long) = {
    for (i <- 0 until updatemats.length)
	accumulators(i) ~ accumulators(i) + updatemats(i) 
  }
  
  override def clear() = {
    for (i <- 0 until updatemats.length)
	accumulators(i).clear
  }
  
  override def updateM():Unit = {
    val mm = modelmats(0)
    accumulators(0) ~ accumulators(0) / accumulators(1)
    min(max(accumulators(0), 0.1f, accumulators(0)), 10f, accumulators(0))
    mm ~ mm ∘ accumulators(0)
  }
}


class ADAGradUpdater(opts:ADAGradUpdater.Options = new ADAGradUpdater.Options) extends Updater {
  
  val options = opts
  var nsteps = 0f
  var modelmat:Mat = null
  var updatemat:Mat = null
  
  var sumSq:Mat = null 

  override def init(model0:Model) = {
    model = model0
	  modelmat = model.modelmats(0)
	  updatemat = model.updatemats(0)
    nsteps = options.initnsteps 
    if (sumSq.asInstanceOf[AnyRef] == null) {
      sumSq = modelmat.ones(size(modelmat,1), size(modelmat,2)) * options.initsumsq
    } else {
    	sumSq(?) = options.initsumsq
    }
  } 
  
	def update(step:Long):Unit = update1(step)
	
	def update1(step:Long):Unit =	{

	  val nw = (options.gradwindow/step)
	  val ee = options.exponent
	  val alpham = options.alpha
	  sumSq  ~ (sumSq * (nw-1) + updatemat *@ updatemat) * (1/nw)
	  if (nsteps > options.waitsteps) {
	  	val tmp = (sumSq*nsteps) ^ ee
	  	modelmat ~ modelmat + ((updatemat / tmp) * alpham)
	  }
	  nsteps += step
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
	  		ftmp0.data(indx) = nsteps * fsumSq.data(indx)
	  		j += 1
	  	}
	    i += 1
	  }	  
	  if (nsteps > options.waitsteps) {
	  	ftmp1 ~ ftmp0 ^ options.exponent
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
    
  }
}

object ADAGradUpdater {
  class Options extends Updater.Options {
    var gradwindow:FMat = 1e6f
    var alpha:FMat = 100f
    var exponent:FMat = 0.5f
    var initnsteps:Float = 1000f
    var initsumsq:Float = 1e-4f
    var waitsteps = 200000
    var minmodel = 1e-7f
  }
}

object Updater {
  class Options {
    
  }
}
