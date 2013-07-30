package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._


trait Updater {
  def update(step:Long):Unit
  def updateM():Unit = {}
  def init(model:Model):Unit
}

abstract class BatchUpdater extends Updater {
	def updateM():Unit;
}

class BatchMultUpdater(opts:BatchMultUpdater.Options = new BatchMultUpdater.Options) extends BatchUpdater {
  var model:Model = null
  var modelmats:Array[Mat] = null
  var updatemats:Array[Mat] = null
  var accumulators:Array[Mat] = null
  
  def init(model0:Model) = {
    model = model0
    modelmats = model.modelmats
    accumulators = new Array[Mat](updatemats.size)
    for (i <- 0 until updatemats.size) accumulators(i) = updatemats(i).zeros(updatemats(i).nrows, updatemats(i).ncols)
  }
  
  def clear() = {
    for (i <- 0 until accumulators.size) {
      accumulators(i).clear
    }
  }
  
  def update(step:Long) = {
    for (i <- 0 until updatemats.size) {
      accumulators(i) ~ accumulators(i) + updatemats(i)
    }   
  }
  
  override def updateM():Unit = {
    val modelmat = modelmats(0)
    modelmat ~ modelmat *@ (updatemats(0) / updatemats(1))
  }
}

class ADAGradUpdater(opts:ADAGradUpdater.Options = new ADAGradUpdater.Options) extends Updater {
  
  val options = opts
  var nsteps = 0f
  var model:Model = null
  var modelmat:Mat = null
  var updatemat:Mat = null
  
  var sumSq:Mat = null 

  def init(model0:Model) = {
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
