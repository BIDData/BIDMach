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
      
  def update(step:Long) = {
  	val modelmat = modelmats(0)
  	val updatemat = updatemats(0)
  	val rr = if (step == 0) 1f else {
  	  if (firstStep == 0f) {
  	    firstStep = step
  	    1f
  	  } else {
  	    step / firstStep
  	  }
  	}
    modelmat ~ modelmat + updatemat / (sum(updatemat,2) * rr) 
    modelmat ~ modelmat / sum(modelmat,2)
  }
  
  override def clear() = {
	  firstStep = 0f
  }
}

class BatchNormUpdater(val opts:BatchNormUpdater.Options = new BatchNormUpdater.Options) extends Updater {
  var accumulator:Mat = null
  
  override def init(model0:Model) = {
    super.init(model0)
    accumulator = updatemats(0).zeros(updatemats(0).nrows, updatemats(0).ncols)
  }
     
  def update(step:Long) = {
    accumulator ~ accumulator + updatemats(0) 
  }
  
  override def clear() = {
	  accumulator.clear
  }
  
  override def updateM():Unit = {
    val modelmat = modelmats(0)
    modelmat ~ accumulator / sum(accumulator,2)
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
    
  }
}

object BatchNormUpdater {
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
