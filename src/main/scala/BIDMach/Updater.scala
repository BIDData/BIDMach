package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import Learner._

trait Updater {
  def update(step:Int):Unit
}

class ADAGradUpdater(val model:RegressionModel, opts:Updater.Options = new Updater.Options) extends Updater {

  var tmp0:Mat = null
  var tmp1:Mat = null
  var tmp2:Mat = null
  
  val modelmat = model.modelmat
  val updatemat = model.updatemat
  val options = opts
  var nsteps = options.initnsteps
  val sumSq = modelmat.ones(size(modelmat,1), size(modelmat,2)) * options.initsumsq

	def update(step:Int):Unit = update1(step)
	
	def update1(step:Int):Unit =	{
	  val nw = (options.gradwindow/step)
	  tmp0 = checkSize(tmp0, modelmat)
	  tmp1 = checkSize(tmp1, modelmat)
	  tmp2 = checkSize(tmp2, modelmat)
	  sumSq ~ (tmp1 ~ (tmp0 ~ updatemat *@ updatemat) + (tmp2 ~ sumSq*(nw-1))) * (1/nw)
	  if (nsteps > options.waitsteps) {
	  	sqrt(tmp0 ~ sumSq*nsteps, tmp1)
	  	modelmat ~ modelmat + (tmp1 ~ (tmp0 ~ updatemat /@ tmp1) * options.alpha)
	  }
	  nsteps += step
	}
	
  def update2(step:Int):Unit =	{
	  val nw = (options.gradwindow/step)
	  val fmodel = modelmat.asInstanceOf[FMat]
	  val fsumSq = sumSq.asInstanceOf[FMat]
	  val fupdate = updatemat.asInstanceOf[FMat]
	  val ftmp0 = checkSize(tmp0, fmodel).asInstanceOf[FMat]
	  val ftmp1 = checkSize(tmp1, fmodel).asInstanceOf[FMat]

	  var i = 0 
	  while (i < modelmat.length) {
	    fsumSq.data(i) = 1/nw*(fupdate.data(i)*fupdate.data(i) + (nw-1)*fsumSq.data(i))
	    ftmp0.data(i) = nsteps * fsumSq.data(i)
	    i += 1
	  }	  
	  if (nsteps > options.waitsteps) {
	  	sqrt(ftmp0, ftmp1)
	  	i = 0 
	  	while (i < modelmat.length) {
	  		fmodel.data(i) = fmodel.data(i) + options.alpha*fupdate.data(i)/ftmp1.data(i)
	  		i += 1
	  	}	
	  }
	  Mat.nflops += 8L * modelmat.length 
	  nsteps += step
	}
}

object Updater {
  class Options {
    var gradwindow:Float = 1e6f
    var alpha:Float = 100f
    var initnsteps:Float = 1000f
    var initsumsq:Float = 1e-4f
    var waitsteps = 100000
  }
}
