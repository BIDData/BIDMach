package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import Learner._

trait Optimizer {
	def init(learner:Learner, model:Mat):Unit
  def update(model:Mat, update:Mat, step:Int):Unit
}

class ADAGradOptimizer extends Optimizer {

  var sumSq:FMat = null
  var tmp0:FMat = null
  var tmp1:FMat = null
  var nsteps:Float = 1e5f
  var options:Learner.Options = null
  
  def init(learner:Learner, model:Mat) = {
  	sumSq = 0.0001f*ones(size(model,1), size(model,2))
  	nsteps = 1e5f
    options = learner.options
  }
  
	def update(model:Mat, update:Mat, step:Int):Unit = {
	  val fmodel = model.asInstanceOf[FMat]
	  val fupdate = update.asInstanceOf[FMat]
	  tmp0 = checkSize(tmp0, fmodel)
	  tmp1 = checkSize(tmp1, fmodel)
	  val nw = (options.gradwindow/step)
	  var i = 0 
	  while (i < model.length) {
	    sumSq.data(i) = (1/nw)*(fupdate.data(i)*fupdate.data(i) + (nw-1)*sumSq.data(i))
	    i += 1
	  }	  
	  sqrt(tmp0 ~ sumSq*nsteps, tmp1)
	  i = 0 
	  while (i < model.length) {
	    fmodel.data(i) = fmodel.data(i) + options.alpha*fupdate.data(i)/tmp1.data(i)
	    i += 1
	  }	 
	  Mat.nflops += 7L * model.length 
	  nsteps += step
	}
}