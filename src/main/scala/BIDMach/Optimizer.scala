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

  var sumSq:Mat = null
  var tmp0:Mat = null
  var tmp1:Mat = null
  var tmp2:Mat = null
  var nsteps:Float = 1e5f
  var options:Learner.Options = null
  
  def init(learner:Learner, model:Mat) = {
  	sumSq = model.ones(size(model,1), size(model,2)) * 0.0001f
  	nsteps = 1e5f
    options = learner.options
  }
  
	def update(model:Mat, update:Mat, step:Int):Unit = {
	  val nw = (options.gradwindow/step)
	  tmp0 = checkSize(tmp0, model)
	  tmp1 = checkSize(tmp1, model)
	  tmp2 = checkSize(tmp2, model)
/*	  val fmodel = model.asInstanceOf[FMat]
	  val fupdate = update.asInstanceOf[FMat]
	  tmp0 = checkSize(tmp0, fmodel).asInstanceOf[FMat]
	  tmp1 = checkSize(tmp1, fmodel).asInstanceOf[FMat]

	  var i = 0 
	  while (i < model.length) {
	    sumSq.data(i) = (1/nw)*(fupdate.data(i)*fupdate.data(i) + (nw-1)*sumSq.data(i))
	    i += 1
	  }	  */
	  sumSq ~ (tmp1 ~ (tmp0 ~ update *@ update) + (tmp2 ~ sumSq*(nw-1))) * (1/nw)
	  sqrt(tmp0 ~ sumSq*nsteps, tmp1)
/*	  i = 0 
	  while (i < model.length) {
	    fmodel.data(i) = fmodel.data(i) + options.alpha*fupdate.data(i)/tmp1.data(i)
	    i += 1
	  }	 */
	  model ~ model + (tmp1 ~ (tmp0 ~ update /@ tmp1) * options.alpha);
	  Mat.nflops += 7L * model.length 
	  nsteps += step
	}
}