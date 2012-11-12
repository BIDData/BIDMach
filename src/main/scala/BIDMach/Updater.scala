package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import Learner._

trait Updater {
  def update(step:Int):Unit;
  def initupdater:Unit;
}

class MultUpdater(val model:FactorModel, opts:Updater.Options = new Updater.Options) extends Updater {
  val options = opts  
  var nsteps = 0
  var updatesum:Mat = null
  var updateDenomsum:Mat = null
  var ratio = blank
  var msum = blank
  
  def initupdater = {
    nsteps = 0 
    val modelmat = model.modelmat
    updatesum = modelmat.zeros(size(modelmat,1),size(modelmat,2))
    updateDenomsum = modelmat.zeros(size(modelmat,1),size(modelmat,2))
  }
  
  def update(step:Int):Unit =	{
  	val nw = (options.gradwindow/step)
	  val modelmat = model.modelmat
	  val updatemat = model.updatemat
	  val updateDenom = model.updateDenom
	  val sdim = 1f/math.sqrt(size(modelmat,2)).asInstanceOf[Float]
//	  updatesum ~ updatesum + updatemat
//	  updateDenomsum ~ updateDenomsum + updateDenom
//	  if (math.sqrt(nsteps) % 1 < 1e-9) {
	  
	  	ratio = ratio ~ updatemat /@ updateDenom
	  	max(0.2f, ratio, ratio)
	  	min(5f, ratio, ratio)
	  	modelmat ~ modelmat *@ ratio
//	  	updatesum.clear
//	  	updateDenomsum.clear
//	  } 
	  	msum = sum(modelmat, 2, msum)
  	  msum ~ msum *@ sdim
	  modelmat ~ modelmat /@ msum
	  nsteps += 1
  }
}

class ADAMultUpdater(val model:FactorModel, opts:Updater.Options = new Updater.Options) extends Updater {

  var ratio = blank
  var msum = blank
  var tmp0 = blank
  var tmp1 = blank
  var diff = blank
  var sumSq:Mat = null
  val options = opts  
  var nsteps = options.initnsteps
  var lastdiff:Mat = null
  
	def update(step:Int):Unit = update1(step)
	
	def initupdater = {
    nsteps = options.initnsteps 
    val modelmat  = model.modelmat
    lastdiff = modelmat.zeros(size(modelmat,1),size(modelmat,2))
    nsteps = options.initnsteps 
    if (sumSq.asInstanceOf[AnyRef] == null) {
      sumSq = modelmat.ones(size(modelmat,1), size(modelmat,2)) * options.initsumsq
    } else {
    	sumSq(?) = options.initsumsq
    }
  }  
	
	def update1(step:Int):Unit =	{
  	val nw = (options.gradwindow/step)
	  val modelmat = model.modelmat
	  val updatemat = model.updatemat
	  val updateDenom = model.updateDenom
	  val sdim = 1f/math.sqrt(size(modelmat,2)).asInstanceOf[Float]
	  
	  val weight = options.alpha * math.sqrt(options.initnsteps/nsteps).asInstanceOf[Float]
//	  val weight = options.alpha * math.sqrt(math.sqrt(options.initnsteps/nsteps)).asInstanceOf[Float]
//  	val weight = options.alpha
  	ratio = ratio ~ updatemat /@ updateDenom
  	diff = ln(ratio, diff)
//	  diff = diff   ~ updatemat - updateDenom
/*	         diff   ~ diff *@ modelmat
	  tmp0 = tmp0   ~ diff *@ diff
	  tmp1 = tmp1   ~ sumSq * (nw-1) 
	         tmp0   ~ tmp0 + tmp1
	         sumSq  ~ tmp0 * (1/nw)*/
	         
//	  if (nsteps > options.waitsteps) {	  	
//	  	diff ~ diff /@ sqrt(tmp0 ~ sumSq*(options.alpha * nsteps / options.initnsteps), tmp1)
	  	diff ~ diff *@ weight
    	max(-1f, min(1f, diff, diff), diff)  	  
//    	println("mean diff=%f" format mean(abs(diff.asInstanceOf[FMat])(?)).dv)
  	  ratio = exp(diff, ratio)
      modelmat ~ modelmat *@ ratio
  	  max(options.minmodel, modelmat, modelmat)
//	  }

  	msum = sum(modelmat, 2, msum)
  	msum ~ msum *@ sdim
	  modelmat ~ modelmat /@ msum
	  nsteps += step
	}
}

class ADAGradUpdater(val model:Model, opts:Updater.Options = new Updater.Options) extends Updater {

  var tmp0 = blank
  var tmp1 = blank
  
  val options = opts
  var nsteps = 0f
  
  var sumSq:Mat = null 

  def initupdater = {
	  val modelmat = model.modelmat
    nsteps = options.initnsteps 
    if (sumSq.asInstanceOf[AnyRef] == null) {
      sumSq = modelmat.ones(size(modelmat,1), size(modelmat,2)) * options.initsumsq
    } else {
    	sumSq(?) = options.initsumsq
    }
  } 
  
	def update(step:Int):Unit = update1(step)
	
	def update1(step:Int):Unit =	{
	  val modelmat = model.modelmat
	  val updatemat = model.updatemat
	  val nw = (options.gradwindow/step)
	  tmp0 = tmp0   ~ updatemat *@ updatemat
	  tmp1 = tmp1   ~ sumSq * (nw-1) 
	         tmp0   ~ tmp0 + tmp1
	         sumSq  ~ tmp0 * (1/nw)
	  if (nsteps > options.waitsteps) {
	  	sqrt(tmp0 ~ sumSq*nsteps, tmp1)
	  	modelmat ~ modelmat + (tmp1 ~ (tmp0 ~ updatemat /@ tmp1) * options.alpha)
	  }
	  nsteps += step
	}
	
  def update2(step:Int):Unit =	{
    val modelmat = model.modelmat
    val updatemat = model.updatemat
	  val nw = (options.gradwindow/step)
	  val fmodel = modelmat.asInstanceOf[FMat]
	  val fsumSq = sumSq.asInstanceOf[FMat]
	  val fupdate = updatemat.asInstanceOf[FMat]
	  val ftmp0 = recycleTry(tmp0, fmodel).asInstanceOf[FMat]
	  val ftmp1 = recycleTry(tmp1, fmodel).asInstanceOf[FMat]

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
    var waitsteps = 200000
    var minmodel = 1e-7f
  }
}
