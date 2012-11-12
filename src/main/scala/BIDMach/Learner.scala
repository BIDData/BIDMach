package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Plotting._
import Learner._
import scala.collection.immutable.List

case class Learner(datamat0:Mat, targetmat0:Mat, datatest0:Mat, targtest0:Mat, 
		model:Model, regularizer:Regularizer, updater:Updater, val opts:Learner.Options = new Learner.Options) {

  val n = datamat0.ncols
  val options = opts
  var tscores:List[Double] = List()
  var tscorex:List[Double] = List()
  var tsteps:List[Double] = List()
  var targetmat:Mat = null

  def run() = {

  	var done:Boolean = false
  	var ipass = 0
  	var llest = 0.0
  	var llder = 0.0
  	var llold = 0.0
  	var tsecs:Double = options.secprint
  	var nsteps:Long = 0
  	val (targetm, targettest) = model.initmodel(datamat0, targetmat0, datatest0, targtest0)
  	targetmat = targetm
  	updater.initupdater
  	if (regularizer != null) regularizer.initregularizer
  	var blocksize = options.blocksize
  	tic
  	while (ipass < options.npasses && ! done) {
  		ipass += 1
//  		if (ipass > 2) blocksize = size(datamat0, 2) //math.min(size(datamat0, 2), 2*blocksize)
  		var i = 0
  		val nw = math.round(options.memwindow/blocksize).asInstanceOf[Double]
  		val nww = options.convwindow/blocksize
  		while (i < n && ! done) {
  			var iend = math.min(n, i+blocksize)
  			nsteps += iend - i
  			var dslice = datamat0(?, i->iend)
  			var tslice = targetmat(?, i->iend)
  			val nmodel = model.asInstanceOf[NMFmodel]
//  	    val v0 = nmodel.eval(dslice, model.modelmat, tslice)
  			model.gradfun(dslice, tslice)
  			val (tll, tllx) = model.eval(dslice, tslice)
  			targetmat(?, i->iend) = tslice
  			if (regularizer != null) regularizer.compute(1.0f*(iend-i)/n)
  			val v1 = model.eval(dslice, tslice) 			
  			updater.update(iend-i)
  			val v2 = model.eval(dslice, tslice)
//  			println("delta = %f, %f, %f, nw=%f" format (tll, v1._1, v2._1, nw))

  			llest = (1/(nw+1))*(tll + nw*llest)
  			llder = (1/(nww+1))*(tll-llold + nww*llder)
  			llold = tll
  			i += blocksize
  			if (llder > 0 && llder < options.convslope) {
  				done = true
  			}
  			if (toc >= tsecs || done || (i >= n)) {
  			  val tmp = model.asInstanceOf[FactorModel].options.uiter
  			  model.asInstanceOf[FactorModel].options.uiter = 20
  			  val (llx, llxa) = model.eval(datatest0, targettest)
  			  model.asInstanceOf[FactorModel].options.uiter = tmp
  				println("pass=%d, n=%dk t=%3.1f secs, ll=%5.4f, llx=%5.4f(u%5.4f)" format (ipass, nsteps/1000, toc, llest, llx, llxa))
//  				println("normu=%f, normm=%f, block=%d" format (norm(targetmat), norm(model.modelmat), blocksize))
  				tscores = tscores :+ -llest
  				tscorex = tscorex :+ -llx
  				tsteps = tsteps :+ nsteps.asInstanceOf[Double]
  			  tsecs += options.secprint
  			}
  		}
  	}
  }
}


object Learner {
	class Options {
		var blocksize:Int = 8000
		var npasses:Int = 100
	  var memwindow:Double = 40000
		var convwindow:Double = 40000	
		var convslope:Double = -1e-6
		var secprint:Double = 1
		var eps:Float = 1e-8f
  }
  
  def fsqrt(v:Float):Float = math.sqrt(v).asInstanceOf[Float]
  
  def mapfun2x2(fn:(Float, Float)=>(Float, Float), in0:FMat, in1:FMat, out0:FMat, out1:FMat) = {
    if (in0.nrows != in1.nrows || in0.nrows != out0.nrows || in0.nrows != out1.nrows ||
        in0.ncols != in1.ncols || in0.ncols != out0.ncols || in0.ncols != out1.ncols) {
      throw new RuntimeException("dimensions mismatch")
    }
    var i = 0
    while (i < in0.length) {
      val (v1, v2) = fn(in0.data(i), in1.data(i))
      out0.data(i) = v1
      out1.data(i) = v2
      i += 1
    }
  }
  def mapfun2x1(fn:(Float, Float)=>Float, in0:FMat, in1:FMat, out0:FMat) = {
    if (in0.nrows != in1.nrows || in0.nrows != out0.nrows ||
        in0.ncols != in1.ncols || in0.ncols != out0.ncols) {
      throw new RuntimeException("dimensions mismatch")
    }
    var i = 0
    while (i < in0.length) {
      out0.data(i) = fn(in0.data(i), in1.data(i))
      i += 1
    }
  }
}
