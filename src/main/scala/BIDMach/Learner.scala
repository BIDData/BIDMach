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
  	updater.initupdater(model)
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
//  			val v1 = model.eval(dslice, tslice) 			
  			updater.update(iend-i)
//  			val v2 = model.eval(dslice, tslice)
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
  
   def runpar() = {

  	val numthreads = options.numGPUthreads
  	var ipass = 0
  	var llest = 0.0
  	var llder = 0.0
  	var llold = 0.0
  	var tsecs:Double = options.secprint
  	var nsteps:Long = 0
  	val curdevice = 0
  	device(0)
  	val (targetm, targettest) = model.initmodel(datamat0, targetmat0, datatest0, targtest0) 
  	targetmat = targetm
  	updater.initupdater(model)
  	val models = new Array[FactorModel](numthreads)
//  	println("got here 1, device=%d" format curdevice)
  	val nmodel = model.asInstanceOf[FactorModel]
  	models(0) = nmodel
  	for (i <- 1 until numthreads) {
  	  device(i)
  	  if (curdevice != i) connect(curdevice)
  	  models(i) = model.make(model.options).asInstanceOf[FactorModel]
      models(i).initmodel(datamat0, ones(1,1), datatest0, ones(1,1))
  	}
//    println("got here 2")
  	device(curdevice)
  	updater.initupdater(model)
  	if (regularizer != null) regularizer.initregularizer
  	var blocksize = options.blocksize
  	tic
  	while (ipass < options.npasses) {
  		ipass += 1
  		var ipos = 0
  		val nw = math.round(options.memwindow/blocksize).asInstanceOf[Double]
  		val nww = options.convwindow/blocksize
  		while (ipos < n) {
//  			println("got here 3")
  			val ind = math.min(n, ipos+numthreads*blocksize)
  			nsteps += ind - ipos
  			val tlls = zeros(numthreads,1)
  			val tllxs = zeros(numthreads,1)
  			val done = zeros(numthreads,1) 
//  			var ithread = 0
  			for (ithread <- 0 until numthreads) {
  			  scala.actors.Actor.actor {
  			    device(ithread)
  			    if (ithread > 0) models(ithread).modelmat <-- model.modelmat
  			  	val iloc = ipos + ithread*blocksize
  			  	var iend = math.min(n, iloc+blocksize)
  			  	if (iloc < iend) {
  			  		var dslice = datamat0(?, iloc->iend)
  			  		var tslice = targetmat(?, iloc->iend)
  			  		models(ithread).gradfun(dslice, tslice)
  			  		val (tll, tllx) = models(ithread).eval(dslice, tslice)
  			  		if (tll < -100) {
  			  			println("iloc=%d, iend=%d, ithread=%d, tll=%f" format (iloc, iend, ithread, tll))
  			  		}
  			  		tlls(ithread) = tll
  			  		tllxs(ithread) = tllx
  			  		targetmat(?, iloc->iend) = tslice
  			  		if (regularizer != null) regularizer.compute(1.0f*(iend-iloc)/n)	
  			  	}
  			    done(ithread) = 1
//  			  	println("done %d" format ithread)
  			  }
  			}
// 			println("got here 5")
  			while (sum(done).dv < numthreads) {Thread.sleep(10)}
 			  device(curdevice)
//  			println("got here 6")
 			  var iend = 0
  			for (ithread <- 1 until numthreads) {
  				val iloc = ipos + ithread*blocksize
  				iend = math.min(n, iloc+blocksize)
  				model.updatemat ~ model.updatemat + models(ithread).updatemat
  				nmodel.updateDenom ~ nmodel.updateDenom + models(ithread).updateDenom
  			}
  			updater.update(iend-ipos)
  			llest = (1/(nw+1))*(mean(tlls).dv + nw*llest)
  			ipos += numthreads*blocksize
  			if (toc >= tsecs || (ipos >= n)) {
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
		var numGPUthreads = 1
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
