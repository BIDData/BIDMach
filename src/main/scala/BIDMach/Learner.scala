package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Plotting._
import scala.collection.immutable.List

case class Learner(
    val datasource:DataSource, 
    val model:Model, 
    val regularizer:Regularizer, 
    val updater:Updater, 
		val opts:Learner.Options = new Learner.Options) {
  
  def run() = {
    flip 
    var done = false
    var ipass = 0
    var here = 0L
    while (ipass < opts.npasses && ! done) {
      datasource.reset
      updater.clear
      var istep = 0
      print("i=%2d" format ipass)
      while (datasource.hasNext) {
        val mats = datasource.next
        here += datasource.opts.blockSize
        if (datasource.hasNext) {
        	model.doblockg(mats, here)
        	if (regularizer != null) regularizer.compute(here)
        	updater.update(here)
        	print(".")
        } else {
          val scores = model.evalblockg(mats)
          print("ll="); scores.data.foreach(v => print(" %4.3f" format v)); println(" mem=%f" format GPUmem._1)
        }   
        if (model.opts.putBack >= 0) datasource.putBack(mats, model.opts.putBack)
        istep += 1
      }
      updater.updateM
      ipass += 1
    }
    val gf = gflop
    println("Time=%5.4f secs, gflops=%4.2f" format (gf._2, gf._1))
  }
}

/*
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
  			val nmodel = model.asInstanceOf[FactorModel]
 // 	    val (v0,x0) = model.eval(dslice, tslice)
  			model.gradfun(dslice, tslice)
  			val (tll, tllx) = model.eval(dslice, tslice)
  			targetmat(?, i->iend) = tslice
  			if (regularizer != null) regularizer.compute(1.0f*(iend-i)/n)
//  			val (v1,x1) = model.eval(dslice, tslice) 			
  			updater.update(iend-i)
//  			val (v2,x2) = model.eval(dslice, tslice)
//  			println("i=%d, delta = %f, %f, %f, %f" format (i, tll, v0, v1, v2))

  			llest = (1/(nw+1))*(tll + nw*llest)
  			llder = (1/(nww+1))*(tll-llold + nww*llder)
  			llold = tll
  			i += blocksize
  			if (llder > 0 && llder < options.convslope) {
  				done = true
  			}
  			if (toc >= tsecs || done || (i >= n)) {
  			  val tmp = model.asInstanceOf[FactorModel].options.uiter
  			  model.asInstanceOf[FactorModel].options.uiter = 10
  			  model.gradfun(datatest0, targettest)
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
  			    if (ithread == curdevice) models(ithread).modelmat <-- model.modelmat
  			  	val iloc = ipos + ithread*blocksize
  			  	var iend = math.min(n, iloc+blocksize)
  			  	if (iloc < iend) {
  			  		var dslice = datamat0(?, iloc->iend)
  			  		var tslice = targetmat(?, iloc->iend)
  			  		models(ithread).gradfun(dslice, tslice)
  			  		val (tll, tllx) = models(ithread).eval(dslice, tslice)
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
  			for (ithread <- 1 until numthreads) {
  				model.updatemat ~ model.updatemat + models(ithread).updatemat
  				nmodel.updateDenom ~ nmodel.updateDenom + models(ithread).updateDenom
  			}
 			  val iend = math.min(n, ipos+4*blocksize)
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
} */


object Learner {
	class Options {
		var blockSize:Int = 100000
		var npasses:Int = 20
	  var memwindow:Double = 40000
		var convwindow:Double = 40000	
		var convslope:Double = -1e-6
		var secprint:Double = 1
		var eps:Float = 1e-8f
		var useGPU = false
		var numGPUthreads = 1
  }
}

class TestLDA {
  var fname:String = "/big/twitter/test/smat_20_100_400.lz4"
  var dd:MatDataSource = null
  var model:LDAModel = null
  var updater:Updater = null
  var lda:Learner = null
  var lopts = new Learner.Options
  var mopts = new LDAModel.Options
  var dopts = new MatDataSource.Options
  def setup = { 
    val aa = if (mopts.putBack >= 0) new Array[Mat](2) else new Array[Mat](1)
    aa(0) = HMat.loadSMat(fname)
    dd = new MatDataSource(aa, dopts)
    model = new LDAModel(mopts)
    if (mopts.putBack >= 0) aa(1) = ones(model.opts.dim, aa(0).ncols)
    updater = new IncNormUpdater
    dd.init
    model.init(dd)
    updater.init(model)
    lda = new Learner(dd, model, null, updater, lopts)   
  }
  
  def init = {
    if (dd.mats.length > 1) dd.mats(1) = ones(model.opts.dim, dd.mats(0).ncols)
    dd.init
    model.init(dd)
    updater.init(model)
  }
  
  def run = lda.run
}