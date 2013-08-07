package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Plotting._
import scala.collection.immutable.List
import scala.actors.Actor

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
      println("i=%2d" format ipass)
      while (datasource.hasNext) {
        val mats = datasource.next
        here += datasource.opts.blockSize
        if ((istep + 1) % opts.evalStep == 0 || ! datasource.hasNext) {
        	val scores = model.evalblockg(mats)
        	print("ll="); scores.data.foreach(v => print(" %4.3f" format v)); println(" mem=%f" format GPUmem._1)
        } else {
        	model.doblockg(mats, here)
        	if (regularizer != null) regularizer.compute(here)
        	updater.update(here)
        	print(".")
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

case class ParLearner(
    val datasources:Array[DataSource], 
    val models:Array[Model], 
    val regularizers:Array[Regularizer], 
    val updaters:Array[Updater], 
		val opts:Learner.Options = new Learner.Options) {
  
  var um:FMat = null
  var mm:FMat = null
  
  def run() = {
    flip 
    val mm0 = models(0).modelmats(0)
    mm = zeros(mm0.nrows, mm0.ncols)
    um = zeros(mm0.nrows, mm0.ncols)
    
    val done = izeros(opts.nthreads, 1)
    var ipass = 0
    var here = 0L
    while (ipass < opts.npasses) {     
      for (i <- 0 until opts.nthreads) {
      	datasources(i).reset
        updaters(i).clear
      }
      var istep = 0
      println("i=%2d" format ipass)
      while (datasources(0).hasNext) {
      	here += datasources(0).opts.blockSize
        for (ithread <- 0 until opts.nthreads) {
        	done(ithread) = 1
        	Actor.actor {
        		setGPU(ithread) 
        		if (datasources(ithread).hasNext) {
        			val mats = datasources(ithread).next
        	  	if ((istep + ithread + 1) % opts.evalStep == 0 || ithread == 0 && !datasources(0).hasNext ) {
        	  		val scores = models(ithread).evalblockg(mats)
        	  		print("ll="); scores.data.foreach(v => print(" %4.3f" format v)); println(" mem=%f" format GPUmem._1)
        	  	} else {
        	  		models(ithread).doblockg(mats, here)
        	  		if (regularizers != null && regularizers(ithread) != null) regularizers(ithread).compute(here)
        	  		updaters(ithread).update(here)
        	  		print(".")
        	  	}
        			if (models(ithread).opts.putBack >= 0) datasources(ithread).putBack(mats, models(ithread).opts.putBack)
        		}
        		done(ithread) = 0   
        	}
        }
      	while (mini(done).v > 0) Thread.sleep(1)
      	syncmodels(models)
      	istep += opts.nthreads
      }
      println
      for (i <- 0 until opts.nthreads) updaters(i).updateM
      ipass += 1
    }
    val gf = gflop
    println("Time=%5.4f secs, gflops=%4.2f" format (gf._2, gf._1))
  }
     
  def syncmodels(models:Array[Model]) = {
	  mm.clear
	  for (i <- 0 until models.length) {
	  	setGPU(i)
	  	um <-- models(i).modelmats(0)
	  	mm ~ mm + um
	  }
	  mm ~ mm * (1f/models.length)
	  for (i <- 0 until models.length) {
	  	setGPU(i)
	  	models(i).modelmats(0) <-- mm
	  }
	  setGPU(0)
  }
}


object Learner {
	class Options {
		var npasses:Int = 3
		var evalStep = 10
		var useGPU = false
		var nthreads = 1
  }
}

class TestLDA(mat:Mat) {
  var dd:MatDataSource = null
  var model:LDAModel = null
  var updater:Updater = null
  var lda:Learner = null
  var lopts = new Learner.Options
  var mopts = new LDAModel.Options
  var dopts = new MatDataSource.Options
  def setup = { 
    val aa = if (mopts.putBack >= 0) {
    	val a = new Array[Mat](2); a(1) = ones(mopts.dim, mat.ncols); a
    } else {
      new Array[Mat](1)
    }
    aa(0) = mat
    dd = new MatDataSource(aa, dopts)
    dd.init
    model = new LDAModel(mopts)
    model.init(dd)
    updater = new IncNormUpdater
    updater.init(model)
    lda = new Learner(dd, model, null, updater, lopts)   
  }
  
  def init = {
    if (dd.omats.length > 1) dd.omats(1) = ones(model.opts.dim, dd.omats(0).ncols)
    dd.init
    model.init(dd)
    updater.init(model)
  }
  
  def run = lda.run
}


class TestParLDA(mat:Mat) {
  var dds:Array[DataSource] = null
  var models:Array[Model] = null
  var updaters:Array[Updater] = null
  var lda:ParLearner = null
  var lopts = new Learner.Options
  var mopts = new LDAModel.Options
  var dopts = new MatDataSource.Options
  
  def setup = {
    dds = new Array[DataSource](lopts.nthreads)
    models = new Array[Model](lopts.nthreads)
    updaters = new Array[Updater](lopts.nthreads)
    for (i <- 0 until lopts.nthreads) {
      setGPU(i)
    	val istart = i * mat.ncols / lopts.nthreads
    	val iend = (i+1) * mat.ncols / lopts.nthreads
    	val mm = mat(?, istart->iend)
    	val aa = if (mopts.putBack >= 0) {
    		val a = new Array[Mat](2); 
    		a(1) = ones(mopts.dim, mm.ncols); 
    		a
    	} else {
    		new Array[Mat](1)
    	}
    	aa(0) = mm
    	dds(i) = new MatDataSource(aa, dopts)
    	dds(i).init
    	models(i) = new LDAModel(mopts)
    	models(i).init(dds(i))
    	updaters(i) = new IncNormUpdater()
    	updaters(i).init(models(i))
    }
    setGPU(0)
    lda = new ParLearner(dds, models, null, updaters, lopts)   
  }
  
  def init = {
  	for (i <- 0 until lopts.nthreads) {
  	  setGPU(i)
  		if (dds(i).omats.length > 1) dds(i).omats(1) = ones(mopts.dim, dds(i).omats(0).ncols)
  		dds(i).init
  		models(i).init(dds(i))
  		updaters(i).init(models(i))
  	}
  	setGPU(0)
  }
  
  def run = lda.run
}