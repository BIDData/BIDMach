package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Plotting._
import BIDMat.about
import BIDMach.models._
import BIDMach.updaters._
import BIDMach.datasources._
import scala.collection.immutable.List
import scala.collection.mutable.ListBuffer
import scala.actors.Actor

case class Learner(
    val datasource:DataSource, 
    val model:Model, 
    val regularizer:Regularizer, 
    val updater:Updater, 
		val opts:Learner.Options = new Learner.Options) {
  var results:FMat = null
  val dopts:DataSource.Opts = datasource.opts
	val mopts:Model.Opts	= model.opts
	val ropts:Regularizer.Opts = if (regularizer != null) regularizer.opts else null
	val uopts:Updater.Opts = updater.opts
	
	def setup = {
    datasource match {
      case ddm:MatDataSource => {
      	if (mopts.putBack >= 0) {
      		ddm.setupPutBack(mopts.putBack+1, mopts.dim)
      	}
      }
      case _ => {}
    }
    init   
  }
  
  def run = {
    setup
    rerun
  }
  
  def init = {
    datasource.init
    model.init(datasource)
    updater.init(model)
  }
   
  def rerun() = {
    flip 
    var done = false
    var ipass = 0
    var here = 0L
    var lasti = 0
    var bytes = 0L
    updater.clear
    val reslist = new ListBuffer[FMat]
    val samplist = new ListBuffer[Float]
    while (ipass < opts.npasses && ! done) {
    	var lastp = 0f
      datasource.reset
      var istep = 0
      println("i=%2d" format ipass)
      while (datasource.hasNext) {
        val mats = datasource.next    
        here += datasource.opts.blockSize
        bytes += 12L*mats(0).nnz
        if ((istep - 1) % opts.evalStep == 0 || ! datasource.hasNext) {
        	val scores = model.evalblockg(mats, ipass)
        	reslist.append(scores.newcopy)
        	samplist.append(here)
        } else {
        	model.doblockg(mats, ipass, here)
        	if (regularizer != null) regularizer.compute(here)
        	updater.update(ipass, here)
        }   
        if (model.opts.putBack >= 0) datasource.putBack(mats, model.opts.putBack)
        istep += 1
        val dsp = datasource.progress
        if (dsp > lastp + opts.pstep && reslist.length > lasti) {
        	val gf = gflop
        	lastp = dsp - (dsp % opts.pstep)
        	print("%5.2f%%, %s, gf=%5.3f, secs=%3.1f, GB=%4.2f, MB/s=%5.2f" format (
        			100f*lastp, 
        			Learner.scoreSummary(reslist, lasti, reslist.length),
        			gf._1,
        			gf._2, 
        			bytes*1e-9,
        			bytes/gf._2*1e-6))  
        			if (model.useGPU) {
        				print(", GPUmem=%3.2f" format GPUmem._1) 
        			}
        	println
        	lasti = reslist.length
        }
      }
      updater.updateM(ipass)
      ipass += 1
    }
    val gf = gflop
    println("Time=%5.4f secs, gflops=%4.2f" format (gf._2, gf._1))
    results = Learner.scores2FMat(reslist) on row(samplist.toList)
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
  var results:FMat = null
  
  def run() = {
	  flip 
	  val mm0 = models(0).modelmats(0)
	  mm = zeros(mm0.nrows, mm0.ncols)
	  um = zeros(mm0.nrows, mm0.ncols)

	  @volatile var done = izeros(opts.nthreads, 1)
	  var ipass = 0
	  var istep0 = 0L
	  var ilast0 = 0L	
	  var bytes = 0L
	  val reslist = new ListBuffer[FMat]
	  val samplist = new ListBuffer[Float]    	  	
	  var lastp = 0f
	  done.clear
	  for (ithread <- 0 until opts.nthreads) {
	  	Actor.actor {
	  		if (ithread < Mat.hasCUDA) setGPU(ithread)
	  		var here = 0L
	  		updaters(ithread).clear
	  		while (ipass < opts.npasses) {
	  			if (ithread == 0) println("i=%2d" format ipass) 
	  			datasources(ithread).reset
	  			var istep = 0
	  			var lasti = 0
	  			while (datasources(ithread).hasNext) {
	  				val mats = datasources(ithread).next
	  				here += datasources(ithread).opts.blockSize
	  				for (j <- 0 until mats.length) bytes += 12L * mats(j).nnz
	  				istep += 1
	  				istep0 += 1
	  				try {
	  					if (istep % opts.evalStep == 0) {
	  						val scores = models(ithread).synchronized {models(ithread).evalblockg(mats, ipass)}
	  						reslist.append(scores)
	  						samplist.append(here)
	  					} else {
	  						models(ithread).synchronized {
	  							models(ithread).doblockg(mats, ipass, here)
	  							if (regularizers != null && regularizers(ithread) != null) regularizers(ithread).compute(here)
	  							updaters(ithread).update(ipass, here)
	  						}
	  					}
	  				} catch {
	  				case e:Exception => {
	  					print("Caught exception in thread %d %s\nTrying restart..." format (ithread, e.toString))
	  					restart(ithread)
	  					println("Keep on truckin...")
	  				}
	  				} 
	  				Thread.sleep(opts.coolit)
	  				if (models(ithread).opts.putBack >= 0) datasources(ithread).putBack(mats, models(ithread).opts.putBack)
//	  				if (istep % (opts.syncStep/opts.nthreads) == 0) syncmodel(models, ithread)
	  				if (ithread == 0 && datasources(0).progress > lastp + opts.pstep) {
	  					lastp += opts.pstep
	  					val gf = gflop
	  					if (reslist.length > lasti) {
	  						print("%5.2f%%, %s, gf=%5.3f, secs=%3.1f, GB=%4.2f, MB/s=%5.2f" format (
	  								100f*lastp, 
	  								Learner.scoreSummary(reslist, lasti, reslist.length),
	  								gf._1,
	  								gf._2, 
	  								bytes*1e-9,
	  								bytes/gf._2*1e-6))  
	  					  if (models(0).useGPU) {
	  					  	for (i <- 0 until math.min(opts.nthreads, Mat.hasCUDA)) {
	  					  		setGPU(i)
	  					  		if (i==0) print(", GPUmem=%3.2f" format GPUmem._1) else print(", %3.2f" format GPUmem._1)
	  					  	}
	  					  }
	  						println
	  					}
	  					lasti = reslist.length
	  				}
	  			}
	  			models(ithread).synchronized {updaters(ithread).updateM(ipass)}
	  			done(ithread) = ipass + 1
	  			while (done(ithread) > ipass) Thread.sleep(1)
	  		}
	  	}
	  }
	  while (ipass < opts.npasses) {
	  	while (mini(done).v == ipass) {
	  		while (istep0 < ilast0 + opts.syncStep) Thread.sleep(1)
	  		syncmodels(models)
	  		ilast0 += opts.syncStep
	  	}
	  	ipass += 1
	  }
	  val gf = gflop
	  println("Time=%5.4f secs, gflops=%4.2f, MB/s=%5.2f, GB=%5.2f" format (gf._2, gf._1, bytes/gf._2*1e-6, bytes*1e-9))
	  results = Learner.scores2FMat(reslist) on row(samplist.toList)
  }
     
  def syncmodels(models:Array[Model]) = {
	  for (j <- 0 until models(0).modelmats.length) {
	  	mm.clear
	  	for (i <- 0 until models.length) {
	  		if (i < Mat.hasCUDA) setGPU(i)
	  		models(i).synchronized {
	  			um <-- models(i).modelmats(j)
	  		}
	  		mm ~ mm + um
	  	}
	  	mm ~ mm *@ (1f/models.length)
	  	for (i <- 0 until models.length) {
	  		if (i < Mat.hasCUDA) setGPU(i)
	  		models(i).synchronized {
	  			models(i).modelmats(j) <-- mm
	  		}
	  	}
	  }
	  if (0 < Mat.hasCUDA) setGPU(0)
  }
  
  def syncmodel(models:Array[Model], ithread:Int) = {
	  mm.synchronized {
	  	um <-- models(ithread).modelmats(0)
	  	um ~ um *@ (1f/opts.nthreads)
	  	mm ~ mm *@ (1 - 1f/opts.nthreads)
	  	mm ~ mm + um
	  	models(ithread).modelmats(0) <-- mm
	  }
  }
  
  def restart(ithread:Int) = {
    if (models(0).useGPU) {
      resetGPU
      Mat.trimCache2(ithread)
    }
    models(ithread).init(datasources(ithread))
    models(ithread).modelmats(0) <-- mm
    updaters(ithread).init(models(ithread))      
  }
}

case class ParLearnerx(
    val datasource:DataSource, 
    val models:Array[Model], 
    val regularizers:Array[Regularizer], 
    val updaters:Array[Updater], 
		val opts:Learner.Options = new Learner.Options) {
  
  var um:FMat = null
  var mm:FMat = null
  var results:FMat = null
  var cmats:Array[Array[Mat]] = null
  
  def run() = {
    flip 
    val mm0 = models(0).modelmats(0)
    mm = zeros(mm0.nrows, mm0.ncols)
    um = zeros(mm0.nrows, mm0.ncols)
    cmats = new Array[Array[Mat]](opts.nthreads)
    for (i <- 0 until opts.nthreads) cmats(i) = new Array[Mat](datasource.omats.length)
    
    val done = iones(opts.nthreads, 1)
    var ipass = 0
    var here = 0L
    var feats = 0L
    var lasti = 0
    var bytes = 0L
    val reslist = new ListBuffer[FMat]
    val samplist = new ListBuffer[Float]
    for (i <- 0 until opts.nthreads) {
    	if (i < Mat.hasCUDA) setGPU(i)
    	updaters(i).clear
    }
    while (ipass < opts.npasses) {
    	datasource.reset
      var istep = 0
      var lastp = 0f
      println("i=%2d" format ipass)
      while (datasource.hasNext) {
        for (ithread <- 0 until opts.nthreads) {
        	if (datasource.hasNext) {
        	  done(ithread) = 0
        		val mats = datasource.next
        		here += datasource.opts.blockSize
        		feats += mats(0).nnz
        		bytes += 12L*mats(0).nnz
        		for (j <- 0 until mats.length) cmats(ithread)(j) = safeCopy(mats(j), ithread)
        		Actor.actor {
        			if (ithread < Mat.hasCUDA) setGPU(ithread)
        			try {
        				if ((istep + ithread + 1) % opts.evalStep == 0 || !datasource.hasNext ) {
        					val scores = models(ithread).evalblockg(cmats(ithread), ipass)
        					reslist.append(scores(0))
        					samplist.append(here)
        				} else {
        					models(ithread).doblockg(cmats(ithread), ipass, here)
        					if (regularizers != null && regularizers(ithread) != null) regularizers(ithread).compute(here)
        					updaters(ithread).update(ipass, here)
        				}
        			} catch {
        			  case e:Exception => {
        			    print("Caught exception in thread %d %s\nTrying restart..." format (ithread, e.toString))
        			    restart(ithread)
        			    println("Keep on truckin...")
        			  }
        			} 
        			done(ithread) = 1 
        		}  
        	}
        }
      	while (mini(done).v == 0) Thread.sleep(1)
      	Thread.sleep(opts.coolit)
      	istep += opts.nthreads
      	if (istep % opts.syncStep == 0) syncmodels(models)
      	if (datasource.progress > lastp + opts.pstep) {
      		lastp += opts.pstep
      		val gf = gflop
      		if (reslist.length > lasti) {
      			print("%5.2f%%, %s, gf=%5.3f, secs=%3.1f, GB=%4.2f, MB/s=%5.2f" format (
      					100f*lastp, 
      					Learner.scoreSummary(reslist, lasti, reslist.length),
      					gf._1,
      					gf._2, 
      					bytes*1e-9,
      					bytes/gf._2*1e-6))  
      		  if (models(0).useGPU) {
      		    for (i <- 0 until math.min(opts.nthreads, Mat.hasCUDA)) {
      		      setGPU(i)
      		      if (i==0) print(", GPUmem=%3.2f" format GPUmem._1) else print(", %3.2f" format GPUmem._1)
      		    }
      		  }
      			println
      		}
      		lasti = reslist.length
      	}
      }
      println
      for (i <- 0 until opts.nthreads) {
        if (i < Mat.hasCUDA) setGPU(i); 
        updaters(i).updateM(ipass)
      }
      ipass += 1
      saveAs("/big/twitter/test/results.mat", Learner.scores2FMat(reslist) on row(samplist.toList), "results")
    }
    val gf = gflop
    println("Time=%5.4f secs, gflops=%4.2f, samples=%4.2g, MB/sec=%4.2g" format (gf._2, gf._1, 1.0*here, bytes/gf._2/1e6))
    results = Learner.scores2FMat(reslist) on row(samplist.toList)
    if (0 < Mat.hasCUDA) setGPU(0)
  }
  
  def safeCopy(m:Mat, ithread:Int):Mat = {
    m match {
      case ss:SMat => {
        val out = SMat.newOrCheckSMat(ss.nrows, ss.ncols, ss.nnz, null, m.GUID, ithread, "safeCopy".##)
        ss.copyTo(out)
      }
    }
  }
     
  def syncmodels(models:Array[Model]) = {
	  for (j <- 0 until models(0).modelmats.length) {
	  	mm.clear
	  	for (i <- 0 until models.length) {
	  		if (i < Mat.hasCUDA) setGPU(i)
	  		um <-- models(i).modelmats(j)
	  		mm ~ mm + um
	  	}
	  	mm ~ mm *@ (1f/models.length)
	  	for (i <- 0 until models.length) {
	  		if (i < Mat.hasCUDA) setGPU(i)
	  		models(i).modelmats(j) <-- mm
	  	}
	  }
	  if (0 < Mat.hasCUDA) setGPU(0)
  }
  
  def restart(ithread:Int) = {
    if (models(0).useGPU) {
      resetGPU
      Mat.trimCaches(ithread)
    }
    models(ithread).init(datasource)
    models(ithread).modelmats(0) <-- mm
    updaters(ithread).init(models(ithread))      
  }
}


class LearnFParModel(
		val mopts:Model.Opts,
		mkmodel:(Model.Opts)=>Model,
		val uopts:Updater.Opts,
		mkupdater:(Updater.Opts)=>Updater,
		ddfun:(Int,Int)=>DataSource
		) {
  var dds:Array[DataSource] = null
  var models:Array[Model] = null
  var updaters:Array[Updater] = null
  var learner:ParLearner = null
  var lopts = new Learner.Options
  
  def setup = {
    dds = new Array[DataSource](lopts.nthreads)
    models = new Array[Model](lopts.nthreads)
    updaters = new Array[Updater](lopts.nthreads)
    for (i <- 0 until lopts.nthreads) {
      if (i < Mat.hasCUDA) setGPU(i)
    	dds(i) = ddfun(lopts.nthreads, i)
    	dds(i).init
    	models(i) = mkmodel(mopts)
    	models(i).init(dds(i))
    	updaters(i) = mkupdater(uopts)
    	updaters(i).init(models(i))
    }
    if (0 < Mat.hasCUDA) setGPU(0)
    learner = new ParLearner(dds, models, null, updaters, lopts)   
  }
  
  def init = {
  	for (i <- 0 until lopts.nthreads) {
  	  if (i < Mat.hasCUDA) setGPU(i)
  		if (dds(i).omats.length > 1) dds(i).omats(1) = ones(mopts.dim, dds(i).omats(0).ncols)
  		dds(i).init
  		models(i).init(dds(i))
  		updaters(i).init(models(i))
  	}
  	if (0 < Mat.hasCUDA) setGPU(0)
  }
  
  def run = learner.run
}


class LearnFParModelx(
		val ds:DataSource,
		val mopts:Model.Opts,
		mkmodel:(Model.Opts)=>Model,
		val uopts:Updater.Opts,
		mkupdater:(Updater.Opts)=>Updater) {
  var models:Array[Model] = null
  var updaters:Array[Updater] = null
  var learner:ParLearnerx = null
  var lopts = new Learner.Options
  
  def setup = {
    models = new Array[Model](lopts.nthreads)
    updaters = new Array[Updater](lopts.nthreads) 
    ds.init
    for (i <- 0 until lopts.nthreads) {
      if (i < Mat.hasCUDA) setGPU(i)
    	models(i) = mkmodel(mopts)
    	models(i).init(ds)
    	updaters(i) = mkupdater(uopts)
    	updaters(i).init(models(i))
    }
    if (0 < Mat.hasCUDA) setGPU(0)
    learner = new ParLearnerx(ds, models, null, updaters, lopts)   
  }
  
  def init = {
	  ds.omats(1) = ones(mopts.dim, ds.omats(0).ncols)
  	for (i <- 0 until lopts.nthreads) {
  	  if (i < Mat.hasCUDA) setGPU(i)
  		if (ds.omats.length > 1) 
  		ds.init
  		models(i).init(ds)
  		updaters(i).init(models(i))
  	}
  	if (0 < Mat.hasCUDA) setGPU(0)
  }  
  def run = learner.run
}

object Learner {
  
  class Options extends BIDMat.Options {
  	var npasses = 10 
  	var evalStep = 11
  	var syncStep = 32
  	var nthreads = 4
  	var pstep = 0.01f
  	var coolit = 60
  }
  
  def scoreSummary(reslist:ListBuffer[FMat], lasti:Int, length:Int):String = {
    var i = lasti
    var sum = 0.0
    while (i < length) {
      sum += reslist(i)(0)
      i += 1
    }
    ("ll=%5.3f" format sum/(length-lasti))    
  }
  
  def scores2FMat(reslist:ListBuffer[FMat]):FMat = {
    val out = FMat(reslist(0).length, reslist.length)
    var i = 0
    while (i < reslist.length) {
      out(?, i) = reslist(i).t
      i += 1
    }
    out
  }
}

