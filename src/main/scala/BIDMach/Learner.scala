package BIDMach
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Plotting._
import BIDMat.about
import BIDMach.models._
import BIDMach.updaters._
import BIDMach.datasources._
import BIDMach.mixins._
import scala.collection.immutable.List
import scala.collection.mutable.ListBuffer
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

case class Learner(
    val datasource:DataSource,
    val model:Model, 
    val mixins:Array[Mixin], 
    val updater:Updater, 
    val opts:Learner.Options = new Learner.Options) {
  
  var results:FMat = null
  val dopts:DataSource.Opts = datasource.opts
  val mopts:Model.Opts	= model.opts
  val ropts:Mixin.Opts = if (mixins != null) mixins(0).opts else null
  val uopts:Updater.Opts = if (updater != null) updater.opts else null
  var useGPU = false
	
  def setup = {
	Learner.setupPB(datasource, dopts.putBack, mopts.dim)   
  }
  
  def init = {
    var cacheState = Mat.useCache
    Mat.useCache = opts.useCache
    datasource.init
    model.bind(datasource)
    model.init
    if (mixins != null) mixins map (_ init(model))
    if (updater != null) updater.init(model)
    Mat.useCache = cacheState;
    useGPU = model.useGPU
  }
    
  def train = {
    setup
    init
    retrain
  }
   
  def retrain() = {
    flip 
    var cacheState = Mat.useCache
    Mat.useCache = opts.useCache
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
      println("pass=%2d" format ipass)
      while (datasource.hasNext) {
        val mats = datasource.next   
        here += datasource.opts.batchSize
        bytes += 12L*mats(0).nnz
        if ((istep - 1) % opts.evalStep == 0 || (istep > 0 && (! datasource.hasNext))) {
          if (opts.updateAll) {
          	model.doblockg(mats, ipass, here);
          	if (mixins != null) mixins map (_ compute(mats, here));
          	updater.update(ipass, here);
          }
        	val scores = model.evalblockg(mats, ipass, here)
        	reslist.append(scores.newcopy)
        	samplist.append(here)
        } else {
        	model.doblockg(mats, ipass, here)
        	if (mixins != null) mixins map (_ compute(mats, here))
        	updater.update(ipass, here)
        }  
        if (datasource.opts.putBack >= 0) datasource.putBack(mats, datasource.opts.putBack)
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
        			if (useGPU) {
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
    Mat.useCache = cacheState
    println("Time=%5.4f secs, gflops=%4.2f" format (gf._2, gf._1))
    if (opts.autoReset) {
      Learner.toCPU(modelmats)
      resetGPUs
    }
    datasource.close
    results = Learner.scores2FMat(reslist) on row(samplist.toList)
  }
  
  def predict() = {
    setup
    datasource.init
    model.bind(datasource)
    val rstate = model.refresh 
    model.refresh = false
    model.init
    val results = repredict
    model.refresh = rstate
    results
  }
  
  def repredict() = {
    flip 
    useGPU = model.useGPU
    var cacheState = Mat.useCache
    Mat.useCache = opts.useCache
    var here = 0L
    var lasti = 0
    var bytes = 0L
    var lastp = 0f
    val reslist = new ListBuffer[FMat]
    val samplist = new ListBuffer[Float]
    println("Predicting")
    datasource.reset
    while (datasource.hasNext) {
      val mats = datasource.next    
      here += datasource.opts.batchSize
      bytes += 12L*mats(0).nnz
      val scores = model.evalblockg(mats, 0, here)
      reslist.append(scores.newcopy)
      samplist.append(here)
      if (dopts.putBack >= 0) datasource.putBack(mats, dopts.putBack)
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
            if (useGPU) {
              print(", GPUmem=%3.2f" format GPUmem._1) 
            }
        println
        lasti = reslist.length
      }
    }
    val gf = gflop
    Mat.useCache = cacheState
    println("Time=%5.4f secs, gflops=%4.2f" format (gf._2, gf._1))
    datasource.close
    results = Learner.scores2FMat(reslist) on row(samplist.toList)
  }
  
  def datamats = datasource.asInstanceOf[MatDS].mats
  def modelmats = model.modelmats
  def datamat = datasource.asInstanceOf[MatDS].mats(0)
  def modelmat = model.modelmats(0)
}

case class ParLearnerx(
    val datasources:Array[DataSource], 
    val models:Array[Model], 
    val mixins:Array[Array[Mixin]], 
    val updaters:Array[Updater], 
		val opts:ParLearner.Options = new ParLearner.Options) {
  
  var um:Array[Mat] = null
  var mm:Array[Mat] = null
  var results:FMat = null
  var useGPU = false
  
  def setup = {
	  for (i <- 0 until opts.nthreads) {
	  	Learner.setupPB(datasources(i), datasources(i).opts.putBack, models(i).opts.dim)
	  }   
  }
  
  def init = {
    val thisGPU = if (Mat.hasCUDA > 0) getGPU else 0
  	for (i <- 0 until opts.nthreads) {
  		if (i < Mat.hasCUDA) setGPU(i)
  		datasources(i).init
  		models(i).bind(datasources(i))
  		models(i).init
  		if (mixins != null) mixins(i) map(_ init(models(i)))
  		updaters(i).init(models(i))
  	}
  	useGPU = models(0).useGPU
  	if (Mat.hasCUDA > 0) setGPU(thisGPU)
  	val mml = models(0).modelmats.length
    um = new Array[Mat](mml)
    mm = new Array[Mat](mml)
    for (i <- 0 until mml) {
    	val mm0 = models(0).modelmats(i)
    	mm(i) = mm0.zeros(mm0.nrows, mm0.ncols)
    	um(i) = mm0.zeros(mm0.nrows, mm0.ncols)
    }
  }
  
  def train = {
    setup
    init
    retrain
  }
  
  def retrain() = {
	  flip 
	  var cacheState = Mat.useCache
    Mat.useCache = opts.useCache
	  val thisGPU = if (useGPU) getGPU else 0
	  if (useGPU) {
	    for (i <- 0 until opts.nthreads) {
	      if (i != thisGPU) connect(i)
	    }
	  }

	  @volatile var done = izeros(opts.nthreads, 1)
	  var ipass = 0
	  var istep0 = 0L
	  var ilast0 = 0L	
	  var bytes = 0L
	  val reslist = new ListBuffer[FMat]
	  val samplist = new ListBuffer[Float]    	  	
	  var lastp = 0f
	  var lasti = 0
	  done.clear
	  for (ithread <- 0 until opts.nthreads) {
	  	Future {
	  		if (useGPU && ithread < Mat.hasCUDA) setGPU(ithread)
	  		var here = 0L
	  		updaters(ithread).clear
	  		while (done(ithread) < opts.npasses) {
	  			var istep = 0
	  			while (datasources(ithread).hasNext) {
	  				val mats = datasources(ithread).next
	  				here += datasources(ithread).opts.batchSize
	  				for (j <- 0 until mats.length) bytes += 12L * mats(j).nnz
	  				models(0).synchronized {
	  					istep += 1
	  					istep0 += 1
	  				}
	  				try {
	  					if (istep % opts.evalStep == 0) {
	  						val scores = models(ithread).synchronized {models(ithread).evalblockg(mats, ipass, here)}
	  						reslist.synchronized { reslist.append(scores) }
	  						samplist.synchronized { samplist.append(here) }
	  					} else {
	  						models(ithread).synchronized {
	  							models(ithread).doblockg(mats, ipass, here)
	  							if (mixins != null && mixins(ithread) != null) mixins(ithread) map (_ compute(mats, here))
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
	  				if (useGPU) Thread.sleep(opts.coolit)
	  				if (datasources(ithread).opts.putBack >= 0) datasources(ithread).putBack(mats, datasources(ithread).opts.putBack)
//	  				if (istep % (opts.syncStep/opts.nthreads) == 0) syncmodel(models, ithread)
	  			}
	  			models(ithread).synchronized { updaters(ithread).updateM(ipass) }
	  			done(ithread) += 1
	  			while (done(ithread) > ipass) Thread.sleep(1)
	  		}
	  	}
	  }
	  println("pass=%2d" format ipass) 
	  while (ipass < opts.npasses) {
	  	while (mini(done).v == ipass) {
	  		if (istep0 >= ilast0 + opts.syncStep) {
	  			ParLearner.syncmodels(models, mm, um, useGPU)
	  			ilast0 += opts.syncStep
	  		}
	  		if (dsProgress > lastp + opts.pstep) {
	  			while (dsProgress > lastp + opts.pstep) lastp += opts.pstep
	  			val gf = gflop
	  			if (reslist.length > lasti) {
	  				print("%5.2f%%, %s, gf=%5.3f, secs=%3.1f, GB=%4.2f, MB/s=%5.2f" format (
	  						100f*lastp, 
	  						reslist.synchronized {
	  							Learner.scoreSummary(reslist, lasti, reslist.length)
	  						},
	  						gf._1,
	  						gf._2, 
	  						bytes*1e-9,
	  						bytes/gf._2*1e-6))  
	  						if (useGPU) {
	  							for (i <- 0 until math.min(opts.nthreads, Mat.hasCUDA)) {
	  								setGPU(i)
	  								if (i==0) print(", GPUmem=%3.2f" format GPUmem._1) else print(", %3.2f" format GPUmem._1)
	  							}
	  							setGPU(thisGPU)
	  						}
	  				println
	  			}
	  			lasti = reslist.length
	  		} else {
	  		  Thread.sleep(1)
	  		}
	  	}
	  	lastp = 0f
	  	if (ipass < opts.npasses) {
	  	  for (i <- 0 until opts.nthreads) datasources(i).reset
	  	  println("pass=%2d" format ipass+1) 
	  	}
	  	if (opts.resFile != null) {
      	saveAs(opts.resFile, Learner.scores2FMat(reslist) on row(samplist.toList), "results")
      }
	  	ipass += 1
	  }
	  val gf = gflop
	  Mat.useCache = cacheState
	  println("Time=%5.4f secs, gflops=%4.2f, MB/s=%5.2f, GB=%5.2f" format (gf._2, gf._1, bytes/gf._2*1e-6, bytes*1e-9))
	  if (opts.autoReset) {
	    Learner.toCPU(modelmats)
	    resetGPUs
	  }
	  for (ithread <- 0 until opts.nthreads) datasources(ithread).close
	  results = Learner.scores2FMat(reslist) on row(samplist.toList)
  }
  
  def syncmodel(models:Array[Model], ithread:Int) = {
	  mm.synchronized {
	    for (i <- 0 until models(ithread).modelmats.length) {
	    	um(i) <-- models(ithread).modelmats(i)
	    	um(i) ~ um(i) *@ (1f/opts.nthreads)
	    	mm(i) ~ mm(i) *@ (1 - 1f/opts.nthreads)
	    	mm(i) ~ mm(i) + um(i)
	    	models(ithread).modelmats(i) <-- mm(i)
	    }
	  }
  }
  
  def restart(ithread:Int) = {
    if (useGPU) {
      resetGPU
      Mat.trimCache2(ithread)
    }
    models(ithread).bind(datasources(ithread))
    models(ithread).init
    for (i <- 0 until models(ithread).modelmats.length) {
    	models(ithread).modelmats(i) <-- mm(i)
    }
    updaters(ithread).init(models(ithread))      
  }
  
  def dsProgress:Float = {
    var sum = 0f
    for (i <- 0 until datasources.length) {
      sum += datasources(i).progress
    }
    sum / datasources.length
  }
  
  def modelmats = models(0).modelmats
  def modelmat = models(0).modelmats(0)

}

class ParLearnerxF(
    dopts:DataSource.Opts,
		ddfun:(DataSource.Opts, Int)=>DataSource,
		mopts:Model.Opts,
		mkmodel:(Model.Opts)=>Model,
		ropts:Mixin.Opts,
		mkreg:(Mixin.Opts)=>Array[Mixin],
		uopts:Updater.Opts,
		mkupdater:(Updater.Opts)=>Updater,
		val lopts:ParLearner.Options = new ParLearner.Options) {

  var dds:Array[DataSource] = null
  var models:Array[Model] = null
  var mixins:Array[Array[Mixin]] = null
  var updaters:Array[Updater] = null
  var learner:ParLearnerx = null
  
  def setup = {
    dds = new Array[DataSource](lopts.nthreads)
    models = new Array[Model](lopts.nthreads)
    if (mkreg != null) mixins = new Array[Array[Mixin]](lopts.nthreads)
    updaters = new Array[Updater](lopts.nthreads)
    val thisGPU = if (Mat.hasCUDA > 0) getGPU else 0
    for (i <- 0 until lopts.nthreads) {
      if (mopts.useGPU && i < Mat.hasCUDA) setGPU(i)
    	dds(i) = ddfun(dopts, i)
    	models(i) = mkmodel(mopts)
    	if (mkreg != null) mixins(i) = mkreg(ropts)
    	updaters(i) = mkupdater(uopts)
    }
    if (0 < Mat.hasCUDA) setGPU(thisGPU)
    learner = new ParLearnerx(dds, models, mixins, updaters, lopts)
    learner.setup
  }
  
  def init = learner.init
  
  def train = {
    setup
    init
    learner.retrain
  }
}

case class ParLearner(
    val datasource:DataSource, 
    val models:Array[Model], 
    val mixins:Array[Array[Mixin]], 
    val updaters:Array[Updater], 
	val opts:ParLearner.Options = new ParLearner.Options) {
  
  var um:Array[Mat] = null
  var mm:Array[Mat] = null
  var results:FMat = null
  var cmats:Array[Array[Mat]] = null
  var useGPU = false
  
  def setup = {
	  val dopts	= datasource.opts
	  Learner.setupPB(datasource, datasource.opts.putBack, models(0).opts.dim)  
  }
  
  def init = {
    datasource.init
    useGPU = models(0).opts.useGPU
    val thisGPU = if (useGPU) getGPU else 0
    for (i <- 0 until opts.nthreads) {
      if (useGPU && i < Mat.hasCUDA) setGPU(i)
    	models(i).bind(datasource)
    	models(i).init
    	if (mixins != null) mixins(i) map (_ init(models(i)))
    	updaters(i).init(models(i))
    }
    if (useGPU) setGPU(thisGPU) 
    val mml = models(0).modelmats.length
    um = new Array[Mat](mml)
    mm = new Array[Mat](mml)
    for (i <- 0 until mml) {
    	val mm0 = models(0).modelmats(i)
    	mm(i) = mm0.zeros(mm0.nrows, mm0.ncols)
    	um(i) = mm0.zeros(mm0.nrows, mm0.ncols)
    }
  }
  
  def train = {
    setup 
    init
    retrain
  }
  
  def retrain = {
    flip
    val mm0 = models(0).modelmats(0)
    var cacheState = Mat.useCache
    Mat.useCache = opts.useCache
    cmats = new Array[Array[Mat]](opts.nthreads)
    for (i <- 0 until opts.nthreads) cmats(i) = new Array[Mat](datasource.omats.length)
    val thisGPU = if (useGPU) getGPU else 0
	  if (useGPU) {
	    for (i <- 0 until opts.nthreads) {
	      if (i != thisGPU) connect(i)
	    }
	  }    
    @volatile var done = iones(opts.nthreads, 1)
    var ipass = 0
    var here = 0L
    var feats = 0L
    var lasti = 0
    var bytes = 0L
    val reslist = new ListBuffer[FMat]
    val samplist = new ListBuffer[Float]
    for (i <- 0 until opts.nthreads) {
    	if (useGPU && i < Mat.hasCUDA) setGPU(i)
    	updaters(i).clear
    }
    setGPU(thisGPU)
    var istep = 0
    var lastp = 0f
    var running = true

    for (ithread <- 0 until opts.nthreads) {
    	Future {
    		if (useGPU && ithread < Mat.hasCUDA) setGPU(ithread)
    		while (running) {
    			while (done(ithread) == 1) Thread.sleep(1)
    			try {
    				if ((istep + ithread + 1) % opts.evalStep == 0 || !datasource.hasNext ) {
    					val scores = models(ithread).evalblockg(cmats(ithread), ipass, here)
    					reslist.synchronized { reslist.append(scores(0)) }
    					samplist.synchronized { samplist.append(here) }
    				} else {
    					models(ithread).doblockg(cmats(ithread), ipass, here)
    					if (mixins != null && mixins(ithread) != null) mixins(ithread) map (_ compute(cmats(ithread), here))
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
    while (ipass < opts.npasses) {
    	datasource.reset
      istep = 0
      lastp = 0f
      println("pass=%2d" format ipass)
    	while (datasource.hasNext) {
    		for (ithread <- 0 until opts.nthreads) {
    			if (datasource.hasNext) {
    				val mats = datasource.next
    				for (j <- 0 until mats.length) {
    				  cmats(ithread)(j) = safeCopy(mats(j), ithread) 
    				}
    				here += datasource.opts.batchSize
    				feats += mats(0).nnz
    				done(ithread) = 0;
    				bytes += 12L*mats(0).nnz
    			} 
    		}
      	while (mini(done).v == 0) Thread.sleep(1)
      	Thread.sleep(opts.coolit)
      	istep += opts.nthreads
      	if (istep % opts.syncStep == 0) ParLearner.syncmodels(models, mm, um, useGPU)
      	if (datasource.progress > lastp + opts.pstep) {
      		while (datasource.progress > lastp + opts.pstep) lastp += opts.pstep
      		val gf = gflop
      		if (reslist.length > lasti) {
      			print("%5.2f%%, %s, gf=%5.3f, secs=%3.1f, GB=%4.2f, MB/s=%5.2f" format (
      					100f*lastp, 
      					Learner.scoreSummary(reslist, lasti, reslist.length),
      					gf._1,
      					gf._2, 
      					bytes*1e-9,
      					bytes/gf._2*1e-6))  
      		  if (useGPU) {
      		    for (i <- 0 until math.min(opts.nthreads, Mat.hasCUDA)) {
      		      setGPU(i)
      		      if (i==0) print(", GPUmem=%3.2f" format GPUmem._1) else print(", %3.2f" format GPUmem._1)
      		    }
      		    setGPU(thisGPU)
      		  }
      			println
      		}
      		lasti = reslist.length
      	}
      }
      for (i <- 0 until opts.nthreads) {
        if (useGPU && i < Mat.hasCUDA) setGPU(i); 
        updaters(i).updateM(ipass)
      }
      setGPU(thisGPU)
      ipass += 1
      if (opts.resFile != null) {
      	saveAs(opts.resFile, Learner.scores2FMat(reslist) on row(samplist.toList), "results")
      }
    }
    running = false;
    datasource.close
    val gf = gflop
    Mat.useCache = cacheState
    println("Time=%5.4f secs, gflops=%4.2f, samples=%4.2g, MB/sec=%4.2g" format (gf._2, gf._1, 1.0*here, bytes/gf._2/1e6))
    results = Learner.scores2FMat(reslist) on row(samplist.toList)
  }
  
  def safeCopy(m:Mat, ithread:Int):Mat = {
    m match {
      case ss:SMat => {
        val out = SMat.newOrCheckSMat(ss.nrows, ss.ncols, ss.nnz, null, m.GUID, ithread, "safeCopy".##)
        ss.copyTo(out)
      }
      case ss:FMat => {
        val out = FMat.newOrCheckFMat(ss.nrows, ss.ncols, null, m.GUID, ithread, "safeCopy".##)
        ss.copyTo(out)
      }
    }
  }
  
  def restart(ithread:Int) = {
    if (useGPU) {
      resetGPU
      Mat.trimCaches(ithread)
    }
    models(ithread).bind(datasource)
    models(ithread).init
    models(ithread).modelmats(0) <-- mm(0)
    updaters(ithread).init(models(ithread))      
  }
  
  def datamats = datasource.asInstanceOf[MatDS].mats
  def modelmats = models(0).modelmats
  def datamat = datasource.asInstanceOf[MatDS].mats(0)
  def modelmat = models(0).modelmats(0)
}

class ParLearnerF(
		val ds:DataSource,
		val mopts:Model.Opts,
		mkmodel:(Model.Opts)=>Model,
		ropts:Mixin.Opts,
		mkreg:(Mixin.Opts)=>Array[Mixin],
		val uopts:Updater.Opts,
		mkupdater:(Updater.Opts)=>Updater,
		val lopts:ParLearner.Options = new ParLearner.Options) {
  var models:Array[Model] = null
  var mixins:Array[Array[Mixin]] = null
  var updaters:Array[Updater] = null
  var learner:ParLearner = null
  
  def setup = {
    models = new Array[Model](lopts.nthreads)
    if (mkreg != null) mixins = new Array[Array[Mixin]](lopts.nthreads)
    updaters = new Array[Updater](lopts.nthreads) 
    val thisGPU = if (Mat.hasCUDA > 0) getGPU else 0
    for (i <- 0 until lopts.nthreads) {
      if (mopts.useGPU && i < Mat.hasCUDA) setGPU(i)
    	models(i) = mkmodel(mopts)
    	if (mkreg != null) mixins(i) = mkreg(ropts)
    	updaters(i) = mkupdater(uopts)
    }
    if (0 < Mat.hasCUDA) setGPU(thisGPU)
    learner = new ParLearner(ds, models, mixins, updaters, lopts)   
    learner.setup
  }
  
  def init =	learner.init
  
  def train = {
    setup
    init
    retrain
  }
  
  def retrain = learner.retrain
}

object Learner {
  
  class Options extends BIDMat.Options {
  	var npasses = 2 
  	var evalStep = 11
  	var pstep = 0.01f
  	var resFile:String = null
  	var autoReset = true
  	var useCache = true
  	var updateAll = false
  }
  
    
  def toCPU(mats:Array[Mat]) {
    for (i <- 0 until mats.length) {
      mats(i) match {
        case g:GMat => mats(i) = FMat(g)
        case g:GSMat => mats(i) = SMat(g)
        case g:GIMat => mats(i) = IMat(g)
        case _ => {}
      }
    }
  }
  
  def setupPB(ds:DataSource, npb:Int, dim:Int) = {
    ds match {
    case ddm:MatDS => {
    	if (npb >= 0) {
    		ddm.setupPutBack(npb, dim)
    	}
    }
    case _ => {}
    }
  }
  
  def scoreSummary(reslist:ListBuffer[FMat], lasti:Int, length:Int):String = {
    var i = lasti
    var sum = 0.0
    while (i < length) {
      val scoremat = reslist(i)
      sum += mean(scoremat(?,0)).v
      i += 1
    }
    ("ll=%6.5f" format sum/(length-lasti))    
  }
  
  def scores2FMat(reslist:ListBuffer[FMat]):FMat = {
    val out = FMat(reslist(0).nrows, reslist.length)
    var i = 0
    while (i < reslist.length) {
      val scoremat = reslist(i)
      out(?, i) = scoremat(?,0)
      i += 1
    }
    out
  }
}

object ParLearner {
  
  class Options extends 
  Learner.Options {
  	var nthreads = math.max(0, Mat.hasCUDA)
  	var syncStep = 32
  	var coolit = 60
  }
  
  def syncmodels(models:Array[Model], mm:Array[Mat], um:Array[Mat], useGPU:Boolean) = {
	  for (j <- 0 until models(0).modelmats.length) {
	  	mm(j).clear
	  	for (i <- 0 until models.length) {
	  		um(j) <-- models(i).modelmats(j)
	  		mm(j) ~ mm(j) + um(j)
	  	}
	  	mm(j) ~ mm(j) * (1f/models.length)
	  	for (i <- 0 until models.length) {
	  		models(i).modelmats(j) <-- mm(j)
	  	}
	  }
  }
  
}

