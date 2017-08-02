package BIDMach
import BIDMat.{BMat,Mat,SBMat,CMat,DMat,FMat,FFilter,IMat,HMat,GDMat,GFilter,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Plotting._
import BIDMat.about
import BIDMat.MatIOtrait
import BIDMach.models._
import BIDMach.updaters._
import BIDMach.datasources._
import BIDMach.datasinks._
import BIDMach.mixins._
import scala.collection.immutable.List
import scala.collection.mutable.ListBuffer
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.logging.Level;

//import scala.concurrent.Future
//import scala.concurrent.ExecutionContext.Implicits.global

/**
 *  Basic sequential Learner class with a single datasource
 */

@SerialVersionUID(100L)
case class Learner(
    val datasource:DataSource,
    val model:Model,
    val mixins:Array[Mixin],
    val updater:Updater,
    val datasink:DataSink,
    val opts:Learner.Opts = new Learner.Options) extends Serializable {

  var myLogger = Mat.consoleLogger;
  var fut:Future[_] = null;
  var results:FMat = null
  val dopts:DataSource.Opts = if (datasource != null) datasource.opts else null
  val mopts:Model.Opts	= model.opts
  val ropts:Mixin.Opts = if (mixins != null) mixins(0).opts else null
  val uopts:Updater.Opts = if (updater != null) updater.opts else null
  var useGPU = false
  var reslist:ListBuffer[FMat] = null;
  var samplist:ListBuffer[Float] = null;
  var lastCheckPoint = 0;
  @volatile var done = false;
  @volatile var paused = false;
  @volatile var pauseAt = -1L;
  @volatile var ipass = 0;
  @volatile var istep = 0;
  var here = 0L;
  var lasti = 0;
  var bytes = 0L;
  var nsamps = 0L;
  var cacheState = false;
  var cacheGPUstate = false;
  var debugMemState = false;
  var debugCPUmemState = false;

  def init = {
    cacheState = Mat.useCache;
    Mat.useCache = opts.useCache;
    cacheGPUstate = Mat.useGPUcache;
    Mat.useGPUcache = opts.useCache;
    datasource.init;
    model.bind(datasource);
    if (datasink.asInstanceOf[AnyRef] != null) {
      datasink.init;
      model.bind(datasink);
    }
    model.init;
    if (model.opts.logDataSink.asInstanceOf[AnyRef] != null)  model.opts.logDataSink.init
    if (mixins != null) mixins map (_ init(model))
    if (updater != null) updater.init(model)
    Mat.useCache = cacheState;
    Mat.useGPUcache = cacheGPUstate;
    useGPU = model.useGPU;
  }
  
  def launch(fn:()=>Unit) = {
    val nthreads = opts match {
      case mopts:FileSource.Opts => mopts.lookahead + 4;
      case _ => 4;
    }
    val tmp = myLogger;
    myLogger = Mat.getFileLogger(opts.logfile);
  	val executor = Executors.newFixedThreadPool(nthreads);
  	val runner = new Runnable{
  	  def run() = {
  	    try {
  	    	fn();
  	    } catch {
  	      case e:Throwable => myLogger.severe("Learner thread failed: %s" format Learner.printStackTrace(e));
  	    }
    	  myLogger = tmp;
    	}
  	}
  	fut = executor.submit(runner);
  	fut;
  }
  
  def launchTrain = {
    println("\nRunning training in the background.\nLogging to file %s in the current directory." format opts.logfile);
    launch(()=>this.train);
  }
  
  def train(doInit:Boolean) = {
    retrain(doInit)
  }
  
  def train:Unit = retrain(true);

  def retrain(doInit:Boolean) = {
    flip
    cacheState = Mat.useCache;
    Mat.useCache = opts.useCache;
    cacheGPUstate = Mat.useGPUcache;
    Mat.useGPUcache = opts.useCache;
    debugMemState = Mat.debugMem;
    debugCPUmemState = Mat.debugCPUmem;
    if (updater != null) updater.clear;
    reslist = new ListBuffer[FMat];
    samplist = new ListBuffer[Float];
    firstPass(null, doInit);
    updateM(ipass-1)
    while (ipass < opts.npasses && ! done) {
      nextPass(null)
      updateM(ipass-1)
    }
    wrapUp(ipass);
  }
  def retrain:Unit = retrain(true)

  def firstPass(iter:Iterator[(AnyRef, MatIOtrait)], doInit:Boolean = true):Unit = {
    if (doInit) {
      init
    }

    done = false;
    ipass = 0;
    here = 0L;
    lasti = 0;
    bytes = 0L;
    nsamps = 0L;
    if (updater != null) updater.clear;
    cacheState = Mat.useCache;
    Mat.useCache = opts.useCache;
    cacheGPUstate = Mat.useGPUcache;
    Mat.useGPUcache = opts.useGPUcache;
    reslist = new ListBuffer[FMat];
    samplist = new ListBuffer[Float];
    flip;
    nextPass(iter);
  }


  def nextPass(iter:Iterator[(AnyRef, MatIOtrait)]): Unit = {
    if (opts.debugMem && ipass > 0) {
      Mat.debugMem = true;
    }
    if (opts.debugCPUmem && ipass > 0) {
      Mat.debugCPUmem = true;
    }
    var lastp = 0f
    if (iter != null) {
      datasource.asInstanceOf[IteratorSource].opts.iter = iter;
    }
    datasource.reset
    istep = 0;
    myLogger.info("pass=%2d" format ipass)
    while (!done && datasource.hasNext) {
      val mats = datasource.next;
      nsamps += mats(0).ncols;
      here += datasource.opts.batchSize
      bytes += mats.map(Learner.numBytes _).reduce(_+_);
      val dsp = datasource.progress;
      val gprogress = (ipass + dsp)/opts.npasses;
      if ((istep - 1) % opts.evalStep == 0 || (istep > 0 && (! datasource.hasNext))) {
        if (opts.updateAll) {
          model.dobatchg(mats, ipass, here);
          if (mixins != null) mixins map (_ compute(mats, here));
          while (paused || (pauseAt > 0 && pauseAt <= istep)) Thread.sleep(1000);
          if (updater != null) updater.update(ipass, here, gprogress);
        }
        val tmpscores = model.evalbatchg(mats, ipass, here);
        val scores = if (tmpscores.ncols > 1) mean(tmpscores, 2) else tmpscores;
        if (datasink != null) datasink.put;
        reslist.append(scores.newcopy)
        samplist.append(here)
      } else {
        model.dobatchg(mats, ipass, here);
        if (mixins != null) mixins map (_ compute(mats, here));
        while (paused || (pauseAt > 0 && pauseAt <= istep)) Thread.sleep(1000);
        if (updater != null) updater.update(ipass, here, gprogress);
      }
      istep += 1
      if (dsp > lastp + opts.pstep && reslist.length > lasti) {
        val gf = gflop
        lastp = dsp - (dsp % opts.pstep)
        myLogger.info(("%5.2f%%, score=%6.5f, secs=%3.1f, samps/s=%4.1f, gf=%4.1f, MB/s=%4.1f" format (
          100f*lastp,
          Learner.scoreSummary(reslist, lasti, reslist.length, opts.cumScore),
          gf._2,
          nsamps/gf._2,
          gf._1,
          bytes/gf._2*1e-6)) + (if (useGPU) {", GPUmem=%3.6f" format GPUmem._1} else ""));
        lasti = reslist.length;
      }
      if (opts.checkPointFile != null && toc > 3600 * opts.checkPointInterval * (1 + lastCheckPoint)) {
        model.save(opts.checkPointFile format lastCheckPoint);
        lastCheckPoint += 1;
      }
    }
    ipass += 1
  }

  def updateM(ipass: Int): Unit = {
    if (updater != null) updater.updateM(ipass)
  }

  def wrapUp(ipass:Int) {
    if (opts.checkPointFile != null) model.save(opts.checkPointFile format lastCheckPoint)
    model.wrapUp(ipass);
    val gf = gflop;
    Mat.useCache = cacheState;
    Mat.useGPUcache = cacheGPUstate;
    Mat.debugMem = debugMemState;
    Mat.debugCPUmem = debugCPUmemState;
    myLogger.info("Time=%5.4f secs, gflops=%4.2f" format (gf._2, gf._1));
    if (opts.autoReset && useGPU) {
      Learner.toCPU(modelmats);
      model.clear;
      resetGPUs;
      Mat.clearCaches;
    }
    datasource.close;
    if (datasink != null) datasink.close;
    if (model.opts.logDataSink.asInstanceOf[AnyRef] != null) model.opts.logDataSink.close
    results = Learner.scores2FMat(reslist) on row(samplist.toList);

    done = true;
  }
  
  def pause = {
    paused = true;
    Thread.sleep(500);
  }
  
  def unpause = {
    paused = false;
  }
  
  def stop = {
    done = true;
  }
  
  def getResults = {
    val top = Learner.scores2FMat(reslist);
    val bottom = row(samplist.toList);
    val n = math.min(top.ncols, bottom.ncols);
    results = top(?,0->n) on bottom(?,0->n);  
    results;
  }
  
  def smoothResults(n:Int=10):FMat = {
    getResults;
    val b = zeros(n, results.ncols/n);
    b(?) = results(0,0->b.length);
    mean(b);
  }
  
  def plotResults(n:Int = 10) = {
    getResults;
    val b = zeros(n, results.ncols/n);
    b(?) = results(0,0->b.length);
    plot(mean(b))
  }

  def predict() = {
    datasource.init;
    model.bind(datasource);
    if (datasink.asInstanceOf[AnyRef] != null) {
      datasink.init;
      model.bind(datasink);
    }
    val rstate = model.refresh;
    model.refresh = false
    model.init
    val results = repredict
    model.refresh = rstate
    results
  }

  def repredict() = {
    flip
    useGPU = model.useGPU
    cacheState = Mat.useCache
    Mat.useCache = opts.useCache
    cacheGPUstate = Mat.useGPUcache;
    Mat.useGPUcache = opts.useCache;
    here = 0L;
    lasti = 0;
    bytes = 0L;
    nsamps = 0;
    var lastp = 0f;
    val reslist = new ListBuffer[FMat]
    val samplist = new ListBuffer[Float]
    myLogger.info("Predicting")
    datasource.reset
    while (datasource.hasNext) {
      val mats = datasource.next
      here += datasource.opts.batchSize
      bytes += mats.map(Learner.numBytes _).reduce(_+_);
      nsamps += mats(0).ncols;
      val tmpscores = model.evalbatchg(mats, 0, here);
      val scores = if (tmpscores.ncols > 1) mean(tmpscores,2) else tmpscores;
      if (datasink != null) datasink.put
      reslist.append(scores.newcopy);
      samplist.append(here);
      val dsp = datasource.progress;
      if (dsp > lastp + opts.pstep && reslist.length > lasti) {
        val gf = gflop
        lastp = dsp - (dsp % opts.pstep);
        myLogger.info(("%5.2f%%, score=%6.5f, secs=%3.1f, samps/s=%4.1f, gf=%4.1f, MB/s=%4.1f" format (
        		100f*lastp,
        		Learner.scoreSummary(reslist, lasti, reslist.length, opts.cumScore),
        		gf._2,
        		nsamps/gf._2,
        		gf._1,
        		bytes/gf._2*1e-6)) + (if (useGPU) {", GPUmem=%3.6f" format GPUmem._1} else ""));
        lasti = reslist.length;
      }
    }
    val gf = gflop;
    Mat.useCache = cacheState;
    Mat.useGPUcache = cacheGPUstate;
    myLogger.info("Time=%5.4f secs, gflops=%4.2f" format (gf._2, gf._1));
    if (opts.autoReset && useGPU) {
      Learner.toCPU(modelmats)
      resetGPUs
      Mat.clearCaches
    }
    datasource.close;
    if (datasink != null) datasink.close;
    results = Learner.scores2FMat(reslist) on row(samplist.toList)
    results
  }

  def datamats = datasource.asInstanceOf[MatSource].mats;
  def modelmats = model.modelmats;
  def datamat = datasource.asInstanceOf[MatSource].mats(0);
  def modelmat = model.modelmats(0);
  def preds = datasink.asInstanceOf[MatSink].mats
}


/**
 * Parallel Learner with a single datasource.
 */

case class ParLearner(
    val datasource:DataSource,
    val models0:Seq[Model],
    val mixins:Seq[Array[Mixin]],
    val updaters:Seq[Updater],
    val datasink:DataSink,
    val opts:ParLearner.Opts = new ParLearner.Options) extends Serializable {

  val models = models0.toArray;
	var myLogger = Mat.consoleLogger;
  var fut:Future[_] = null;
	var workers:Array[Future[_]] = null;
	var executor:ExecutorService = null;
  var um:Array[Mat] = null;
  var mm:Array[Mat] = null;
  var results:FMat = null;
  var cmats:Array[Array[Mat]] = null;
  var useGPU = false;
  var reslist:ListBuffer[FMat] = null;
  var samplist:ListBuffer[Float] = null;
  var lastCheckPoint = 0;
  @volatile var done = false;
  @volatile var paused = false;
  @volatile var ipass = 0;
  @volatile var istep = 0;
  @volatile var here = 0L;
  @volatile var lasti = 0;
  @volatile var bytes = 0L;
  @volatile var nsamps = 0L;
  @volatile var dones:IMat = null;
  var cacheState = false;
  var cacheGPUstate = false;
  var debugMemState = false;
  var debugCPUmemState = false;


  def init = {
    cacheState = Mat.useCache;
    Mat.useCache = opts.useCache;
    cacheGPUstate = Mat.useGPUcache;
    Mat.useGPUcache = opts.useCache;
    datasource.init
    useGPU = models(0).opts.useGPU
    val thisGPU = if (useGPU) getGPU else 0
    for (i <- 0 until opts.nthreads) {
      if (useGPU && i < Mat.hasCUDA) setGPU(i)
    	models(i).bind(datasource)
    	models(i).init
    	if (mixins != null) mixins(i) map (_ init(models(i)))
    	if (updaters != null && updaters(i) != null) updaters(i).init(models(i))
    }
    Mat.useCache = cacheState;
    Mat.useGPUcache = cacheGPUstate;
    useGPU = models(0).useGPU;
    if (executor.asInstanceOf[AnyRef] == null) executor = Executors.newFixedThreadPool(opts.nthreads+4);
    if (workers.asInstanceOf[AnyRef] == null) workers = new Array[Future[_] ](opts.nthreads)
    if (useGPU) setGPU(thisGPU)
    val mml = models(0).modelmats.length
    um = new Array[Mat](mml)
    mm = new Array[Mat](mml)
    for (i <- 0 until mml) {
    	val mm0 = models(0).modelmats(i)
    	mm(i) = zeros(mm0.dims)
    	um(i) = zeros(mm0.dims)
    }
    dones = iones(opts.nthreads)
    ParLearner.syncmodels(models, mm, um, 0, useGPU)
  }
  
  def launch(fn:()=>Unit) = {
    val nthreads = opts match {
      case mopts:FileSource.Opts => mopts.lookahead + opts.nthreads + 2;
      case _ => opts.nthreads + 2;
    }
    val tmp = myLogger;
    myLogger = Mat.getFileLogger(opts.logfile);
  	if (executor.asInstanceOf[AnyRef] == null) executor = Executors.newFixedThreadPool(nthreads);
  	val runner = new Runnable{
  	  def run() = {
  	    try {
  	    	fn();
  	    } catch {
  	      case e:Throwable => myLogger.severe("Learner thread failed: %s" format Learner.printStackTrace(e));
  	    }
    	  myLogger = tmp;
    	}
  	}
  	fut = executor.submit(runner);
  	fut;
  }
  
  def launchTrain = {
    println("\nRunning training in the background.\nLogging to file %s in the current directory." format opts.logfile);
    launch(()=>this.train);
  }

  def train = {
    init
    retrain
  }

  def retrain = {
    flip
    val mm0 = models(0).modelmats(0)
    var cacheState = Mat.useCache;
    var cacheGPUstate = Mat.useGPUcache;
    Mat.useGPUcache = opts.useGPUcache;
    Mat.useCache = opts.useCache
    cmats = new Array[Array[Mat]](opts.nthreads)
    for (i <- 0 until opts.nthreads) cmats(i) = new Array[Mat](datasource.omats.length)
    val thisGPU = if (useGPU) getGPU else 0
	  if (useGPU) {
	    for (i <- 0 until opts.nthreads) {
//	      if (i != thisGPU) connect(i)
	    }
	  }
    iones(opts.nthreads, 1)
    ipass = 0
    istep = 0
    here = 0L
    lasti = 0
    bytes = 0L
    reslist = new ListBuffer[FMat]
    samplist = new ListBuffer[Float]
    for (i <- 0 until opts.nthreads) {
    	if (useGPU && i < Mat.hasCUDA) setGPU(i)
    	if (updaters != null && updaters(i) != null) updaters(i).clear
    }
    setGPU(thisGPU)
    var lastp = 0f
    var progress = 0f;
    var gprogress = 0f;
    
    class LearnThread(val ithread:Int) extends Runnable {
      def run() = {
        if (useGPU && ithread < Mat.hasCUDA) setGPU(ithread)
    		while (!done) {
    			while (dones(ithread) == 1) Thread.sleep(1)
    			try {
    				if ((istep + ithread + 1) % opts.evalStep == 0 || !datasource.hasNext ) {
    					if (opts.updateAll) {
    						models(ithread).dobatchg(cmats(ithread), ipass, here)
    						if (mixins != null && mixins(ithread) != null) mixins(ithread) map (_ compute(cmats(ithread), here));
    						while (paused) Thread.sleep(1000);
    						if (updaters != null && updaters(ithread) != null) updaters(ithread).update(ipass, here, gprogress);
    					}
    					val scores = models(ithread).evalbatchg(cmats(ithread), ipass, here);
    					reslist.synchronized { reslist.append(scores(0)) }
    					samplist.synchronized { samplist.append(here) }
    				} else {
    					models(ithread).dobatchg(cmats(ithread), ipass, here)
    					if (mixins != null && mixins(ithread) != null) mixins(ithread) map (_ compute(cmats(ithread), here));
    					while (paused) Thread.sleep(1000);
    					if (updaters != null && updaters(ithread) != null) updaters(ithread).update(ipass, here, gprogress);
    				}
    			} catch {
    			case e:Throwable => {
    				myLogger.severe("Caught exception in thread %d %s\n" format (ithread, e.toString));
    				val se = e.getStackTrace();
    				for (i <- 0 until 8) {
    					myLogger.severe("thread %d, %s" format (ithread, se(i).toString));
    				}

    				myLogger.severe("Restarted: Keep on truckin...")
    			}
    			}
    			dones(ithread) = 1
    		}
    	}
    }
    
    for (i <- 0 until opts.nthreads) {
      workers(i) = executor.submit(new LearnThread(i));
    }

    while (ipass < opts.npasses) {
    	datasource.reset
      istep = 0
      lastp = 0f
      myLogger.info("pass=%2d" format ipass)
    	while (datasource.hasNext) {
    		for (ithread <- 0 until opts.nthreads) {
    			if (datasource.hasNext) {
    				val mats = datasource.next
            progress = datasource.progress
            gprogress = (ipass + progress)/opts.npasses
    				for (j <- 0 until mats.length) {
    				  cmats(ithread)(j) = safeCopy(mats(j), ithread)
    				}
    				if (ithread == 0) here += datasource.opts.batchSize
    				dones(ithread) = 0;
    				bytes += mats.map(Learner.numBytes _).reduce(_+_);
    			}
    		}
      	while (mini(dones).v == 0) Thread.sleep(1)
      	Thread.sleep(opts.coolit)
      	istep += opts.nthreads
      	if (istep % opts.syncStep == 0) ParLearner.syncmodels(models, mm, um, istep/opts.syncStep, useGPU, opts.elastic_weight)
      	if (datasource.progress > lastp + opts.pstep) {
      		while (datasource.progress > lastp + opts.pstep) lastp += opts.pstep
      		val gf = gflop
      		if (reslist.length > lasti) {
      			val perfStr = ("%5.2f%%, score=%6.5f, secs=%3.1f, samps/s=%4.1f, gf=%4.1f, MB/s=%4.1f" format (
      					100f*lastp,
      					Learner.scoreSummary(reslist, lasti, reslist.length, opts.cumScore),
      					gf._1,
      					gf._2,
      					bytes*1e9,
      					bytes/gf._2*1e-6));
      		  val gpuStr = if (useGPU) {
      		    (0 until math.min(opts.nthreads, Mat.hasCUDA)).map((i)=>{
      		      setGPU(i);
      		      if (i==0) (", GPUmem=%3.2f" format GPUmem._1) else (", %3.2f" format GPUmem._1)
      		    });
      		  } else "";
      		  myLogger.info(perfStr + gpuStr);
      			if (useGPU) setGPU(thisGPU);
      		}
      		lasti = reslist.length
      	}
      }
      for (i <- 0 until opts.nthreads) {
        if (useGPU && i < Mat.hasCUDA) setGPU(i);
        if (updaters != null && updaters(i) != null) updaters(i).updateM(ipass)
      }
      setGPU(thisGPU)
      ParLearner.syncmodelsPass(models, mm, um, ipass)
      ipass += 1
      if (opts.resFile != null) {
      	saveAs(opts.resFile, Learner.scores2FMat(reslist) on row(samplist.toList), "results")
      }
    }
    done = true;
    datasource.close
    val gf = gflop
    Mat.useCache = cacheState
    if (useGPU) {
    	for (i <- 0 until opts.nthreads) {
 //   		if (i != thisGPU) disconnect(i);
    	}
    }
    if (opts.autoReset && useGPU) {
      Learner.toCPU(models(0).modelmats)
      resetGPUs
    }
    val perfStr = ("%5.2f%%, score=%6.5f, secs=%3.1f, samps/s=%4.1f, gf=%4.1f, MB/s=%4.1f" format (
    		           100f*lastp,
    		           Learner.scoreSummary(reslist, lasti, reslist.length, opts.cumScore),
    		           gf._1,
    		           gf._2,
    		           bytes*1e9,
    		           bytes/gf._2*1e-6));
    val gpuStr = if (useGPU) {
    	             (0 until math.min(opts.nthreads, Mat.hasCUDA)).map((i)=>{
    	            	 setGPU(i);
    	            	 if (i==0) (", GPUmem=%3.2f" format GPUmem._1) else (", %3.2f" format GPUmem._1)
    	             });
    } else "";
    myLogger.info(perfStr + gpuStr);
    results = Learner.scores2FMat(reslist) on row(samplist.toList)
  }

  def safeCopy(m:Mat, ithread:Int):Mat = {
    m match {
      case ss:SMat => {
        val out = SMat.newOrCheckSMat(ss.nrows, ss.ncols, ss.nnz, null, m.GUID, ithread, "safeCopy".##)
        ss.copyTo(out)
      }
      case ss:SDMat => {
        val out = SDMat.newOrCheckSDMat(ss.nrows, ss.ncols, ss.nnz, null, m.GUID, ithread, "safeCopy".##)
        ss.copyTo(out)
      }
      case ss:FMat => {
        val out = FMat.newOrCheckFMat(ss.dims, null, m.GUID, ithread, "safeCopy".##)
        ss.copyTo(out)
      }
      case ss:DMat => {
        val out = DMat.newOrCheckDMat(ss.dims, null, m.GUID, ithread, "safeCopy".##)
        ss.copyTo(out)
      }
      case ss:BMat => {
        val out = BMat.newOrCheckBMat(ss.dims, null, m.GUID, ithread, "safeCopy".##)
        ss.copyTo(out)
      }
      case ss:IMat => {
        val out = IMat.newOrCheckIMat(ss.nrows, ss.ncols, null, m.GUID, ithread, "safeCopy".##)
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

  def datamats = datasource.asInstanceOf[MatSource].mats
  def modelmats = models(0).modelmats
  def datamat = datasource.asInstanceOf[MatSource].mats(0)
  def modelmat = models(0).modelmats(0)
}


/**
 * Parallel Learner class with multiple datasources, models, mixins, and updaters.
 * i.e. several independent Learners whose models are synchronized periodically.
 */

case class ParLearnerx(
    val datasources:Array[DataSource],
    val models:Array[Model],
    val mixins:Array[Array[Mixin]],
    val updaters:Array[Updater],
    val datasinks:Array[DataSink],
		val opts:ParLearner.Options = new ParLearner.Options) extends Serializable {

  var um:Array[Mat] = null
  var mm:Array[Mat] = null
  var results:FMat = null
  var useGPU = false

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
    	mm(i) = zeros(mm0.nrows, mm0.ncols)
    	um(i) = zeros(mm0.nrows, mm0.ncols)
    }
  }

  def train = {
    init
    retrain
  }

  def retrain() = {
	  flip
	  var cacheState = Mat.useCache;
    Mat.useCache = opts.useCache;
    var cacheGPUstate = Mat.useGPUcache;
    Mat.useGPUcache = opts.useCache;
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
    var gprogress = 0f
	  done.clear
	  (0 until opts.nthreads).par.foreach((ithread:Int) => {
	  		if (useGPU && ithread < Mat.hasCUDA) setGPU(ithread)
	  		var here = 0L
	  		updaters(ithread).clear
	  		while (done(ithread) < opts.npasses) {
	  			var istep = 0
	  			while (datasources(ithread).hasNext) {
	  				val mats = datasources(ithread).next
	  				here += datasources(ithread).opts.batchSize
	  				bytes += mats.map(Learner.numBytes _).reduce(_+_);
            gprogress = (dsProgress + ipass)/opts.npasses
	  				models(0).synchronized {
	  					istep += 1
	  					istep0 += 1
	  				}
	  				try {
	  					if (istep % opts.evalStep == 0) {
	  						val scores = models(ithread).synchronized {models(ithread).evalbatchg(mats, ipass, here)}
	  						reslist.synchronized { reslist.append(scores) }
	  						samplist.synchronized { samplist.append(here) }
	  					} else {
	  						models(ithread).synchronized {
	  							models(ithread).dobatchg(mats, ipass, here)
	  							if (mixins != null && mixins(ithread) != null) mixins(ithread) map (_ compute(mats, here))
	  							updaters(ithread).update(ipass, here, gprogress)
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
	  	});

	  println("pass=%2d" format ipass)
	  while (ipass < opts.npasses) {
	  	while (mini(done).v == ipass) {
	  		if (istep0 >= ilast0 + opts.syncStep) {
	  			ParLearner.syncmodels(models, mm, um, istep0/opts.syncStep, useGPU)
	  			ilast0 += opts.syncStep
	  		}
	  		if (dsProgress > lastp + opts.pstep) {
	  			while (dsProgress > lastp + opts.pstep) lastp += opts.pstep
	  			val gf = gflop
	  			if (reslist.length > lasti) {
	  				print("%5.2f%%, ll=%6.5f, gf=%5.3f, secs=%3.1f, GB=%4.2f, MB/s=%5.2f" format (
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
	  val gf = gflop;
	  Mat.useCache = cacheState;
    Mat.useGPUcache = cacheGPUstate;
	  println("Time=%5.4f secs, gflops=%4.2f, MB/s=%5.2f, GB=%5.2f" format (gf._2, gf._1, bytes/gf._2*1e-6, bytes*1e-9))
	  if (opts.autoReset && useGPU) {
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

/**
 * Parallel multi-datasource Learner that takes function arguments.
 * This allows classes to be initialized later, when the learner is setup.
 */

class ParLearnerxF(
    dopts:DataSource.Opts,
		ddfun:(DataSource.Opts, Int)=>DataSource,
		mopts:Model.Opts,
		mkmodel:(Model.Opts)=>Model,
		ropts:Mixin.Opts,
		mkreg:(Mixin.Opts)=>Array[Mixin],
		uopts:Updater.Opts,
		mkupdater:(Updater.Opts)=>Updater,
		sopts:DataSink.Opts,
		ssfun:(DataSink.Opts, Int)=>DataSink,
		val lopts:ParLearner.Options = new ParLearner.Options) extends Serializable {

  var dds:Array[DataSource] = null;
  var sss:Array[DataSink] = null
  var models:Array[Model] = null
  var mixins:Array[Array[Mixin]] = null
  var updaters:Array[Updater] = null
  var learner:ParLearnerx = null

  def setup = {
    dds = new Array[DataSource](lopts.nthreads);
    sss = new Array[DataSink](lopts.nthreads);
    models = new Array[Model](lopts.nthreads);
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
    learner = new ParLearnerx(dds, models, mixins, updaters, sss, lopts)
  }

  def init = learner.init

  def train = {
    init
    learner.retrain
  }
}


/**
 * Single-datasource parallel Learner which takes function arguments.
 */

class ParLearnerF(
		val ds:DataSource,
		val mopts:Model.Opts,
		mkmodel:(Model.Opts)=>Model,
		ropts:Mixin.Opts,
		mkreg:(Mixin.Opts)=>Array[Mixin],
		val uopts:Updater.Opts,
		mkupdater:(Updater.Opts)=>Updater,
		val sopts:DataSink.Opts,
		val ss:DataSink,
		val lopts:ParLearner.Options = new ParLearner.Options) extends Serializable {
  var models:Array[Model] = null
  var mixins:Array[Array[Mixin]] = null
  var updaters:Array[Updater] = null
  var learner:ParLearner = null

  def setup = {
    models = new Array[Model](lopts.nthreads)
    if (mkreg != null) mixins = new Array[Array[Mixin]](lopts.nthreads)
    if (mkupdater != null) updaters = new Array[Updater](lopts.nthreads)
    val thisGPU = if (Mat.hasCUDA > 0) getGPU else 0
    for (i <- 0 until lopts.nthreads) {
      if (mopts.useGPU && i < Mat.hasCUDA) setGPU(i)
    	models(i) = mkmodel(mopts)
    	if (mkreg != null) mixins(i) = mkreg(ropts)
    	if (mkupdater != null) updaters(i) = mkupdater(uopts)
    }
    if (0 < Mat.hasCUDA) setGPU(thisGPU)
    learner = new ParLearner(ds, models, mixins, updaters, ss, lopts)
  }

  def init =	learner.init

  def train = {
    init
    retrain
  }

  def retrain = learner.retrain
}

object Learner {

  trait Opts extends BIDMat.Opts {
  	var npasses = 2;
  	var evalStep = 11;
  	var pstep = 0.01f;
  	var resFile:String = null;
  	var autoReset = true;
  	var useCache = true;
  	var useGPUcache = true;
  	var updateAll = false;
  	var debugMem = false;
  	var debugCPUmem = false;
    var cumScore = 0;
    var checkPointFile:String = null;
    var checkPointInterval = 0f;
    var pauseAt = -1L;
    var logfile = "log.txt";
  }
  
  class Options extends Opts {}

  def numBytes(mat:Mat):Long = {
    mat match {
      case a:FMat => 4L * mat.length;
      case a:IMat => 4L * mat.length;
      case a:DMat => 8L * mat.length;
      case a:LMat => 8L * mat.length;
      case a:BMat => 1L * mat.length;
      case a:SMat => 8L * mat.nnz;
      case a:SDMat => 12L * mat.nnz;
    }
  }

  def toCPU(mats:Array[Mat]) {
    for (i <- 0 until mats.length) {
      mats(i) match {
        case g:GFilter => mats(i) = FFilter(g);
        case g:GMat => mats(i) = FMat(g)
        case g:GIMat => mats(i) = IMat(g)
        case g:GDMat => mats(i) = DMat(g)
        case g:GLMat => mats(i) = LMat(g)
        case g:GSMat => mats(i) = SMat(g)
        case g:GSDMat => mats(i) = SDMat(g)
        case g:TMat => mats(i) = cpu(mats(i))
        case _ => {}
      }
    }
  }

  def scoreSummary(reslist:ListBuffer[FMat], lasti:Int, len:Int, cumScore:Int = 0):Double = {
    val istart = if (cumScore == 0) lasti else {if (cumScore == 1) 0 else if (cumScore == 2) len/2 else 3*len/4};
    var i = 0
    var sum = 0.0;
    for (scoremat <- reslist) {
      if (i >= istart) sum += mean(scoremat(?,0)).v
      i += 1
    }
    sum / (len - istart)
  }

  def scores2FMat(reslist:ListBuffer[FMat]):FMat = {
    if (reslist.length == 0) return zeros(0, 0);
    val len = reslist.length;

    val out = FMat(reslist(0).nrows, len);
    var i = 0;
    while (i < len) {
      val scoremat = reslist(i)
      out(?, i) = scoremat(?,0)
      i += 1
    }
    out
  }
  
  def printStackTrace(e:Throwable):String = {
    val baos = new ByteArrayOutputStream();
    val ps = new PrintStream(baos);
    e.printStackTrace(ps);
    val str = baos.toString();
    ps.close();
    str;
  }
}

object ParLearner {

  trait Opts extends
  Learner.Opts {
  	var nthreads = math.max(0, Mat.hasCUDA)
  	var syncStep = 32
  	var elastic_weight = 1f;
  	var coolit = 60
  }
  
  class Options extends Opts{}

  def syncmodelsPass(models:Array[Model], mm:Array[Mat], um:Array[Mat], ipass:Int) = {
    models(0).mergeModelPassFn(models, mm, um, ipass);
  }

  def syncmodels(models:Array[Model], mm:Array[Mat], um:Array[Mat], istep:Long, useGPU:Boolean, elastic_weight:Float=1f) = {
    models(0).mergeModelFn(models, mm, um, istep, elastic_weight);
  }

}
