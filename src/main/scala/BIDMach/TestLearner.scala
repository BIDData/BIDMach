package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Plotting._


object TestLearner {
/*  
  def runLDALearner(rt:SMat, rtest:SMat, ndims:Int, nthreads:Int, useGPU:Boolean):Learner = {
    
//    Mat.numThreads = 1
    val model = new LDAmodel()
    model.options.dim = ndims
    model.options.uiter = 4
    model.options.uprior = 1e-1f
    model.options.mprior = 1e0f
    model.options.minuser = 1e-7f
    model.options.nzPerColumn = 400
    model.options.useGPU = useGPU

    val updater = new MultUpdater
    updater.options.alpha = 0.3f
//    val updater = new MultUpdater(model)
//    updater.options.alpha = 0.1f
    updater.options.initnsteps = 8000f
    
  	val learner = Learner(rt, null, rtest, null, model, null, updater)
  	learner.options.npasses = 20
  	learner.options.secprint = 100
  	learner.options.blocksize = 8000 //size(rt,2)
  	learner.options.numGPUthreads = nthreads
  	learner.run
  	learner
  }
  
  def runNMFLearner(rt:SMat, rtest:SMat, ndims:Int, nthreads:Int, useGPU:Boolean):Learner = {
    val model = new NMFmodel()
    model.options.dim = ndims
    model.options.uiter = 4
    model.options.uprior = 1e-4f
    model.options.mprior = 1e2f
    model.options.minuser = 1e-8f
    model.options.nzPerColumn = 400
    model.options.useGPU = useGPU

    val updater = new MultUpdater
    updater.options.alpha = 0.1f
//    val updater = new MultUpdater(model)
//    updater.options.alpha = 0.1f
    updater.options.initnsteps = 8000f
    
  	val learner = Learner(rt, null, rtest, null, model, null, updater)
  	learner.options.npasses = 10
  	learner.options.secprint = 100
  	learner.options.blocksize = 16000/nthreads //size(rt,2)//40000 //
  	learner.options.numGPUthreads = nthreads
  	learner.run
  	learner
  }
  
  def runLogLearner(rt:SMat, st:FMat, rtest:SMat, stest:FMat):Learner = {
    val model = new LogisticModel()
    model.options.useGPU = false
    
    val regularizer = new L1Regularizer(model)
   	regularizer.options.mprior = 1e-7f
   	
    val updater = new ADAGradUpdater
    updater.options.alpha = 300f
    updater.options.gradwindow = 1e6f
    
  	val learner = Learner(rt, st > 4, rtest, stest > 4, model, regularizer, updater)
  	learner.options.npasses = 20
  	learner.run
  	learner
  }
  
  def runLinLearner(rt:SMat, st:FMat, rtest:SMat, stest:FMat):Learner = {
    val model = new LinearRegModel() { 
      override def regfn(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit =  linearMap1(targ, pred, lls, gradw)
    }
    model.options.nzPerColumn = 400
    model.options.transpose = false
    model.options.useGPU = false
    
    val regularizer = new L1Regularizer(model)
    regularizer.options.mprior = 1e-6f
    
    val updater = new ADAGradUpdater { override def update(step:Int):Unit = update1(step) }
 //    regularizer.options.beta = 1e-7f
    updater.options.alpha = 200f
    updater.options.gradwindow = 1e6f
    
    val learner = Learner(rt, st, rtest, stest, model, regularizer, updater)
  	learner.options.npasses = 10
  	learner.options.secprint = 100
  	learner.run
  	learner
  }
  
  def runtest(dirname:String, ntest:Int, ndims:Int, nthreads:Int, useGPU:Boolean):Learner = {
    tic
  	val revtrain:SMat = load(dirname+"xpart1.mat", "revtrain")
  	val revtest:SMat = load(dirname+"xpart1.mat", "revtest")
  	val t1 = toc; tic
  	val rt = revtrain(0->4000,0->(8000*(size(revtrain,2)/8000)))
  	val rtest = revtest(0->4000,0->(8000*(size(revtest,2)/8000)))
  	val scrtrain:IMat = load(dirname+"xpart1.mat", "scrtrain")
  	val scrtest:IMat = load(dirname+"xpart1.mat", "scrtest")
  	val st = FMat(scrtrain).t
  	val stest = (FMat(scrtest).t)(?,0->(8000*(size(revtest,2)/8000)))
  	val t2 = toc
  	println("Reading time=%3.2f+%3.2f seconds" format (t1,t2))
  	val ntargs = ndims
  	val stt = zeros(ntargs, size(st,2))
  	val sttest = zeros(ntargs, size(stest,2))
  	for (i<-0 until size(stt,1)) {stt(i,?) = st; sttest(i,?) = stest}
  	flip
  	val learner:Learner = ntest match {
  	  case 1 => runLinLearner(rt, stt, rtest, sttest)
  	  case 2 => runLogLearner(rt, stt, rtest, sttest)
  	  case 3 => runNMFLearner(rt , rtest, ndims, nthreads, useGPU)
  	  case 4 => runLDALearner(rt , rtest, ndims, nthreads, useGPU)
  	}	
    val (ff, tt) = gflop
    println("Time=%5.3f, gflops=%3.2f" format (tt, ff))
    val xvals = irow(1->(learner.tscores.size+1))
    val tscores = learner.tscores
    val tscorex = learner.tscorex
    val tsteps = learner.tsteps
    val timeplot = semilogy(xvals, drow(tscores), xvals, drow(tscorex))
    val stepplot = semilogy(drow(tsteps), drow(learner.tscores), drow(tsteps), drow(tscorex))
//    val userhist = hist(log10(FMat(targetmat)(?)),100)
    timeplot.setTitle("Neg. log likelihood vs time in seconds")
    stepplot.setTitle("Neg. log likelihood vs number of samples")
    val modelhist = hist(log10(FMat(learner.model.modelmat)(?)),100)
//    val userhist = hist(log10(FMat(learner.targetmat)(?)),100)
    learner
  }
 
 
  def main(args: Array[String]): Unit = {
    val dirname = args(0)
    val ntest = args(1).toInt
    val ndims = args(2).toInt
    val nthreads = args(3).toInt
    val useGPU = args(4).toBoolean
    
    Mat.checkCUDA
    runtest(dirname, ntest, ndims, nthreads, useGPU)
  } */
} 
