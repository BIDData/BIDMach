package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import Learner._


object TestLearner {
  def runLogLearner(rt:SMat, st:FMat, rtest:SMat, stest:FMat) = {
    val model = new LogisticModel(rt, st)
    val regularizer = new L1Regularizer(model)
    val updater = new ADAGradUpdater(model)
  	val learner = Learner(rt, st > 4, rtest, stest > 4, model, regularizer, updater)
  	regularizer.options.beta = 1e-7f
    updater.options.alpha = 300f
    updater.options.gradwindow = 1e6f
  	learner.options.npasses = 20
  	learner.run
  }
  
  def runLinLearner(rt:SMat, st:FMat, rtest:SMat, stest:FMat) = {
    val model = new LinearRegModel(rt, st) { 
      override def initmodel(data:Mat, target:Mat) = initmodelf(data, target) 
      override def regfn(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit =  linearMap1(targ, pred, lls, gradw)
      }
    val regularizer = new L1Regularizer(model)
    val updater = new ADAGradUpdater(model) { override def update(step:Int):Unit = update1(step) }
    val learner = Learner(rt, st, rtest, stest, model, regularizer, updater)
    regularizer.options.beta = 1e-6f
//  regularizer.options.beta = 1e-7f
    updater.options.alpha = 300f
    updater.options.gradwindow = 1e6f
  	learner.options.npasses = 20
  	learner.run
  }
  
  def main(args: Array[String]): Unit = {
    Mat.checkCUDA
    tic
    val dirname = "d:\\sentiment\\sorted_data\\books\\parts\\"
  	val revtrain:SDMat = load(dirname+"part1.mat", "revtrain")
  	val revtest:SDMat = load(dirname+"part1.mat", "revtest")
  	val t1 = toc; tic
  	val rt = SMat(revtrain)(?,0->(8000*(size(revtrain,2)/8000)))
  	val rtest = SMat(revtest)(?,0->(8000*(size(revtest,2)/8000)))
  	val scrtrain:IMat = load(dirname+"part1.mat", "scrtrain")
  	val scrtest:IMat = load(dirname+"part1.mat", "scrtest")
  	val st = FMat(scrtrain).t
  	val stest = (FMat(scrtest).t)(?,0->(8000*(size(revtest,2)/8000)))
  	val t2 = toc
  	println("Reading time=%3.2f+%3.2f seconds" format (t1,t2))
  	val ntargs = 16
  	val stt = zeros(ntargs, size(st,2))
  	val sttest = zeros(ntargs, size(stest,2))
  	for (i<-0 until size(stt,1)) {stt(i,?) = st; sttest(i,?) = stest}
  	flip
    runLogLearner(rt, stt, rtest, sttest)
    val (ff, tt) = gflop
    println("Time=%5.3f, gflops=%3.2f" format (tt, ff))
  }
}
