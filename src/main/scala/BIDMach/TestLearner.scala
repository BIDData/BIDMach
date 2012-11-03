package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import Learner._


object TestLearner {
  def runLogLearner(rt:SMat, st:FMat) = {
    val model = new LogisticModel(rt, st)
    val regularizer = new L1Regularizer(model)
    val updater = new ADAGradUpdater(model)
  	val learner = Learner(rt, st > 4, model, regularizer, updater)
  	learner.run
  }
  
  def runLinLearner(rt:SMat, st:FMat) = {
    val model = new LinearRegModel(rt, st) { 
      override def initmodel(data:Mat, target:Mat) = initmodelf(data, target) 
      override def regfn(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit =  linearMap1(targ, pred, lls, gradw)
      }
    val regularizer = new L1Regularizer(model)
    val updater = new ADAGradUpdater(model) { override def update(step:Int):Unit = update1(step) }
    val learner = Learner(rt, st, model, regularizer, updater)
    regularizer.options.beta = 1e-7f
    updater.options.alpha = 300f
    updater.options.gradwindow = 1e6f
  	learner.options.npasses = 50
  	learner.run
  }
  
  def main(args: Array[String]): Unit = {
    Mat.checkCUDA
    tic
    val dirname = "d:\\sentiment\\sorted_data\\books\\parts\\"
  	val revtrain:SDMat = load(dirname+"part1.mat", "revtrain")
  	val t1 = toc; tic
  	val rt = SMat(revtrain)(0->40000,0->(8000*(size(revtrain,2)/8000)))
  	val scrtrain:IMat = load(dirname+"part1.mat", "scrtrain")
  	val st = FMat(scrtrain).t
  	val t2 = toc
  	println("Reading time=%3.2f+%3.2f seconds" format (t1,t2))
  	val stt = zeros(16, size(st,2))
  	for (i<-0 until size(stt,1)) {stt(i,?) = st}
  	flip
    runLinLearner(rt, stt)
    val (ff, tt) = gflop
    println("Time=%5.3f, gflops=%3.2f" format (tt, ff))
  }
}
