package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import Learner._


object TestLearner {
  def runLogLearner(rt:SMat, st:FMat):Learner = {
  	val learner = Learner(rt, st > 4, new LogisticModel, new ADAGradOptimizer)
  	learner.run
  	learner
  }
  
  def runLinLearner(rt:SMat, st:FMat):Learner = {
  	val learner = Learner(rt, st, new LinearRegModel, new ADAGradOptimizer)
  	learner.run
  	learner
  }
  
  def main(args: Array[String]): Unit = {
    tic
    val dirname = "d:\\sentiment\\sorted_data\\books\\parts\\"
  	val revtrain:SDMat = load(dirname+"part1.mat", "revtrain")
  	val rt = SMat(revtrain)(0->40000,0->(32*(size(revtrain,2)/32)))
  	val scrtrain:IMat = load(dirname+"part1.mat", "scrtrain")
  	val st = FMat(scrtrain).t
  	val t = toc
  	println("Reading time=%3.2f seconds" format t)
  	val stt = zeros(256, size(st,2))
  	for (i<-0 until size(stt,1)) {stt(i,?) = st}
  	flip
    runLinLearner(rt, stt)
    val (ff, tt) = gflop
    println("Time=%5.3f, gflops=%3.2f" format (tt, ff))
  }
}
