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
    val dirname = "d:\\sentiment\\sorted_data\\books\\parts\\"
  	val revtrain:SDMat = load(dirname+"part1.mat", "revtrain")
  	val rt = SMat(revtrain) 
  	val scrtrain:IMat = load(dirname+"part1.mat", "scrtrain")
  	val st = FMat(scrtrain).t
    runLinLearner(rt, st)
  }
}
