package BIDMach.caffe
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,FND,GMat,GIMat,GSMat,HMat,Image,IMat,ND,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import edu.berkeley.bvlc.SGDSOLVER
import edu.berkeley.bvlc.NET
import edu.berkeley.bvlc.CAFFE

class SGDSolver (val sgd:SGDSOLVER) {
  val net = sgd.net
  
  def Solve = sgd.Solve
  
  def SolveResume(fname:String) = sgd.SolveResume(fname)
  
}

object SGDSolver {
  def apply(paramFile:String):SGDSolver = new SGDSolver(new SGDSOLVER(paramFile))
}



