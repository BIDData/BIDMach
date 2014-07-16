package BIDMach.models
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,GMat,GIMat,GSMat,HMat,IMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import edu.berkeley.bvlc.SGDSOLVER
import edu.berkeley.bvlc.NET
import edu.berkeley.bvlc.CAFFE

class Net (net:NET) {
  def init(mfile:String, pfile:String) = net.init(mfile, pfile)
}

class Classifier (val net:Net) {
  
  def init(model_file:String, pretrained_file:String, image_dims:Array[Int] = null, 
      gpu:Boolean = false, mean_file:String = null, input_scale:Double = 1.0, channel_swap:Array[Int] = null) = {
    
    net.init(model_file, pretrained_file);
    
    CAFFE.set_phase(1);
    
    CAFFE.set_mode(if (gpu) 1 else 0)
    
//    if (mean_file != null) net.set_mean(mean_file)
    
//    if (input_scale != 1.0) net.set_input_scale(input_scale)
    
  }
  
}
class SGDSolver (val sgd:SGDSOLVER) {
  val net = sgd.net
  
  def Solve = sgd.Solve
  
  def SolveResume(fname:String) = sgd.SolveResume(fname)
  
}

object SGDSolver {
  def apply(paramFile:String):SGDSolver = new SGDSolver(new SGDSOLVER(paramFile))
}



