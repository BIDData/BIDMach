package BIDMach.models
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,FND,GMat,GIMat,GSMat,HMat,Image,IMat,ND,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import edu.berkeley.bvlc.SGDSOLVER
import edu.berkeley.bvlc.NET
import edu.berkeley.bvlc.CAFFE

// Caffe Images are W < H < D (< N), Java images are D < W < H, Matlab means file is W < H < D

class Net (_net:NET) {
  def init(mfile:String, pfile:String) = _net.init(mfile, pfile)
  
  def inwidth = _net.input_blob(0).width
  
  def inheight = _net.input_blob(0).height
  
  def set_mean(mfile:String, varname:String = "image_mean") = {
    var meanf:FND = load(mfile, varname)                                   // Matlab means file is W < H < D, BGR
    if (meanf.dims(0) != inwidth || meanf.dims(1) != inheight) {
    	meanf = meanf.transpose(2, 0, 1)                                     // First go to resizing order D < W < H
    	meanf = Image(meanf).resize(inwidth, inheight).toFND                 // Resize if needed
    	meanf = meanf.transpose(1, 2, 0)                                     // Now back to W < H < D
    }
    _mean = meanf
  }
  
  def set_input_scale(v:Float) = {_scale = v};
  
  def set_channel_swap(v:IMat) = {_channel_swap = v};
  
  def set_image_dims(dd:Array[Int]) = {_image_dims = dd};  
  
  def preprocess(im:FND):FND = {                                          // Preprocess a D < W < H image
    var cafimg = im;
    if (cafimg.dims(1) != _image_dims(0) || cafimg.dims(2) != _image_dims(1)) {
      cafimg = Image(cafimg).resize(_image_dims(0), _image_dims(1)).toFND  
    }
    if (_scale != 1f) {
      cafimg = cafimg *@ _scale
    }
    if (_channel_swap.asInstanceOf[AnyRef] != null) {
      cafimg = cafimg(_channel_swap, ?, ?)
    }
    cafimg = cafimg.transpose(1, 2, 0)                                    // to W < H < D
    if (_mean.asInstanceOf[AnyRef] != null) {
      cafimg = cafimg - _mean
    }
    cafimg
  }
  
  private var _mean:FND = null
  
  private var _scale:Float = 1f
  
  private var _channel_swap:IMat = null
  
  private var _image_dims:Array[Int] = null
  
}

class Classifier (val net:Net) {
  
  def init(model_file:String, pretrained_file:String, image_dims:Array[Int] = Array(256, 256), 
      gpu:Boolean = false, mean_file:String = null, input_scale:Float = 255f, channel_swap:IMat = 2\1\0) = {
    
    net.init(model_file, pretrained_file);
    
    CAFFE.set_phase(1);
    
    CAFFE.set_mode(if (gpu) 1 else 0)
        
    if (image_dims != null) {
      net.set_image_dims(image_dims)
    } else {
      net.set_image_dims(Array(net.inwidth, net.inheight))
    }
    
    if (mean_file != null) net.set_mean(mean_file)
    
    if (input_scale != 1f) net.set_input_scale(input_scale)
    
    if (channel_swap.asInstanceOf[AnyRef] != null) net.set_channel_swap(channel_swap)

    
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



