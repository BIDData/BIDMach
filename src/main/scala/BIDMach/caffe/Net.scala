package BIDMach.caffe
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,FND,GMat,GIMat,GSMat,HMat,Image,IMat,ND,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import edu.berkeley.bvlc.SGDSOLVER
import edu.berkeley.bvlc.NET
import edu.berkeley.bvlc.CAFFE

// Caffe Images are W < H < D (< N), Java images are D < W < H, Matlab means file is W < H < D

class Net () {
  
  val _net = new NET
  
  def initIO = {
    input_data = new Array[FND](num_inputs)
   	for (i <- 0 until num_inputs) {
   	  val iblob = _net.input_blob(i)
   	  input_data(i) = FND(iblob.width, iblob.height, iblob.channels, iblob.num)
   	}
   	output_data = new Array[FND](num_outputs)
   	for (i <- 0 until num_outputs) {
   	  val oblob = _net.output_blob(i)
   	  output_data(i) = FND(oblob.width, oblob.height, oblob.channels, oblob.num)
   	}
  }
  
  def init(modelfile:String, paramfile:String) = {
  	_net.init(modelfile, paramfile)
  	initIO
  }
  
  def init(modelfile:String) = {
  	_net.init(modelfile)
  	initIO
  }
  
  def num_inputs = _net.num_inputs()
  
  def num_outputs = _net.num_outputs()

  def inchannels = _net.input_blob(0).channels
  
  def inwidth = _net.input_blob(0).width
  
  def inheight = _net.input_blob(0).height
  
  def outchannels = _net.output_blob(0).channels
  
  def outwidth = _net.output_blob(0).width
  
  def outheight = _net.output_blob(0).height
  
  def num = _net.input_blob(0).num
  
  def set_mean(mfile:String, varname:String = "image_mean") = {
    var meanf:FND = load(mfile, varname)                                   // Matlab means file is W < H < D, BGR
    println("mean file %d %d %d" format(meanf.dims(0), meanf.dims(1), meanf.dims(2)))
    if (meanf.dims(0) != _image_dims(0) || meanf.dims(1) != _image_dims(1)) {
    	meanf = meanf.transpose(2, 0, 1)                                     // First go to resizing order D < W < H
    	meanf = Image(meanf).resize(inwidth, inheight).toFND                 // Resize if needed
    	meanf = meanf.transpose(1, 2, 0)                                     // Now back to W < H < D
    }
    meanf = crop(meanf)
    _mean = meanf
  }
  
  def set_input_scale(v:Float) = {_scale = v};
  
  def set_channel_swap(v:IMat) = {_channel_swap = v};
  
  def set_image_dims(dd:Array[Int]) = {_image_dims = dd};  
  
  def forward() = {
    push_inputs
    _net.forward
    pull_outputs
  }
  
  def backward() = _net.backward
  
  def preprocess(im:Image):FND = {                                          // Preprocess a D < W < H image
    var cafimg = im.toFND;
    if (cafimg.dims(1) != _image_dims(0) || cafimg.dims(2) != _image_dims(1)) {
      cafimg = Image(cafimg).resize(_image_dims(0), _image_dims(1)).toFND;
    }
    if (_scale != 1f) {
      cafimg = cafimg *@ _scale;
    }
    if (_channel_swap.asInstanceOf[AnyRef] != null) {
      cafimg = cafimg(_channel_swap, ?, ?);
    }
    cafimg = cafimg.transpose(1, 2, 0);                                   // to W < H < D
    cafimg = crop(cafimg);
    if (_mean.asInstanceOf[AnyRef] != null) {
      cafimg = cafimg - _mean;
    }
    cafimg;
  }
  
  def crop(im:FND):FND = {                                                // Image should be D < W < H
    if (im.dims(0) > inwidth || im.dims(1) > inheight) {
      val x0 = (im.dims(0) - inwidth)/2;
      val y0 = (im.dims(1) - inheight)/2;
      val x1 = x0 + inwidth;
      val y1 = y0 + inheight;
      im(icol(x0->x1), icol(y0->y1), ?);
    } else {
      im
    }
  }
  
  def clear_inputs = {
  	for (i <- 0 until num_inputs) {
  		input_data(i).clear 
  	}
  }
  
  def add_input(im:FND, i:Int, j:Int) = {
    val inblob = _net.input_blob(0)
    if (im.dims(0) != inblob.width || im.dims(1) != inblob.height || im.dims(2) != inblob.channels) {
      throw new RuntimeException("add_input dimensions mismatch")
    } else if (i < 0 || i >= num) {
      throw new RuntimeException("add_input index out of range %d %d" format (i, num))
    }
    input_data(i)(?,?,?,j) = im
  }
  
  def push_inputs = {
    for (i <- 0 until num_inputs) {
    	_net.input_blob(i).put_data(input_data(i).data)
    }
  }
  
  def pull_outputs = {
  	for (i <- 0 until num_outputs) {
  		_net.output_blob(i).get_data(output_data(i).data)
  	}
  }
  
  private var _mean:FND = null
  
  private var _scale:Float = 1f
  
  private var _channel_swap:IMat = null
  
  private var _image_dims:Array[Int] = null
  
  var input_data:Array[FND] = null
  
  var output_data:Array[FND] = null
  
}



