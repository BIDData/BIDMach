package BIDMach.caffe
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,FND,GMat,GIMat,GSMat,HMat,Image,IMat,ND,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import edu.berkeley.bvlc.SGDSOLVER
import edu.berkeley.bvlc.NET
import edu.berkeley.bvlc.CAFFE

class Classifier {
  
  val net = new Net
  
  def init(model_file:String, pretrained_file:String, image_dims:Array[Int] = Array(256, 256), 
      gpu:Boolean = false, mean_file:String = null, input_scale:Float = 1f, channel_swap:IMat = 2\1\0) = {
    
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
    
  def classify(im:Image) = {
  	val fnd = net.preprocess(im)
  	net.clear_inputs
  	net.add_input(fnd, 0)
  	net.forward
  	net.output_data(?,?,?,0)
  }


}



