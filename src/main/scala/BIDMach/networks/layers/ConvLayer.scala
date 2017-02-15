package BIDMach.networks.layers

import BIDMat.FFilter._
import BIDMat.Filter._


import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat,TMat,FFilter}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach._
import edu.berkeley.bid.CPUMACH
import edu.berkeley.bid.CUMACH
import scala.util.hashing.MurmurHash3
import java.util.HashMap
import BIDMach.networks._
// import java.util.Arrays
// import java.util.List

/* Many issues to think of    
   How to consider bias...(maybe very difficult?)
*/

class ConvolutionLayer(override val net:Net, override val opts:ConvolutionNodeOpts = new ConvolutionNode) extends ModelLayer(net, opts) {

  def initModelMat(imageDim:IMat):FFilter = {
    // image should be something like - val image = FND(irow(channel_in,h,w,n));

    val channel_in = imageDim(0);
    val filter_h = opts.kernel(0); // 3;0
    val filter_w = opts.kernel(1); // 3;
    val npad = opts.pad(0); //1;
    val nstride = opts.stride(0); // 1;
    val channel_out = opts.noutputs // actually # of filters;
    
    var filter = FFilter2Ddn(filter_h,filter_w,channel_in,channel_out,nstride,npad);
    filter = rand(filter)-0.5f; // How to randomize? using rand(out:FND)?
    filter;
  }

  override def forward = {
    val start = toc;
    if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
      modelmats(imodel) = initModelMat(inputData.dims); //Set the model
      updatemats(imodel) = initModelMat(inputData.dims);  // Set the updatemats to be another FFilter
      updatemats(imodel) = 0;
    }

    if (output.asInstanceOf[AnyRef] == null){ // if output not exist, should make a result to know the exact dimension of output
      var result = modelmats(imodel)*inputData;
      createOutput(result.dims);
      output = result;
    }
    else{
      output ~ modelmats(imodel)*inputData; // actually it's same as filter.convolve(inputData)
    }
    // Not considering Bias for now (probably a little bit complicated?)
    clearDeriv;
    forwardtime += toc - start;
  }

  override def backward = {
    val start = toc;
    // Guess: convolveT - the gradient of input data
    //        convolveM - the gradient of the current Model (Filter)

    // deriv is the backwarded gradient of the following layers (same dimension as output)
    // inputDeriv is the derivative you will compute to assign to the input
    if (inputDeriv.asInstanceOf[AnyRef] != null) {
      inputDeriv.asMat = modelmats(imodel)^*(deriv.asMat);  // ^* is actually filter.convolveT(b)
    }
    else{
      inputDeriv.asMat ~ modelmats(imodel)^*(deriv.asMat);
    }

    updatemats(imodel).convolveM(inputData,deriv.asMat);

    //Should we handle the update of updatemats(imodel)? I think it should be handled in learner?
    backwardtime += toc - start;
  }

  override def toString = {
    "convolution@" + Integer.toHexString(hashCode() % 0x10000)
  }

}

trait ConvolutionNodeOpts extends ModelNodeOpts {
  var noutputs:Int = 0
  var hasBias:Boolean = true
  var pad:List[Integer] = null
  var kernel:List[Integer] = null
  var stride:List[Integer] = null
  var dilation:List[Integer] = Arrays.asList(1)
  var group:Int = 1
  var axis:Int = 1
  var forceND:Boolean = false
}

class ConvolutionNode extends Node with ConvolutionNodeOpts {

  override def clone:ConvolutionNode = copyTo(new ConvolutionNode).asInstanceOf[ConvolutionNode]
  
  override def create(net:Net):ConvolutionLayer = ConvolutionLayer(net, this)
  
  override def toString = {
    "convolution@" + Integer.toHexString(hashCode() % 0x10000)
  }

}

object ConvolutionLayer {
  
  def apply(net:Net) = new ConvolutionLayer(net, new ConvolutionNode)
  
  def apply(net:Net, opts:ConvolutionNodeOpts) = new ConvolutionLayer(net, opts)

}
