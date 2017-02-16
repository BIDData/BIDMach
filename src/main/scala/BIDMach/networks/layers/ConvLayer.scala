package BIDMach.networks.layers

import BIDMat.FFilter._
import BIDMat.Filter._


import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat,TMat,FFilter,FND}
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
import java.util.Arrays
// import java.util.List

/* Many issues to think of    
   How to consider bias...(maybe very difficult?)
*/

class ConvolutionLayer(override val net:Net, override val opts:ConvolutionNodeOpts = new ConvolutionNode ) extends ModelLayer(net, opts, 2) {
    var filter:FFilter = null;
    var updateFilter:FFilter = null;
    var bias_mat:FND = null;
    var update_bias_mat:FND = null;

  def initModelMat(imageDim:IMat, initFilter:Boolean, initBias:Boolean = false):Mat = {
    // image should be something like - val image = FND(irow(channel_in,h,w,n));

    val channel_in = imageDim(0);
    val filter_h = opts.kernel(0); // 3;0
    val filter_w = opts.kernel(1); // 3;
    val npad = opts.pad(0); //1;
    val nstride = opts.stride(0); // 1;
    val channel_out = opts.noutputs // actually # of filters;

    if(initBias){
      // initialize bias matrix, should be the size of channel_out*h*w, would be applied to n samples
      val biasDim = Array[Float](channel_out,filter_h,filter_w)
      if(initFilter){
        bias_mat = FND(biasDim, new Array[Float](channel_out*filter_h*filter_w))
        bias_mat.asMat;
      }
      else{
        update_bias_mat = FND(biasDim, new Array[Float](channel_out*filter_h*filter_w))
        update_bias_mat.asMat;
      }
    } 
    else if(initFilter){ // if true, we are initializing initFilter, not updateFilter
      filter = FFilter2Ddn(filter_h,filter_w,channel_in,channel_out,nstride,npad);
      filter.apply(rand(filter.asMat)-0.5f); // How to randomize? using rand(out:FND)?
      filter.asMat;
    } 
    else{
      updateFilter = FFilter2Ddn(filter_h,filter_w,channel_in,channel_out,nstride,npad);
      updateFilter.asMat;
    }
  }

  override def forward = {
    val start = toc;
    if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
      modelmats(imodel) = initModelMat(inputData.dims,true); //Set the model
      if(opts.hasBias) modelmats(imodel+1) = initModelMat(inputData.dims,true,true); // Set the bias in modelmats
      updatemats(imodel) = initModelMat(inputData.dims,false);  // Set the updatemats to be another FFilter // Set it to 0?
      if(opts.hasBias) updatemats(imodel+1) = initModelMat(inputData.dims,false,true); // Set the bias in modelmats
    }

    //Load the modelmats back to the layer's own filter
    filter.apply(modelmats(imodel))
    updateFilter.apply(updatemats(imodel))

    if(opts.hasBias){
      bias_mat.apply(modelmats(imodel+1))
      update_bias_mat.apply(updatemats(imodel+1))
    }

    if (output.asInstanceOf[AnyRef] == null){ // if output not exist, should make a result to know the exact dimension of output
      var result = filter*inputData;
      createOutput(result.dims);
      if(opts.hasBias) result+=bias_mat // We want it to broadcast on the n samples.
      output = result;
    }
    else{
      if(opts.hasBias){
        output ~ filter*inputData + bias_mat;
      }
      else output ~ filter*inputData; // actually it's same as filter.convolve(inputData)
    }
    // Not considering Bias for now (probably a little bit complicated?)
    clearDeriv;
    forwardtime += toc - start;
  }

  override def backward = {
    val start = toc;
    // Guess: convolveT - the gradient of input data
    //        convolveM - the gradient of the current Model (Filter)

    //Load the modelmats back to the layer's own filter
    filter.apply(modelmats(imodel))
    updateFilter.apply(updatemats(imodel))

    if(opts.hasBias){
      bias_mat.apply(modelmats(imodel+1))
      update_bias_mat.apply(updatemats(imodel+1))
    }

    //Sum the matrix along the last dimension (of sample size n), need to find the correct syntax
    //update_bias_mat ~ deriv.sum(axis = -1)
    //updatemats(imodel+1) = update_bias_mat

    // deriv is the backwarded gradient of the following layers (same dimension as output)
    // inputDeriv is the derivative you will compute to assign to the input
    if (inputDeriv.asInstanceOf[AnyRef] != null) {
      filter.convolveT(deriv.asInstanceOf[FND], inputDeriv.asInstanceOf[FND],true) // it actually put the computation result 
      //inputDeriv.asMat = modelmats(imodel)^*(deriv.asMat);  // ^* is actually filter.convolveT(b)
    }
    else{
      filter.convolveT(deriv.asInstanceOf[FND], inputDeriv.asInstanceOf[FND],true) // Have to check whether to set doclear = true?
      //inputDeriv.asMat ~ modelmats(imodel)^*(deriv.asMat);
    }

    updateFilter.convolveM(inputData.asInstanceOf[FND],deriv.asInstanceOf[FND])

    //save back the updateFilter
    updatemats(imodel) = updateFilter.asMat

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
  var dilation:List[Integer] = null //was dilation:List[Integer] = Arrays.asList(1)
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
