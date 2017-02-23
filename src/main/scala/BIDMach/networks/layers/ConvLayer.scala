package BIDMach.networks.layers

import BIDMat.FFilter._
import BIDMat.GFilter._
import BIDMat.Filter._

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat,TMat,FFilter,FND,GND,Filter,ND,GFilter}
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

/* Many issues to think of    
   How to consider bias...(maybe very difficult?)
*/

class ConvolutionLayer(override val net:Net, override val opts:ConvolutionNodeOpts = new ConvolutionNode ) extends ModelLayer(net, opts, 2) {
    var filter:Filter = null; 
    var updateFilter:Filter = null;
    var bias_mat:ND = null; // it should be size (channel_out*1*1*1), to better broadcast?
    var update_bias_mat:ND = null;
    var inputDim:IMat = null; // Should be three numbers
    var outputDim:IMat = null; //Should be three numbers
    var outputND:ND = null;
    var inputDataND:ND = null

  def initModelMat(initFilter:Boolean, initBias:Boolean = false):Mat = {
    // image should be something like - val image = FND(irow(channel_in,h,w,n));
    inputDim = opts.imageDim
    val channel_in = inputDim(0);
    val filter_h = opts.kernel(0); // 3;0
    val filter_w = opts.kernel(1); // 3;
    val npad = opts.pad(0); //1;
    val nstride = opts.stride(0); // 1;
    val channel_out = opts.noutputs // actually # of filters;

    if (initBias){
      // initialize bias matrix, should be the size of channel_out*h*w, would be applied to n samples
      val biasDim = Array[Int](channel_out,filter_h,filter_w)
      if(initFilter){
        bias_mat = FND(biasDim, new Array[Float](channel_out*filter_h*filter_w))
        bias_mat.asMat
      }
      else{
        update_bias_mat = FND(biasDim, new Array[Float](channel_out*filter_h*filter_w))
        update_bias_mat.asMat
      }
    } 
    else if (initFilter) { // if true, we are initializing initFilter, not updateFilter
      if (opts.useGPU) {
        filter = GFilter2Ddn(filter_h,filter_w,channel_in,channel_out,nstride,npad)
        filter = new GFilter(filter.inDims,filter.outDims,filter.stride,filter.pad,filter.outPad,
	                     (rand(filter.asInstanceOf[GFilter].asMat)-0.5f).data)
        filter.asInstanceOf[GFilter].asMat
      } else {
        filter = FFilter2Ddn(filter_h,filter_w,channel_in,channel_out,nstride,npad)
        filter = new FFilter(filter.inDims,filter.outDims,filter.stride,filter.pad,filter.outPad,
	                     (rand(filter.asInstanceOf[FFilter].asMat)-0.5f).data)
        filter.asInstanceOf[FFilter].asMat
      }
    } 
    else{
      if (opts.useGPU) {
        updateFilter = GFilter2Ddn(filter_h,filter_w,channel_in,channel_out,nstride,npad);
        updateFilter.asInstanceOf[GFilter].asMat
      } else {
        updateFilter = FFilter2Ddn(filter_h,filter_w,channel_in,channel_out,nstride,npad);
        updateFilter.asInstanceOf[FFilter].asMat
      }
    }
  }

  override def forward = {
    val start = toc;

    // Must get the real inputData.dims in  (channel_in,h,w,n)
    var inputDataNDdims = opts.imageDim \ inputData.ncols
    if (inputDataND.asInstanceOf[AnyRef] == null) {
      if (opts.useGPU) {
        inputDataND = GND(inputDataNDdims.data)
      } else {
        inputDataND = FND(inputDataNDdims.data)
      }
    }
    inputDataND <-- inputData

    if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
      modelmats(imodel) = initModelMat(true); //Set the model
      if(opts.hasBias) modelmats(imodel+1) = initModelMat(true,true); // Set the bias in modelmats
      updatemats(imodel) = initModelMat(false);  // Set the updatemats to be another FFilter // Set it to 0?
      if(opts.hasBias) updatemats(imodel+1) = initModelMat(false,true); // Set the bias in modelmats
    }

    /*
    if(opts.hasBias){
      bias_mat = FND(modelmats(imodel+1),bias_mat.dims)
      update_bias_mat = FND(updatemats(imodel+1),update_bias_mat.dims)
    }
    */

    if (output.asInstanceOf[AnyRef] == null){ // if output not exist, should make a result to know the exact dimension of output
      var outputBatchDim = Filter.getOutputDims(
        inputDataND.dims, filter.inDims, filter.outDims, filter.stride, filter.pad, filter.outPad)
      outputDim = outputBatchDim(?, 0 -> 3)
      if (opts.useGPU) {
        output = GMat.zeros(outputDim.data.reduce(_*_), inputData.ncols)
        outputND = new GND(outputBatchDim.data, output.asInstanceOf[GMat].data)
      } else {
        output = FMat.zeros(outputDim.data.reduce(_*_), inputData.ncols)
        outputND = new FND(outputBatchDim.data, output.asInstanceOf[FMat].data)
      }
    }

    if (opts.hasBias) {
      filter.convolve(inputDataND, outputND, true)
      outputND ~ outputND + bias_mat
    } else {
      filter.convolve(inputDataND, outputND, true)
    }

    clearDeriv
    forwardtime += toc - start
  }

  override def backward = {
    val start = toc;
    // Guess: convolveT - the gradient of input data
    //        convolveM - the gradient of the current Model (Filter)
    
    if(opts.hasBias){
      bias_mat.apply(modelmats(imodel+1))
      update_bias_mat.apply(updatemats(imodel+1))
    }

    //Sum the matrix along the last dimension (of sample size n), need to find the correct syntax
    //update_bias_mat ~ deriv.sum(axis = -1)
    //updatemats(imodel+1) = update_bias_mat

    var inputDataNDdims = opts.imageDim \ inputData.ncols
    var inputDataND:ND = null
    if (opts.useGPU) {
      inputDataND = new GND(inputDataNDdims.data, inputData.asInstanceOf[GMat].data)
    } else {
      inputDataND = new FND(inputDataNDdims.data, inputData.asInstanceOf[FMat].data)
    }

    var derivNDdims = outputDim \ output.ncols
    var derivND:ND = null
    if (opts.useGPU) {
      derivND = new GND(derivNDdims.data, deriv.asMat.asInstanceOf[GMat].data)
    } else {
      derivND = new FND(derivNDdims.data, deriv.asMat.asInstanceOf[FMat].data)
    }

    // deriv is the backwarded gradient of the following layers (same dimension as output)
    // inputDeriv is the derivative you will compute to assign to the input

    if (inputDeriv.asInstanceOf[AnyRef] != null) {      
      if (opts.useGPU) {
        filter.convolveT(derivND, 
                         inputDeriv.asInstanceOf[GND].reshapeView(inputDataND.dims.data), true)
      } else {
        filter.convolveT(derivND, 
                         inputDeriv.asInstanceOf[FND].reshapeView(inputDataND.dims.data), true)
      }
    }

    updateFilter.convolveM(inputDataND, derivND, true)

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
  var pad:IMat = null
  var kernel:IMat = null
  var stride:IMat = null
  var dilation:IMat = null //was dilation:List[Integer] = Arrays.asList(1)
  var group:Int = 1
  var axis:Int = 1
  var forceND:Boolean = false
  var imageDim:IMat = null // it should be something like 1*28*28 for MNIST, i.e. channel_in*h*w
  var useGPU:Boolean = true

  def copyOpts(opts:ConvolutionNodeOpts):ConvolutionNodeOpts = {
      super.copyOpts(opts);
      opts.noutputs = noutputs;
      opts.hasBias = hasBias;
      opts.pad = pad;
      opts.kernel = kernel;
      opts.stride = stride;
      opts.dilation = dilation;
      opts.group = group;
      opts.axis = axis;
      opts.forceND = forceND;
      opts.useGPU = useGPU;
      opts;
  }

}

class ConvolutionNode extends Node with ConvolutionNodeOpts {

  def copyTo(opts:ConvolutionNode):ConvolutionNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }

  override def clone:ConvolutionNode = {
    copyTo(new ConvolutionNode ).asInstanceOf[ConvolutionNode]
  }
  
  override def create(net:Net):ConvolutionLayer = {
    ConvolutionLayer(net, this)
  }

  override def toString = {
    "convolution@" + Integer.toHexString(hashCode() % 0x10000)
  }

}

object ConvolutionLayer {
  
  def apply(net:Net) = new ConvolutionLayer(net, new ConvolutionNode)
  
  def apply(net:Net, opts:ConvolutionNodeOpts) = new ConvolutionLayer(net, opts)

}
