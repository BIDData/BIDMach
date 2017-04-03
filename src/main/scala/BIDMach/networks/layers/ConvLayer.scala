package BIDMach.networks.layers


import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat,TMat,FFilter,Filter,GFilter}
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
    var filter:FMat = null; 
    var ffilter:Filter = null;
    var updateFilter:FMat = null;
    var updateFFilter:Filter = null;
    var bias_mat:FMat = null; // it should be size (channel_out*1*1*1), to better broadcast?
    var update_bias_mat:FMat = null;
    var inputDim:IMat = null; // Should be three numbers
    var outputDim:IMat = null; //Should be three numbers

  def initModelMats = {
    // image should be something like - val image = FND(irow(channel_in,h,w,n));
    inputDim = opts.imageDim
    val channel_in = inputDim(0);
    val filter_h = opts.kernel(0); // 3;0
    val filter_w = opts.kernel(1); // 3;
    val npad = opts.pad(0); //1;
    val nstride = opts.stride(0); // 1;
    val channel_out = opts.noutputs; // actually # of filters;

    
    filter = if (opts.useGPU) {
    	GFilter.GFilter2Ddn(filter_h,filter_w,channel_in,channel_out,nstride,npad); 
    } else {
      FFilter2Ddn(filter_h,filter_w,channel_in,channel_out,nstride,npad);
    }
    ffilter = filter.asInstanceOf[Filter];
    rand(filter);
    filter ~ filter - 0.5f;
    updateFilter = filter.copy;
    updateFFilter = updateFilter.asInstanceOf[Filter];
    if (opts.hasBias) {
      // initialize bias matrix, should be the size of channel_out*h*w, would be applied to n samples
      val biasDim = irow(channel_out,filter_h,filter_w);
      bias_mat = filter.zeros(biasDim);
      update_bias_mat = filter.zeros(biasDim);
    } 
  }

  override def forward = {
    val start = toc;
    
    // Create filter model, filter update and bias model if needed
    if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
      initModelMats
      modelmats(imodel) = filter;
      updatemats(imodel) = updateFilter;
      if (opts.hasBias) {
        modelmats(imodel+1) = bias_mat;
        updatemats(imodel+1) = update_bias_mat;
      }
    }

    // Create the output tensor if needed
    if (output.asInstanceOf[AnyRef] == null){ // if output not exist, should make a result to know the exact dimension of output
    	var outputBatchDim = Filter.getOutputDims(inputData.dims, ffilter.inDims, ffilter.outDims, ffilter.stride, ffilter.pad, ffilter.outPad);
    	output = filter.zeros(outputBatchDim)
    }
    
    output ~ filter * inputData;

    if (opts.hasBias) output ~ output + bias_mat;

    clearDeriv
    forwardtime += toc - start
  }

  override def backward = {
    val start = toc;
    val ndims = output.dims.length;
    
    if(opts.hasBias){
      update_bias_mat ~ update_bias_mat + (deriv.sum(irow(ndims - 1)) / inputData.ncols);
    }

    if (inputDeriv.asInstanceOf[AnyRef] != null) {      
        ffilter.convolveT(deriv, inputDeriv, false)
    }

    updateFFilter.convolveM(inputData, deriv, false)

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
