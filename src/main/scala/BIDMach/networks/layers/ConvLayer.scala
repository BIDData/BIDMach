package BIDMach.networks.layers

import BIDMach.networks._
import java.util.Arrays
import java.util.List




/* Many issues to think of
1. outDims seems to be of dimension 2, showing the output dimension (how it affects in filter?)
  FFilter shows:
  // The filter is stored with the output dimension major, and the other dimensions in the same order as a data tensor.
  // so a 4D filter block would be OHWC. This matches the ordering in NVIDIA CUDNN.

  But I don't see a member to store "number of output" in FFilter.

  Otherwise we must use modelmats to store noutputs matrix?

2. updatemats(imodel)

3. How to consider bias...(maybe very difficult?)

*/

class ConvolutionLayer(override val net:Net, override val opts:ConvolutionNodeOpts = new ConvolutionNode) extends ModelLayer(net, opts) {
  // var some data
  //var filter:FFilter = null; // Should put this is modelmats(imodel)?

  def initModelMat(inDims0:IMat, outDims0:IMat, stride0:IMat, pad0:IMat, outPad0:IMat, data0:Array[Float]):Mat = {
    val filter = FFilter(inDims0, outDims0, stride0, pad0, outPad0, data0);
    filter = rand(filter)-0.5f; // How to randomize? using rand(out:FND)?
    filter;
  }

  override def forward = {
    val start = toc;
    if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
      modelmats(imodel) = convertMat(initModelMat(opts.kernel, opts.noutputs, opts.stride, opts.pad, null, null)); //Set the model
      updatemats(imodel) = modelmats(imodel).zeros(modelmats(imodel).dims);  // Set the updatemats
      // Alternative way to make updatemats(imodel) also a FFilter
      // updatemats(imodel) = convertMat(initModelMat(opts.kernel, opts.noutputs, opts.stride, opts.pad, null, null));
      // updatemats(imodel) = 0;
    }

    if (output.asInstanceOf[AnyRef] == null){ // if output not exist, should make a result to know the exact dimension of output
      val result = modelmats(imodel)*inputData;
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

    // deriv is the backwarded gradient of the following layers
    // inputDeriv is the derivative you will compute to assign to the input
    if (inputDeriv.asInstanceOf[AnyRef] != null) {
      inputDeriv.asMat = modelmats(imodel)^*(deriv.asMat);  // ^* is actually filter.convolveT(b)
    }
    else{
      inputDeriv.asMat ~ modelmats(imodel)^*(deriv.asMat);
    }

    val grad = FFilter(opts.kernel, opts.noutputs, opts.stride, opts.pad, null, null); 
    // Make a new one because I don't want convolveM to mess up the modelmats(imodel)
    grad.convolveM(inputData,output); // convolveM(a,b) - a is input, b should be output, grad itself should be the gradient?
    updatemats(imodel) <-- grad;
    // Alternatively, can we set updatemats(model) to be a FFilter, with the same parameters?
    // So here it would be:

    // updatemats(imodel).convolveM(inputData,output);


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
