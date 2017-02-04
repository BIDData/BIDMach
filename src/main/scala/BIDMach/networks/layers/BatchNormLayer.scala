package BIDMach.networks.layers

import jcuda._
import jcuda.runtime.JCuda._
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GFilter,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._
import BIDMach.networks._


class BatchNormLayer(override val net:Net, override val opts:BatchNormNodeOpts = new BatchNormNode) extends Layer(net, opts) {

  var means:Mat = null
  var variances:Mat = null
  var sdevs:Mat = null;
  var batchDim:IMat = null;
  var debugMe = false;

  override def forward = {
    val cuTensorFormat = Net.getCUDNNformat(opts.tensorFormat, net.opts.tensorFormat);
    if (opts.batchNormMode == BatchNormLayer.SPATIAL && cuTensorFormat == cudnnTensorFormat.CUDNN_TENSOR_NCHW) {
      throw new RuntimeException("Spatial BatchNorm with NCHW tensors requires CUDNN fused BatchNormScaleLayer");
    }
    val start = toc;
    inplaceNoConnectGetOutput();
      
    if (batchDim.asInstanceOf[AnyRef] == null) {
    	batchDim = opts.batchNormMode match {
    	case BatchNormLayer.SPATIAL => irow(1->inputData.dims.length);
    	case BatchNormLayer.PER_ACTIVATION => irow(inputData.dims.length-1);
    	}
    }
     
    means = inputData.mean(batchDim);
    variances = inputData.variance(batchDim) + opts.epsilon;
    sdevs = sqrt(variances);
    output ~ inputData - means;                    // minimize tmp storage;
    output ~ output / sdevs;                       // Works even if output = input;
 
    forwardtime += toc - start
  }
  
  override def backward = {
    val start = toc;
    inplaceNoConnectGetInputDerivs();

    if (debugMe) {
    	val diff = inputData - means;
    	val normedvar = diff *@ diff / variances;
    	val terms = (1f - normedvar) *@ deriv - (deriv.mean(batchDim) + (normedvar dotr deriv));
    	inputDeriv ~ inputDeriv + (terms / sdevs);
    } else {
    	// minimize tmp storage
    	val tmp = inputData - means;
    	tmp ~ tmp *@ tmp;
    	tmp ~ tmp / variances;                     // Normalized per-element variance (normedvar)
    	val lastTerm = tmp dotr deriv;
    	tmp ~ 1f - tmp;
    	tmp ~ tmp *@ deriv;
    	tmp ~ tmp - (lastTerm + deriv.mean(batchDim));
    	tmp ~ tmp / sdevs;
    	inputDeriv ~ inputDeriv + tmp;
    }  

    inplaceNoConnectReleaseDeriv()
    backwardtime += toc - start
  }
  
  override def clear = {
    clearMats;
    means = null;
    variances = null;
    sdevs= null;
    batchDim = null;
  }

  override def toString = {
    "batchnorm@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

trait BatchNormNodeOpts extends ModelNodeOpts {
	var expAvgFactor:Float = 1f;
  var epsilon:Float = 1e-4f;
  var batchNormMode = BatchNormLayer.SPATIAL;
  
  def copyOpts(opts:BatchNormNodeOpts):BatchNormNodeOpts = {
      super.copyOpts(opts);
      opts.expAvgFactor = expAvgFactor;
      opts.epsilon = epsilon;
      opts.batchNormMode = batchNormMode;
      opts;
  }
}

class BatchNormNode extends Node with BatchNormNodeOpts {
  
  def copyTo(opts:BatchNormNode):BatchNormNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }
  override def clone:BatchNormNode = copyTo(new BatchNormNode).asInstanceOf[BatchNormNode]

  override def create(net:Net) = BatchNormLayer(net, this)
  
  override def toString = {
    "batchnorm@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

object BatchNormLayer {
  final val SPATIAL = 1;
  final val PER_ACTIVATION = 2;

  def apply(net:Net) = new BatchNormLayer(net)
  
  def apply(net:Net, opts:BatchNormNodeOpts) = new BatchNormLayer(net, opts)

}
