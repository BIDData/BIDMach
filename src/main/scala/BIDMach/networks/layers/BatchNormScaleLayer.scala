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


class BatchNormScaleLayer(override val net:Net, override val opts:BatchNormScaleNodeOpts = new BatchNormScaleNode) extends ModelLayer(net, opts, 2) {
   
  val data_type = cudnnDataType.CUDNN_DATA_FLOAT;
  
  var means:Mat = null;
  var variances:Mat = null;
  var runningMeans:Mat = null;
  var runningVariances:Mat = null;
  var sdevs:Mat = null;
  
  var scale:Mat = null;
  var bias:Mat = null;
  var updateScale:Mat = null;
  var updateBias:Mat = null;
  
  var debugMe = false;
  var batchDim:IMat = null;
  
  def initModelMats = {
    val bdims = inputData.dims.copy;
    opts.batchNormMode match {
    case BatchNormLayer.SPATIAL => {
    	batchDim = irow(1->inputData.dims.length);
    	bdims(1->bdims.length) = 1;

    }
    case BatchNormLayer.PER_ACTIVATION => {
    	batchDim = irow(inputData.dims.length-1);
    	bdims(bdims.length-1) = 1;
    }
    }
    if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
    	modelmats(imodel) = convertMat(ones(bdims));
    	modelmats(imodel+1) = modelmats(imodel).zeros(bdims);
    	updatemats(imodel) = modelmats(imodel).zeros(bdims);
    	updatemats(imodel+1) = modelmats(imodel).zeros(bdims);
    }
    if (lr_scales.asInstanceOf[AnyRef] != null) {
    	lr_scales(imodel) = opts.lr_scale;
    	lr_scales(imodel+1) = opts.bias_scale;
    }
    scale = modelmats(imodel);
    bias = modelmats(imodel+1);
    updateScale = updatemats(imodel);
    updateBias = updatemats(imodel+1);
  }

  override def forward = {
    val start = toc;
    createOutput
    if (batchDim.asInstanceOf[AnyRef] == null) initModelMats;
    
    if (Mat.hasCUDA > 0 && net.opts.useGPU && Mat.hasCUDNN) {
      forwardCUDNN
    } else {
      forwardGeneric
    }  
    clearDeriv
    forwardtime += toc - start
  }
  
  override def backward = {
    val start = toc
    if (Mat.hasCUDA > 0 && net.opts.useGPU && Mat.hasCUDNN) {
      backwardCUDNN
    } else {
      backwardGeneric
    }   
    backwardtime += toc - start
  }
  
    
  def forwardGeneric = {
    // Do BatchNorm
    means = inputData.mean(batchDim);
    variances = inputData.variance(batchDim) + opts.epsilon;
    if (runningMeans.asInstanceOf[AnyRef] == null) {
      runningMeans = means + 0f;
      runningVariances = variances + 0f;
    }
    if (opts.expAvgFactor == 1.0f) {
    	sdevs = sqrt(variances);
      output ~ inputData - means;
    } else {
      runningMeans ~ (runningMeans *@ (1f - opts.expAvgFactor)) + (means *@ opts.expAvgFactor);
      runningVariances ~ (runningVariances *@ (1f - opts.expAvgFactor)) + (variances *@ opts.expAvgFactor);
      sdevs = sqrt(runningVariances);
      output ~ inputData - runningMeans;
    }
    output ~ output / sdevs;                       // Works even if output = input;
    
    // Now do scale and bias:    
    output ~ output *@ scale;
    output ~ output + bias;
  }
  
  // Backward formula is (deriv - mean(deriv,2))/sdevs - (in - mean(in,2))^2 *@ deriv /std^3 - (in - mean(in,2))^2 * deriv / std^3
  // Compute it as ((1 - ((in -mean(in))^2/var)) *@ deriv - mean(deriv,2) - ((in -mean(in))^2/var) dotr deriv)/sdevs to save tmp storage
  
  def backwardGeneric = {
  		// Do scale and bias first
  		updateScale ~ updateScale + (inputData dotr deriv);    
  		if (opts.hasBias) updateBias ~ updateBias + deriv.sum(batchDim);
  		deriv ~ scale *@ deriv;
  		
  		if (opts.expAvgFactor == 1.0) {
  			if (debugMe) {
  				// Now do BatchNorm
  				val diff = inputData - means;
  				val normedvar = diff *@ diff / variances;
  				val terms = (1f - normedvar) *@ deriv - (deriv.mean(batchDim) + (normedvar dotr deriv));
  				inputDeriv ~ inputDeriv + (terms / sdevs);
  			} else {
  				// Now do BatchNorm, minimize tmp storage
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
  		} else {
  		  inputDeriv ~ inputDeriv + (deriv / sdevs);
  		}
  }
  
  // TODO: enable exceptions instead?
  def forwardCUDNN = {
    var xDesc:cudnnTensorDescriptor = null;
    var yDesc:cudnnTensorDescriptor = null;
    var scaleBiasMeanVarDesc:cudnnTensorDescriptor = null;
    
    val cuTensorFormat = Net.getCUDNNformat(opts.tensorFormat, net.opts.tensorFormat);
    val xdims = inputData.dims;

    if (means.asInstanceOf[AnyRef] == null) {
    	means = convertMat(zeros(scale.dims));
    	variances = convertMat(zeros(scale.dims));
    	runningMeans = convertMat(zeros(scale.dims));
    	runningVariances = convertMat(zeros(scale.dims));
    }
    
    val inputGMat = inputData.asInstanceOf[GMat];
    val outputGMat = output.asInstanceOf[GMat];
    val meansGMat = means.asInstanceOf[GMat];
    val variancesGMat = variances.asInstanceOf[GMat];
    val runningMeansGMat = runningMeans.asInstanceOf[GMat];
    val runningVariancesGMat = runningVariances.asInstanceOf[GMat];
    val scaleGMat = scale.asInstanceOf[GMat];
    val biasGMat = bias.asInstanceOf[GMat];
    
    val normMode = opts.batchNormMode match {
      case BatchNormLayer.SPATIAL => cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;
      case BatchNormLayer.PER_ACTIVATION => cudnnBatchNormMode.CUDNN_BATCHNORM_PER_ACTIVATION;
    }
    
    try {
      xDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(xDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val xSetStatus = cudnnSetTensor4dDescriptor(xDesc, cuTensorFormat, data_type, xdims(3), xdims(0), xdims(2), xdims(1));
      if (xSetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm forward");
      
      yDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(yDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val ySetStatus = cudnnSetTensor4dDescriptor(yDesc, cuTensorFormat, data_type, xdims(3), xdims(0), xdims(2), xdims(1))
      if (ySetStatus > 0) throw new CUDAException(ySetStatus, "Error creating y tensor for batch norm forward");
      
      scaleBiasMeanVarDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(scaleBiasMeanVarDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val sbmvDeriveStatus = cudnnDeriveBNTensorDescriptor(scaleBiasMeanVarDesc, xDesc, normMode);
      if (sbmvDeriveStatus == cudnnStatus.CUDNN_STATUS_BAD_PARAM) throw new CUDAException(sbmvDeriveStatus, "Error creating scale/bias/mean/var tensor for batch norm forward")

      var err = cudnnBatchNormalizationForwardTraining(GFilter.getHandle, normMode,
        BatchNormScaleLayer.ONE, BatchNormScaleLayer.ZERO, xDesc, inputGMat.pdata, yDesc, outputGMat.pdata, scaleBiasMeanVarDesc, scaleGMat.pdata,
        biasGMat.pdata, opts.expAvgFactor, runningMeansGMat.pdata, runningVariancesGMat.pdata, opts.epsilon, meansGMat.pdata, variancesGMat.pdata);
      
      cudaDeviceSynchronize();
      if (err == 0) {
        err = cudaGetLastError();
      }
      if (err > 0) {
        throw new CUDAException(err, "Error in CUDNN forward batch normalization: " + cudaGetErrorString(err))
      }
          
    } finally {
      if (scaleBiasMeanVarDesc != null) cudnnDestroyTensorDescriptor(scaleBiasMeanVarDesc);
      if (yDesc != null) cudnnDestroyTensorDescriptor(yDesc);
      if (xDesc != null) cudnnDestroyTensorDescriptor(xDesc);
    }
  }
  

  
  def backwardCUDNN = {
    var xDesc:cudnnTensorDescriptor = null
    var dyDesc:cudnnTensorDescriptor = null
    var dxDesc:cudnnTensorDescriptor = null
    var scaleBiasDiffDesc:cudnnTensorDescriptor = null
    
    val cuTensorFormat = Net.getCUDNNformat(opts.tensorFormat, net.opts.tensorFormat);
    val xdims = inputData.dims;
    
    val inputGMat = inputData.asInstanceOf[GMat];
    val meansGMat = means.asInstanceOf[GMat];
    val variancesGMat = variances.asInstanceOf[GMat];
    val scaleGMat = scale.asInstanceOf[GMat];
    val biasGMat = bias.asInstanceOf[GMat];
    val derivGMat = deriv.asInstanceOf[GMat];
    val inputDerivGMat = inputDeriv.asInstanceOf[GMat];
    val updateScaleGMat = updateScale.asInstanceOf[GMat];
    val updateBiasGMat = updateBias.asInstanceOf[GMat];
    
    val normMode = opts.batchNormMode match {
      case BatchNormLayer.SPATIAL => cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;
      case BatchNormLayer.PER_ACTIVATION => cudnnBatchNormMode.CUDNN_BATCHNORM_PER_ACTIVATION;
    }
    
    try {
      xDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(xDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val xSetStatus = cudnnSetTensor4dDescriptor(xDesc, cuTensorFormat, data_type, xdims(3), xdims(0), xdims(2), xdims(1));
      if (xSetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward");
      
      dyDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(dyDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val dySetStatus = cudnnSetTensor4dDescriptor(dyDesc, cuTensorFormat, data_type, xdims(3), xdims(0), xdims(2), xdims(1));
      if (dySetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward");
      
      dxDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(dxDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val dxSetStatus = cudnnSetTensor4dDescriptor(dxDesc, cuTensorFormat, data_type, xdims(3), xdims(0), xdims(2), xdims(1));
      if (dxSetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward");
      
      scaleBiasDiffDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(scaleBiasDiffDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val sbmvDeriveStatus = cudnnDeriveBNTensorDescriptor(scaleBiasDiffDesc, xDesc, normMode)
      if (sbmvDeriveStatus == cudnnStatus.CUDNN_STATUS_BAD_PARAM) throw new CUDAException(sbmvDeriveStatus, "Error creating scale/bias diff tensor for batch norm backward")
      
      cudnnBatchNormalizationBackward(GFilter.getHandle, normMode,
          BatchNormScaleLayer.ONE, BatchNormScaleLayer.ZERO, BatchNormScaleLayer.ONE, BatchNormScaleLayer.ZERO, 
          xDesc, inputGMat.pdata, dyDesc, derivGMat.pdata, dxDesc, inputDerivGMat.pdata,
          scaleBiasDiffDesc, scaleGMat.pdata, updateScaleGMat.pdata, updateBiasGMat.pdata, opts.epsilon,
          meansGMat.pdata, variancesGMat.pdata)

    } finally {
      
      if (scaleBiasDiffDesc != null) cudnnDestroyTensorDescriptor(scaleBiasDiffDesc);
      if (dxDesc != null) cudnnDestroyTensorDescriptor(dxDesc);
      if (dyDesc != null) cudnnDestroyTensorDescriptor(dyDesc);
      if (xDesc != null) cudnnDestroyTensorDescriptor(xDesc);                       
    }
  }
  
  override def clear = {
    clearMats;  
    means = null;
    variances = null;
    runningMeans = null;
    runningVariances = null;
    sdevs = null;
    scale = null;
    bias = null;
    updateScale= null;
    updateBias = null;
    batchDim = null;
  }

  override def toString = {
    "bns@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

trait BatchNormScaleNodeOpts extends ModelNodeOpts {
	var hasBias:Boolean = true;
  var expAvgFactor:Float = 1.0f;                  
  var epsilon:Float = 1e-4f;
  var batchNormMode:Int = BatchNormLayer.SPATIAL;
  var tensorFormat:Int = Net.UseNetFormat;

  def copyOpts(opts:BatchNormScaleNodeOpts):BatchNormScaleNodeOpts = {
		super.copyOpts(opts);
		opts.hasBias = hasBias;
		opts.expAvgFactor = expAvgFactor;
		opts.epsilon = epsilon;
		opts.batchNormMode = batchNormMode;
		opts.tensorFormat = tensorFormat;
		opts;
  }
}

class BatchNormScaleNode extends Node with BatchNormScaleNodeOpts {
  
	def copyTo(opts:BatchNormScaleNode):BatchNormScaleNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }
  override def clone:BatchNormScaleNode = copyTo(new BatchNormScaleNode).asInstanceOf[BatchNormScaleNode]

  override def create(net:Net) = BatchNormScaleLayer(net, this)
  
  override def toString = {
    "bns@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

object BatchNormScaleLayer {
  
  val ONE = Pointer.to(Array(1.0f));
  val ZERO = Pointer.to(Array(0.0f));

  def apply(net:Net) = new BatchNormScaleLayer(net);
  
  def apply(net:Net, opts:BatchNormScaleNodeOpts) = new BatchNormScaleLayer(net, opts);

}

// TODO: consider replacing with jcuda.CudaException
class CUDAException(val status:Int, val message:String = null, val cause:Throwable = null)
  extends RuntimeException("CUDA error " + status + (if (message != null) ": " + message else ""), cause) {
}
