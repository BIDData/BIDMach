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
  import BatchNormScaleLayer._
   
  var means:Mat = null;
  var variances:Mat = null;
  var sdevs:Mat = null;
  
  var scale:Mat = null;
  var bias:Mat = null;
  var updateScale:Mat = null;
  var updateBias:Mat = null;
  
  var debugMe = false;
  var batchDim:IMat = null;
  
   def initModelMats = {
    val dims = inputData.dims.copy;
    dims(dims.length-1) = 1;
    batchDim = irow(dims.length-1);
    scale = convertMat(ones(dims));
    bias = convertMat(zeros(dims));
    updateScale = convertMat(zeros(dims));
    updateBias = convertMat(zeros(dims));
    modelmats(imodel) = scale;
    modelmats(imodel+1) = bias;
    updatemats(imodel) = updateScale;
    updatemats(imodel+1) = updateBias;
  }

  override def forward = {
    val start = toc;
    createOutput
    if (scale.asInstanceOf[AnyRef] == null) initModelMats;
    
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
    sdevs = sqrt(variances);
    output ~ inputData - means;                    // minimize tmp storage;
    output ~ output / sdevs;                       // Works even if output = input;
    
    // Now do scale and bias:    
    output ~ output *@ scale;
    output ~ output + bias;
  }
  
  // Backward formula is (deriv - mean(deriv,2))/sdevs - (in - mean(in,2))^2 *@ deriv /std^3 - (in - mean(in,2))^2 * deriv / std^3
  // Compute it as ((1 - ((in -mean(in))^2/var)) *@ deriv - mean(deriv,2) - ((in -mean(in))^2/var) dotr deriv)/sdevs to save tmp storage
  
  def backwardGeneric = {
    if (debugMe) {
      
      // Do scale and bias first
    	updateScale ~ updateScale + (inputData dotr deriv);    
    	if (opts.hasBias) updateBias ~ updateBias + deriv.sum(batchDim);
    	deriv ~ scale *@ deriv;
      
    	// Now do BatchNorm
      val diff = inputData - means;
      val normedvar = diff *@ diff / variances;
      val terms = (1f - normedvar) *@ deriv - (deriv.mean(batchDim) + (normedvar dotr deriv));
      inputDeriv ~ inputDeriv + (terms / sdevs);
      
    } else {
      
    	// Do scale and bias first
    	updateScale ~ updateScale + (inputData dotr deriv);    
    	if (opts.hasBias) updateBias ~ updateBias + deriv.sum(batchDim);
    	deriv ~ scale *@ deriv;
    	
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
  }
  
  // TODO: enable exceptions instead?
  def forwardCUDNN = {
    var xDesc:cudnnTensorDescriptor = null
    var yDesc:cudnnTensorDescriptor = null
    var scaleBiasMeanVarDesc:cudnnTensorDescriptor = null

    if (means.asInstanceOf[AnyRef] == null) {
    	means = GMat.make(Array(inputData.dims(0), inputData.dims(2), inputData.dims(1)));
    	variances = GMat.make(Array(inputData.dims(0), inputData.dims(2), inputData.dims(1)));
    	scale = GMat.ones(irow(1, inputData.dims(0), 1, 1));
    	bias = GMat.zeros(irow(1, inputData.dims(0), 1, 1));
    }
    
    val inputGMat = inputData.asInstanceOf[GMat];
    val outputGMat = output.asInstanceOf[GMat];
    val meansGMat = means.asInstanceOf[GMat];
    val variancesGMat = variances.asInstanceOf[GMat];
    val scaleGMat = scale.asInstanceOf[GMat];
    val biasGMat = bias.asInstanceOf[GMat];
    
    try {
      xDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(xDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) {
        xDesc = null
        throw new OutOfMemoryError()
      }
      val xSetStatus = cudnnSetTensor4dDescriptor(xDesc, TENSOR_FORMAT, DATA_TYPE, inputData.dims(3), inputData.dims(0), inputData.dims(2), inputData.dims(1))
      if (xSetStatus > 0) {
        throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm forward, bad stride?")
      }
      
      yDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(yDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) {
        yDesc = null
        throw new OutOfMemoryError()
      }
      val ySetStatus = cudnnSetTensor4dDescriptor(yDesc, TENSOR_FORMAT, DATA_TYPE, inputData.dims(3), inputData.dims(0), inputData.dims(2), inputData.dims(1))
      if (ySetStatus > 0) {
        throw new CUDAException(ySetStatus, "Error creating y tensor for batch norm forward, bad stride?")
      }
      
      scaleBiasMeanVarDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(scaleBiasMeanVarDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) {
        scaleBiasMeanVarDesc = null
        throw new OutOfMemoryError()
      }
      val sbmvDeriveStatus = cudnnDeriveBNTensorDescriptor(scaleBiasMeanVarDesc, xDesc, cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL)
      if (sbmvDeriveStatus == cudnnStatus.CUDNN_STATUS_BAD_PARAM) {
        throw new CUDAException(sbmvDeriveStatus, "Error creating scale/bias/mean/var tensor for batch norm forward, bad stride?")
      }

      var err = cudnnBatchNormalizationForwardTraining(getHandle, cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL,
        ONE, ZERO, xDesc, inputGMat.pdata, yDesc, outputGMat.pdata, scaleBiasMeanVarDesc, scaleGMat.pdata,
        biasGMat.pdata, 1.0, new Pointer(), new Pointer(), opts.epsilon, meansGMat.pdata, variancesGMat.pdata);
      
      cudaDeviceSynchronize()
      if (err == 0) {
        err = cudaGetLastError()
      }
      if (err > 0) {
        throw new CUDAException(err, "Error in CUDNN forward batch normalization: " + cudaGetErrorString(err))
      }
          
    } finally {
      if (scaleBiasMeanVarDesc != null) {
        cudnnDestroyTensorDescriptor(scaleBiasMeanVarDesc)
      }
      if (yDesc != null) {
        cudnnDestroyTensorDescriptor(yDesc)
      }
      if (xDesc != null) {
        cudnnDestroyTensorDescriptor(xDesc)
      }
    }
  }
  

  
  def backwardCUDNN = {
    var xDesc:cudnnTensorDescriptor = null
    var dyDesc:cudnnTensorDescriptor = null
    var dxDesc:cudnnTensorDescriptor = null
    var scaleBiasDiffDesc:cudnnTensorDescriptor = null
    
    val inputGMat = inputData.asInstanceOf[GMat];
    val outputGMat = output.asInstanceOf[GMat];
    val meansGMat = means.asInstanceOf[GMat];
    val variancesGMat = variances.asInstanceOf[GMat];
    val scaleGMat = scale.asInstanceOf[GMat];
    val biasGMat = bias.asInstanceOf[GMat];
    val derivGMat = deriv.asInstanceOf[GMat];
    val inputDerivGMat = inputDeriv.asInstanceOf[GMat];
    val updateScaleGMat = updateScale.asInstanceOf[GMat];
    val updateBiasGMat = updateBias.asInstanceOf[GMat];
    
    try {
      // TODO: try and avoid duplication of this
      xDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(xDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) {
        xDesc = null
        throw new OutOfMemoryError()
      }
      val xSetStatus = cudnnSetTensor4dDescriptor(xDesc, TENSOR_FORMAT, DATA_TYPE, inputData.dims(3), inputData.dims(0), inputData.dims(2), inputData.dims(1))
      if (xSetStatus > 0) {
        throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward, bad stride?")
      }
      
      dyDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(dyDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) {
        dyDesc = null
        throw new OutOfMemoryError()
      }
      val dySetStatus = cudnnSetTensor4dDescriptor(dyDesc, TENSOR_FORMAT, DATA_TYPE, inputData.dims(3), inputData.dims(0), inputData.dims(2), inputData.dims(1))
      if (dySetStatus > 0) {
        throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward, bad stride?")
      }
      
      dxDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(dxDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) {
        dxDesc = null
        throw new OutOfMemoryError()
      }
      val dxSetStatus = cudnnSetTensor4dDescriptor(dxDesc, TENSOR_FORMAT, DATA_TYPE, inputData.dims(3), inputData.dims(0), inputData.dims(2), inputData.dims(1))
      if (dxSetStatus > 0) {
        throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward, bad stride?")
      }
      
      scaleBiasDiffDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(scaleBiasDiffDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) {
        scaleBiasDiffDesc = null
        throw new OutOfMemoryError()
      }
      val sbmvDeriveStatus = cudnnDeriveBNTensorDescriptor(scaleBiasDiffDesc, xDesc, cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL)
      if (sbmvDeriveStatus == cudnnStatus.CUDNN_STATUS_BAD_PARAM) {
        throw new CUDAException(sbmvDeriveStatus, "Error creating scale/bias diff tensor for batch norm backward, bad stride?")
      }
      
      cudnnBatchNormalizationBackward(BatchNormScaleLayer.getHandle, cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL,
          ONE, ZERO, ONE, ZERO, xDesc, inputGMat.pdata, dyDesc, derivGMat.pdata, dxDesc, inputDerivGMat.pdata,
          scaleBiasDiffDesc, updateScaleGMat.pdata, updateScaleGMat.pdata, updateBiasGMat.pdata, opts.epsilon,
          meansGMat.pdata, variancesGMat.pdata)

    } finally {
      if (xDesc != null) {
        cudnnDestroyTensorDescriptor(xDesc)
      }
    }
  }
  
  override def toString = {
    "BatchNormScale@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

trait BatchNormScaleNodeOpts extends ModelNodeOpts {
  var hasBias:Boolean = true;
  var epsilon:Float = 1e-5f
}

class BatchNormScaleNode extends Node with BatchNormScaleNodeOpts {
  override def clone:BatchNormScaleNode = copyTo(new BatchNormScaleNode).asInstanceOf[BatchNormScaleNode]

  override def create(net:Net) = BatchNormScaleLayer(net, this)
  
  override def toString = {
    "BatchNormScale@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

object BatchNormScaleLayer {
  // TODO: is this the right tensor format
  val TENSOR_FORMAT = cudnnTensorFormat.CUDNN_TENSOR_NCHW
  val DATA_TYPE = cudnnDataType.CUDNN_DATA_FLOAT
  
  val ONE = Pointer.to(Array(1.0f))
  val ZERO = Pointer.to(Array(0.0f))

  var cudnnContexts:Array[cudnnHandle] = null
  var cudnnContextsInitialized = false

  def apply(net:Net) = new BatchNormScaleLayer(net)
  
  def apply(net:Net, opts:BatchNormScaleNodeOpts) = new BatchNormScaleLayer(net, opts)

  def getHandle = {
    if (!GFilter.cudnnContextsInitialized) GFilter.initHandles()
    GFilter.cudnnContexts(getGPU)
  }
}

// TODO: consider replacing with jcuda.CudaException
class CUDAException(val status:Int, val message:String = null, val cause:Throwable = null)
  extends RuntimeException("CUDA error " + status + (if (message != null) ": " + message else ""), cause) {
}
