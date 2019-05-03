package BIDMach.networks.layers

import jcuda._
import jcuda.runtime._
import jcuda.runtime.JCuda._
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GFilter,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._
import BIDMach.networks._


class LayerNormScaleLayer(override val net:Net, override val opts:LayerNormScaleNodeOpts = new LayerNormScaleNode) extends ModelLayer(net, opts, 4) {
   
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

  var cudnnMainHandle:cudnnHandle = null;
  var cudnnMainStream:cudaStream_t = null;
  
  def initHandles() = { 
    cudnnMainHandle = new cudnnHandle;
    cudnnMainStream = new cudaStream_t;
    var err = cudnnCreate(cudnnMainHandle);
    if (err == 0) err = cudaStreamCreate(cudnnMainStream);
    if (err == 0) err = cudnnSetStream(cudnnMainHandle, cudnnMainStream);
    
    if (err != 0) throw new RuntimeException("Error in CUDNN LayerNormScaleLayer creation %s" format cudaGetErrorString(err))
  }
  
  def initModelMats = {
    initHandles();
    val bdims = inputData.dims.copy;

    batchDim = irow(0->(inputData.dims.length-1));
    bdims(0->(inputData.dims.length-1)) = 1;

    if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
    	modelmats(imodel) = convertMat(ones(bdims));
    	modelmats(imodel+1) = modelmats(imodel).zeros(bdims);
    	modelmats(imodel+2) = modelmats(imodel).zeros(bdims);
    	modelmats(imodel+3) = modelmats(imodel).zeros(bdims);
    }
    updatemats(imodel) = modelmats(imodel).zeros(bdims);
    updatemats(imodel+1) = modelmats(imodel).zeros(bdims);
    if (lr_scales.asInstanceOf[AnyRef] != null) {
    	lr_scales(imodel) = opts.lr_scale;
    	lr_scales(imodel+1) = opts.bias_scale;
    }
    net.l2reg_scales(imodel) = opts.weight_decay_scale;
    net.l2reg_scales(imodel+1) = opts.weight_decay_scale;
    scale = modelmats(imodel);
    bias = modelmats(imodel+1);
    runningMeans = modelmats(imodel+2);
    runningVariances = modelmats(imodel+3)
    updateScale = updatemats(imodel);
    updateBias = updatemats(imodel+1);
  }

  override def forward = {
    val start = toc;
    inplaceNoConnectGetOutput(true);

    if (batchDim.asInstanceOf[AnyRef] == null) initModelMats;
    
    if (Mat.hasCUDA > 0 && net.opts.useGPU && Mat.hasCUDNN) {
      if (net.predicting) {
      	forwardCUDNN(false);     
      } else {
      	forwardCUDNN(true);        
      }
    } else {
      forwardGeneric;
    }  
    forwardtime += toc - start;
  }
  
  override def backward = {
    val start = toc;
    inplaceNoConnectGetInputDerivs();
    
    if (inputDeriv.asInstanceOf[AnyRef] != null) {
    	if (Mat.hasCUDA > 0 && net.opts.useGPU && Mat.hasCUDNN) {
    		backwardCUDNN
    	} else {
    		backwardGeneric
    	}
    }
    
    inplaceNoConnectReleaseDeriv()
    backwardtime += toc - start;
  }
  
    
  def forwardGeneric = {
    // Do LayerNorm
    means = inputData.mean(batchDim);
    variances = inputData.variance(batchDim) + opts.epsilon;
    sdevs = sqrt(variances);
    output ~ inputData - means;
    output ~ output / sdevs;                       // Works even if output = input;
    
    // Now do scale and bias:    
    output ~ output *@ scale;
    output ~ output + bias;
  }
  
  def backwardGeneric = {
    // Do scale and bias first
    updateScale ~ updateScale + (inputData dotr deriv);    
    if (opts.hasBias) updateBias ~ updateBias + deriv.sum(batchDim);
    deriv ~ scale *@ deriv;
    inputDeriv ~ inputDeriv + (deriv / sdevs);
  }
  
  // TODO: enable exceptions instead?
  def forwardCUDNN(dotrain:Boolean) = {
    var xDesc:cudnnTensorDescriptor = null;
    var yDesc:cudnnTensorDescriptor = null;
    var scaleBiasMeanVarDesc:cudnnTensorDescriptor = null;
    
    val cuTensorFormat = Net.getCUDNNformat(opts.tensorFormat, net.opts.tensorFormat);
    val xdims = inputData.dims;

    if (means.asInstanceOf[AnyRef] == null) {
    	means = convertMat(zeros(scale.dims));
    	variances = convertMat(zeros(scale.dims));
    }
    
    val inputGMat = inputData.asInstanceOf[GMat];
    val outputGMat = output.asInstanceOf[GMat];
    val meansGMat = means.asInstanceOf[GMat];
    val variancesGMat = variances.asInstanceOf[GMat];
    val runningMeansGMat = runningMeans.asInstanceOf[GMat];
    val runningVariancesGMat = runningVariances.asInstanceOf[GMat];
    val scaleGMat = scale.asInstanceOf[GMat];
    val biasGMat = bias.asInstanceOf[GMat];
    
    val normMode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;
    
    try {
      xDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(xDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val xSetStatus = cudnnSetTensor4dDescriptor(xDesc, cuTensorFormat, data_type, 1, xdims(3), xdims(0)*xdims(2), xdims(1));
      if (xSetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm forward");
      
      yDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(yDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val ySetStatus = cudnnSetTensor4dDescriptor(yDesc, cuTensorFormat, data_type, 1, xdims(3), xdims(0)*xdims(2), xdims(1))
      if (ySetStatus > 0) throw new CUDAException(ySetStatus, "Error creating y tensor for batch norm forward");
      
      scaleBiasMeanVarDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(scaleBiasMeanVarDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val sbmvDeriveStatus = cudnnDeriveBNTensorDescriptor(scaleBiasMeanVarDesc, xDesc, normMode);
      if (sbmvDeriveStatus == cudnnStatus.CUDNN_STATUS_BAD_PARAM) throw new CUDAException(sbmvDeriveStatus, "Error creating scale/bias/mean/var tensor for batch norm forward")

      var err = if (dotrain) {
        cudnnBatchNormalizationForwardTraining(cudnnMainHandle, normMode, BatchNormScaleLayer.ONE, BatchNormScaleLayer.ZERO, 
            xDesc, inputGMat.pdata, yDesc, outputGMat.pdata, scaleBiasMeanVarDesc, scaleGMat.pdata, biasGMat.pdata, 
            opts.expAvgFactor, runningMeansGMat.pdata, runningVariancesGMat.pdata, opts.epsilon, meansGMat.pdata, variancesGMat.pdata);
      } else {
      	cudnnBatchNormalizationForwardInference(cudnnMainHandle, normMode, BatchNormScaleLayer.ONE, BatchNormScaleLayer.ZERO, 
      	    xDesc, inputGMat.pdata, yDesc, outputGMat.pdata, scaleBiasMeanVarDesc, scaleGMat.pdata,
      	    biasGMat.pdata, runningMeansGMat.pdata, runningVariancesGMat.pdata, opts.epsilon);        
      }
      cudaStreamSynchronize(cudnnMainStream);
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
    
    val normMode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;
    
    try {
      xDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(xDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val xSetStatus = cudnnSetTensor4dDescriptor(xDesc, cuTensorFormat, data_type, 1, xdims(3), xdims(0)*xdims(2), xdims(1));
      if (xSetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward");
      
      dyDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(dyDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val dySetStatus = cudnnSetTensor4dDescriptor(dyDesc, cuTensorFormat, data_type, 1, xdims(3), xdims(0)*xdims(2), xdims(1));
      if (dySetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward");
      
      dxDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(dxDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val dxSetStatus = cudnnSetTensor4dDescriptor(dxDesc, cuTensorFormat, data_type, 1, xdims(3), xdims(0)*xdims(2), xdims(1));
      if (dxSetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward");
      
      scaleBiasDiffDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(scaleBiasDiffDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val sbmvDeriveStatus = cudnnDeriveBNTensorDescriptor(scaleBiasDiffDesc, xDesc, normMode)
      if (sbmvDeriveStatus == cudnnStatus.CUDNN_STATUS_BAD_PARAM) throw new CUDAException(sbmvDeriveStatus, "Error creating scale/bias diff tensor for batch norm backward")
      
      cudnnBatchNormalizationBackward(cudnnMainHandle, normMode,
          BatchNormScaleLayer.ONE, BatchNormScaleLayer.ZERO, BatchNormScaleLayer.ONE, BatchNormScaleLayer.ZERO, 
          xDesc, inputGMat.pdata, dyDesc, derivGMat.pdata, dxDesc, inputDerivGMat.pdata,
          scaleBiasDiffDesc, scaleGMat.pdata, updateScaleGMat.pdata, updateBiasGMat.pdata, opts.epsilon,
          meansGMat.pdata, variancesGMat.pdata);

      cudaStreamSynchronize(cudnnMainStream);

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
    cudnnMainHandle = null;
    cudnnMainStream = null;
  }

  override def toString = {
    "lns@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

trait LayerNormScaleNodeOpts extends ModelNodeOpts {
  var hasBias:Boolean = true;
  var expAvgFactor:Float = 1.0f;                  
  var epsilon:Float = 1e-4f;
  weight_decay_scale = 0f;

  def copyOpts(opts:LayerNormScaleNodeOpts):LayerNormScaleNodeOpts = {
		super.copyOpts(opts);
		opts.hasBias = hasBias;
		opts.expAvgFactor = expAvgFactor;
		opts.epsilon = epsilon;
		opts.inplace = inplace;
		opts;
  }
}

class LayerNormScaleNode extends ModelNode with LayerNormScaleNodeOpts {
  
  def copyTo(opts:LayerNormScaleNode):LayerNormScaleNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }
  override def clone:LayerNormScaleNode = copyTo(new LayerNormScaleNode).asInstanceOf[LayerNormScaleNode]

  override def create(net:Net) = LayerNormScaleLayer(net, this)
  
  override def toString = {
    "lns@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

@SerialVersionUID(100L)
object LayerNormScaleLayer {
  
  val ONE = Pointer.to(Array(1.0f));
  val ZERO = Pointer.to(Array(0.0f));

  def apply(net:Net) = new LayerNormScaleLayer(net);
  
  def apply(net:Net, opts:LayerNormScaleNodeOpts) = new LayerNormScaleLayer(net, opts);

}

