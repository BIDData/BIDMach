package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,GFilter, IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach.networks._
import BIDMach._
import jcuda._
import jcuda.runtime.JCuda._
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import scala.util.hashing.MurmurHash3;
import scala.collection.mutable.HashMap;

/**
 * LRN across channel unit
 */

class LRNacrossLayer(override val net:Net, override val opts:LRNacrossNode = new LRNacrossNode) extends Layer(net, opts) {
  
	val data_type = cudnnDataType.CUDNN_DATA_FLOAT;
	val lrnMode = cudnnLRNMode.CUDNN_LRN_CROSS_CHANNEL_DIM1;
	
	val aArray = Array(0f);
	val bArray = Array(0f);
  
	override def clear = {
			clearMats;
	}
	
	  def forwardCUDNN = {
    var xDesc:cudnnTensorDescriptor = null;
    var yDesc:cudnnTensorDescriptor = null;
    var pDesc:cudnnLRNDescriptor = null;
    
    val cuTensorFormat = Net.getCUDNNformat(opts.tensorFormat, net.opts.tensorFormat);
    val xdims = inputData.dims;
    val ydims = output.dims;

    val inputGMat = inputData.asInstanceOf[GMat];
    val outputGMat = output.asInstanceOf[GMat];
    
    val dim = opts.dim;
    val alpha = opts.alpha;
    val beta = opts.beta;
    val lrnK = 2f;
    
    aArray(0) = alpha;
    bArray(0) = beta;
    val pAlpha = Pointer.to(aArray);
    val pBeta = Pointer.to(bArray);
    
    try {
      xDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(xDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val xSetStatus = cudnnSetTensor4dDescriptor(xDesc, cuTensorFormat, data_type, xdims(3), xdims(0), xdims(2), xdims(1));
      if (xSetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for pooling forward");
      
      yDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(yDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val ySetStatus = cudnnSetTensor4dDescriptor(yDesc, cuTensorFormat, data_type, ydims(3), ydims(0), ydims(2), ydims(1))
      if (ySetStatus > 0) throw new CUDAException(ySetStatus, "Error creating y tensor for pooling forward");
      
      pDesc = new cudnnLRNDescriptor;
      cudnnCreateLRNDescriptor(pDesc);
      val cstatus = cudnnSetLRNDescriptor(pDesc, dim, alpha, beta, lrnK);
      if (cstatus > 0) throw new RuntimeException("Error setting LRN descriptor %d" format cstatus);

      var err = cudnnLRNCrossChannelForward(GFilter.getHandle, pDesc, lrnMode, pAlpha, xDesc, inputGMat.pdata, pBeta, yDesc, outputGMat.pdata);
      
      cudaDeviceSynchronize();
      if (err == 0) err = cudaGetLastError();
      if (err > 0) throw new CUDAException(err, "Error in CUDNN forward pooling: " + cudaGetErrorString(err));
          
    } finally {
      if (pDesc != null) cudnnDestroyLRNDescriptor(pDesc);
      if (yDesc != null) cudnnDestroyTensorDescriptor(yDesc);
      if (xDesc != null) cudnnDestroyTensorDescriptor(xDesc);
    }
  }
	  
	    
  def backwardCUDNN = {
    var xDesc:cudnnTensorDescriptor = null;
	  var yDesc:cudnnTensorDescriptor = null;
    var dyDesc:cudnnTensorDescriptor = null;
    var dxDesc:cudnnTensorDescriptor = null;
    var pDesc:cudnnLRNDescriptor = null;
    
    val cuTensorFormat = Net.getCUDNNformat(opts.tensorFormat, net.opts.tensorFormat);
    val xdims = inputData.dims;
    val ydims = output.dims;
    
    val inputGMat = inputData.asInstanceOf[GMat];
    val outputGMat = output.asInstanceOf[GMat];
    val inputDerivGMat = inputDeriv.asInstanceOf[GMat];
    val derivGMat = deriv.asInstanceOf[GMat];
    
    val dim = opts.dim;
    val alpha = opts.alpha;
    val beta = opts.beta;
    val lrnK = 2f;
    
    aArray(0) = alpha;
    bArray(0) = beta;
    val pAlpha = Pointer.to(aArray);
    val pBeta = Pointer.to(bArray);
    
    try {
      xDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(xDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val xSetStatus = cudnnSetTensor4dDescriptor(xDesc, cuTensorFormat, data_type, xdims(3), xdims(0), xdims(2), xdims(1));
      if (xSetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward");
      
      yDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(yDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val ySetStatus = cudnnSetTensor4dDescriptor(yDesc, cuTensorFormat, data_type, ydims(3), ydims(0), ydims(2), ydims(1));
      if (ySetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward");
      
      dxDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(dxDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val dxSetStatus = cudnnSetTensor4dDescriptor(dxDesc, cuTensorFormat, data_type, xdims(3), xdims(0), xdims(2), xdims(1));
      if (dxSetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward");
      
      dyDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(dyDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val dySetStatus = cudnnSetTensor4dDescriptor(dyDesc, cuTensorFormat, data_type, ydims(3), ydims(0), ydims(2), ydims(1));
      if (dySetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward");
      
      pDesc = new cudnnLRNDescriptor;
      cudnnCreateLRNDescriptor(pDesc);
      val cstatus = cudnnSetLRNDescriptor(pDesc, dim, alpha, beta, lrnK);
      if (cstatus > 0) throw new RuntimeException("Error setting LRN descriptor %d" format cstatus);

      cudnnLRNCrossChannelBackward(GFilter.getHandle, pDesc, lrnMode,
          pAlpha, yDesc, outputGMat.pdata, dyDesc, derivGMat.pdata, xDesc, inputGMat.pdata, pBeta, dxDesc, inputDerivGMat.pdata);

    } finally {
      
      if (pDesc != null) cudnnDestroyLRNDescriptor(pDesc);
      if (dyDesc != null) cudnnDestroyTensorDescriptor(dyDesc);
      if (dxDesc != null) cudnnDestroyTensorDescriptor(dxDesc);
      if (yDesc != null) cudnnDestroyTensorDescriptor(yDesc); 
      if (xDesc != null) cudnnDestroyTensorDescriptor(xDesc);                       
    }
  }
  
  	  
  override def toString = {
    "LRNacross@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait LRNacrossNodeOpts extends CompoundNodeOpts {
    var dim = 5;
    var alpha = 1f;
    var beta = 0.5f;
    var tensorFormat:Int = Net.UseNetFormat;
    
   def copyOpts(opts:LRNacrossNodeOpts):LRNacrossNodeOpts = {
  		super.copyOpts(opts);
  		opts.dim = dim;
  		opts.alpha = alpha;
  		opts.beta = beta;
  		opts.tensorFormat = tensorFormat;
  		opts;
    }
}

class LRNacrossNode extends CompoundNode with LRNacrossNodeOpts {	
	 
	  override def clone:LRNacrossNode = {
		  copyTo(new LRNacrossNode).asInstanceOf[LRNacrossNode];
	  }

	  override def create(net:Net):LRNacrossLayer = {
		  LRNacrossLayer(net, this);
	  }
    
    override def toString = {
      "LRNacross@"+Integer.toHexString(hashCode % 0x10000).toString
    }

	}


  
object LRNacrossNode {   
  
  def apply() = {
    val n = new LRNacrossNode;
    n
  }
  
  def apply(opts:LRNacrossNodeOpts) = {
    val n = new LRNacrossNode;
    opts.copyOpts(n);
    n
  }
}

object LRNacrossLayer {    
  
  def apply(net:Net) = new LRNacrossLayer(net, new LRNacrossNode);
  
  def apply(net:Net, opts:LRNacrossNode) = {
    val x = new LRNacrossLayer(net, opts);
    x;
  }
}
