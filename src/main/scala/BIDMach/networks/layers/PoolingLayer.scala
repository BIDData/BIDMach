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


class PoolingLayer(override val net:Net, override val opts:PoolingNodeOpts = new PoolingNode) extends Layer(net, opts) {
   
  val data_type = cudnnDataType.CUDNN_DATA_FLOAT;
  
  def initModelMats = {
    val outdims = inputData.dims.copy;
    val outh = (outdims(2) + 2*opts.pad - opts.h)/opts.stride + 1;
    val outw = (outdims(1) + 2*opts.pad - opts.w)/opts.stride + 1;
    outdims(2) = outh;
    outdims(1) = outw;
    createOutput(outdims);
  }

  override def forward = {
    val start = toc;
    if (output.asInstanceOf[AnyRef] == null) initModelMats;
    
    if (Mat.hasCUDA > 0 && net.opts.useGPU && Mat.hasCUDNN) {
      forwardCUDNN
    } else {
      forwardGeneric
    }  
    clearDeriv
    forwardtime += toc - start
  }
  
  override def backward = {
    val start = toc;
    if (inputDeriv.asInstanceOf[AnyRef] != null) {
    	if (Mat.hasCUDA > 0 && net.opts.useGPU && Mat.hasCUDNN) {
    		backwardCUDNN
    	} else {
    		backwardGeneric
    	}  
    }
    backwardtime += toc - start
  }
  
    
  def forwardGeneric = {
  		val in = inputData.asInstanceOf[FMat];
  		val out = output.asInstanceOf[FMat];
  		val indims = in.dims;
  		val outdims = out.dims;
  		val nch = outdims(0);
  		val inh = indims(2);
  		val inw = indims(1);
  		val outh = outdims(2);
  		val outw = outdims(1);
  		val stridex = opts.stride;
  		val stridey = opts.stride;
  		var iimg = 0;
  		val nbatch = in.ncols;
  		while (iimg < nbatch) {
  			var x = 0;
  			while (x < outw) {
  				var ixmin = math.max(0, x * stridex - (opts.w-1)/2);
  				var ixmax = math.min(x * stridex + opts.w/2, inw-1);
  				var y = 0;
  				while (y < outh) {
  					var iymin = math.max(0, y * stridey - (opts.h-1)/2);
  					var iymax = math.min(y * stridey + opts.h/2, inh-1);
  					var outi = nch * (x + outw * (y + outh * iimg));

  					var firststep = true;
  					var ix = ixmin;
  					while (ix <= ixmax) {
  						var iy = iymin;
  						while (iy <= iymax) {
  							var ini = nch * (ix + inw * (iy + inh * iimg));
  							if (firststep) {
  								var d = 0;
  								while (d < nch) {
  									out.data(outi + d) = in.data(ini + d);
  									d += 1;
  								}
  							} else {
  								var d = 0;
  								while (d < nch) {
  									out.data(outi + d) = math.max(in.data(ini + d), out.data(outi + d));
  									d += 1;
  								}
  							}
  							firststep = false;
  							iy += 1;
  						}
  						ix += 1;
  					}
  					y += 1;
  				}
  				x += 1;
  			}
  			iimg += 1;
  		}
  }
  
  // Backward formula is (deriv - mean(deriv,2))/sdevs - (in - mean(in,2))^2 *@ deriv /std^3 - (in - mean(in,2))^2 * deriv / std^3
  // Compute it as ((1 - ((in -mean(in))^2/var)) *@ deriv - mean(deriv,2) - ((in -mean(in))^2/var) dotr deriv)/sdevs to save tmp storage
  
  def backwardGeneric = {
  		val in = inputData.asInstanceOf[FMat];
  		val out = output.asInstanceOf[FMat];
  		val inderiv = inputDeriv.asInstanceOf[FMat];
  		val outderiv = deriv.asInstanceOf[FMat];
  		val indims = in.dims;
  		val outdims = out.dims;
  		val nch = outdims(0);
  		val inh = indims(2);
  		val inw = indims(1);
  		val outh = outdims(2);
  		val outw = outdims(1);
  		val stridex = opts.stride;
  		val stridey = opts.stride;
  		var iimg = 0;
  		val nbatch = in.ncols;
  		while (iimg < nbatch) {
  			var x = 0;
  			while (x < outw) {
  				var ixmin = math.max(0, x * stridex - (opts.w-1)/2);
  				var ixmax = math.min(x * stridex + opts.w/2, inw-1);
  				var y = 0;
  				while (y < outh) {
  					var iymin = math.max(0, y * stridey - (opts.h-1)/2);
  					var iymax = math.min(y * stridey + opts.h/2, inh-1);
  					var outi = nch * (x + outw * (y + outh * iimg));

  					var ix = ixmin;
  					while (ix <= ixmax) {
  						var iy = iymin;
  						while (iy <= iymax) {
  							var ini = nch * (ix + inw * (iy + inh * iimg));
  							var d = 0;
  							while (d < nch) {
  								if (out.data(outi + d) == in.data(ini + d)) {
  								  inderiv(ini + d) += outderiv(outi + d);
  								}
  								d += 1;
  							}
  							iy += 1;
  						}
  						ix += 1;
  					}
  					y += 1;
  				}
  				x += 1;
  			}
  			iimg += 1;
  		}
  }

  def forwardCUDNN = {
    var xDesc:cudnnTensorDescriptor = null;
    var yDesc:cudnnTensorDescriptor = null;
    var pDesc:cudnnPoolingDescriptor = null;
    
    val cuTensorFormat = Net.getCUDNNformat(opts.tensorFormat, net.opts.tensorFormat);
    val xdims = inputData.dims;
    val ydims = output.dims;

    val inputGMat = inputData.asInstanceOf[GMat];
    val outputGMat = output.asInstanceOf[GMat];
    
    val h = opts.h;
    val w = opts.w;
    val pady = opts.pad;
    val padx = opts.pad;
    val stridey = opts.stride;
    val stridex = opts.stride;
    
    try {
      xDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(xDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val xSetStatus = cudnnSetTensor4dDescriptor(xDesc, cuTensorFormat, data_type, xdims(3), xdims(0), xdims(2), xdims(1));
      if (xSetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for pooling forward");
      
      yDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(yDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val ySetStatus = cudnnSetTensor4dDescriptor(yDesc, cuTensorFormat, data_type, ydims(3), ydims(0), ydims(2), ydims(1))
      if (ySetStatus > 0) throw new CUDAException(ySetStatus, "Error creating y tensor for pooling forward");
      
      pDesc = new cudnnPoolingDescriptor;
      cudnnCreatePoolingDescriptor(pDesc);
      val cstatus = cudnnSetPooling2dDescriptor(pDesc, opts.poolingMode, opts.poolingNaN, h, w, pady, padx, stridey, stridex);
      if (cstatus > 0) throw new RuntimeException("Error setting pooling descriptor %d" format cstatus);

      var err = cudnnPoolingForward(GFilter.getHandle, pDesc, PoolingLayer.ONE, xDesc, inputGMat.pdata, PoolingLayer.ZERO, yDesc, outputGMat.pdata);
      
      cudaDeviceSynchronize();
      if (err == 0) err = cudaGetLastError();
      if (err > 0) throw new CUDAException(err, "Error in CUDNN forward pooling: " + cudaGetErrorString(err));
          
    } finally {
      if (pDesc != null) cudnnDestroyPoolingDescriptor(pDesc);
      if (yDesc != null) cudnnDestroyTensorDescriptor(yDesc);
      if (xDesc != null) cudnnDestroyTensorDescriptor(xDesc);
    }
  }
  

  
  def backwardCUDNN = {
    var xDesc:cudnnTensorDescriptor = null;
    var yDesc:cudnnTensorDescriptor = null;
    var dyDesc:cudnnTensorDescriptor = null;
    var dxDesc:cudnnTensorDescriptor = null;
    var pDesc:cudnnPoolingDescriptor = null;
    
    val cuTensorFormat = Net.getCUDNNformat(opts.tensorFormat, net.opts.tensorFormat);
    val xdims = inputData.dims;
    val ydims = output.dims;
    
    val inputGMat = inputData.asInstanceOf[GMat];
    val outputGMat = output.asInstanceOf[GMat];
    val derivGMat = deriv.asInstanceOf[GMat];
    val inputDerivGMat = inputDeriv.asInstanceOf[GMat];
    
    val h = opts.h;
    val w = opts.w;
    val pady = opts.pad;
    val padx = opts.pad;
    val stridey = opts.stride;
    val stridex = opts.stride;
    
    try {
      xDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(xDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val xSetStatus = cudnnSetTensor4dDescriptor(xDesc, cuTensorFormat, data_type, xdims(3), xdims(0), xdims(2), xdims(1));
      if (xSetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for pooling backward");
      
      yDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(yDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val ySetStatus = cudnnSetTensor4dDescriptor(yDesc, cuTensorFormat, data_type, ydims(3), ydims(0), ydims(2), ydims(1));
      if (ySetStatus > 0) throw new CUDAException(xSetStatus, "Error creating y tensor for pooling backward");
      
      dxDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(dxDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val dxSetStatus = cudnnSetTensor4dDescriptor(dxDesc, cuTensorFormat, data_type, xdims(3), xdims(0), xdims(2), xdims(1));
      if (dxSetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for pooling backward");
 
      dyDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(dyDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val dySetStatus = cudnnSetTensor4dDescriptor(dyDesc, cuTensorFormat, data_type, ydims(3), ydims(0), ydims(2), ydims(1));
      if (dySetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for pooling backward");
      
      pDesc = new cudnnPoolingDescriptor;
      cudnnCreatePoolingDescriptor(pDesc);
      val cstatus = cudnnSetPooling2dDescriptor(pDesc, opts.poolingMode, opts.poolingNaN, h, w, pady, padx, stridey, stridex);
      if (cstatus > 0) throw new RuntimeException("Error setting pooling descriptor %d" format cstatus);

      var err = cudnnPoolingBackward(GFilter.getHandle, pDesc, PoolingLayer.ONE, yDesc, outputGMat.pdata, dyDesc, derivGMat.pdata,
          xDesc, inputGMat.pdata, PoolingLayer.ONE, dxDesc, inputDerivGMat.pdata);
         
      cudaDeviceSynchronize();
      if (err == 0) err = cudaGetLastError();
      if (err > 0) throw new CUDAException(err, "Error in CUDNN backward pooling: " + cudaGetErrorString(err));

    } finally {
      
      if (pDesc != null) cudnnDestroyPoolingDescriptor(pDesc);
      if (dxDesc != null) cudnnDestroyTensorDescriptor(dxDesc);
      if (dyDesc != null) cudnnDestroyTensorDescriptor(dyDesc);
      if (yDesc != null) cudnnDestroyTensorDescriptor(yDesc); 
      if (xDesc != null) cudnnDestroyTensorDescriptor(xDesc); 
    }
  }
  
  override def toString = {
    "pool@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

trait PoolingNodeOpts extends ModelNodeOpts {
	var h:Int = 1;
	var w:Int = 1;
	var pad:Int = 1;
	var stride:Int = 1;
	var poolingMode:Int = cudnnPoolingMode.CUDNN_POOLING_MAX;
	var poolingNaN:Int = cudnnNanPropagation.CUDNN_PROPAGATE_NAN;

  def copyOpts(opts:PoolingNodeOpts):PoolingNodeOpts = {
		super.copyOpts(opts);
		opts.h = h;
		opts.w = w;
		opts.pad = pad;
		opts.stride = stride;
		opts.poolingMode = poolingMode;
		opts.poolingNaN = poolingNaN;
		opts;
  }
}

class PoolingNode extends Node with PoolingNodeOpts {
  
	def copyTo(opts:PoolingNode):PoolingNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }
  override def clone:PoolingNode = copyTo(new PoolingNode).asInstanceOf[PoolingNode]

  override def create(net:Net) = PoolingLayer(net, this)
  
  override def toString = {
    "pool@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

object PoolingLayer {
  
  val ONE = Pointer.to(Array(1.0f));
  val ZERO = Pointer.to(Array(0.0f));

  def apply(net:Net) = new PoolingLayer(net);
  
  def apply(net:Net, opts:PoolingNodeOpts) = new PoolingLayer(net, opts);

}

