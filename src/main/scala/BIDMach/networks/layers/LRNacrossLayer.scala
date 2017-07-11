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
import jcuda.runtime._
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
	
	val ZERO = Array(0f);
	val ONE = Array(1f);
	
	val pZERO = Pointer.to(ZERO);
	val pONE = Pointer.to(ONE);


  var cudnnMainHandle:cudnnHandle = null;
  var cudnnMainStream:cudaStream_t = null;
  
  def initHandles() = { 
    cudnnMainHandle = new cudnnHandle;
    cudnnMainStream = new cudaStream_t;
    var err = cudnnCreate(cudnnMainHandle);
    if (err == 0) err = cudaStreamCreate(cudnnMainStream);
    if (err == 0) err = cudnnSetStream(cudnnMainHandle, cudnnMainStream);
    
    if (err != 0) throw new RuntimeException("Error in CUDNN LRNacrossLayer creation %s" format cudaGetErrorString(err))
  }

  override def forward = {
  	val start = toc;
  	inplaceNoConnectGetOutput();
    inputData match {
      case in:GMat => forwardGPU;
      case in:FMat => forwardCPU;
    }
    forwardtime += toc - start;
  }
  
  override def backward = {
  	val start = toc;
  	inplaceNoConnectGetInputDerivs();
    inputData match {
      case in:GMat => backwardGPU;
      case in:FMat => backwardCPU;
    }
    inplaceNoConnectReleaseDeriv();
    backwardtime += toc - start;
  }
  
  def forwardCPU = {
  	val inputFMat = inputData.asInstanceOf[FMat];
    val outputFMat = output.asInstanceOf[FMat];
    if (getTensorFormat != Net.TensorNHWC) throw new RuntimeException("CPU LRN reqruires NHWC")
    val idims = inputFMat.dims.data;
    val npos = idims.slice(1,4).reduce(_*_);
    val k0 = (opts.dim + 1)/2;
    var i = 0;
    while (i < npos) {
    	var j = 0;
    	while (j < idims(0)) {
    		val ii = i * idims(0);
    	  var ij = j + i * ii;
    	  var k = 0;
    	  var ss = 0f;
    	  val klim = math.min(opts.dim, idims(0) - j);
    	  while (k < klim) {
    	    ss += inputFMat.data(k + ij);
    	    k += 1;
    	  }
    	  k = j + opts.dim - idims(0) - 1;
    	  while (k >= 0) {
    	  	ss += inputFMat.data(k + ii);
    	    k -= 1;   
    	  }
    	  val k1 = ((k0 + j) % idims(0)) + ii;
    	  outputFMat.data(k1) = inputFMat.data(k1) / math.pow(1 + opts.alpha * ss, opts.beta).toFloat;
    		j += 1;
    	} 
      i += 1;
    } 
  }
  
  def backwardCPU = {
  	val inputFMat = inputData.asInstanceOf[FMat];
    val outputFMat = output.asInstanceOf[FMat];
    val iderivFMat = inputDeriv.asInstanceOf[FMat];
    val derivFMat = deriv.asInstanceOf[FMat];
    if (getTensorFormat != Net.TensorNHWC) throw new RuntimeException("CPU LRN reqruires NHWC")
    val idims = inputFMat.dims.data;
    val npos = idims.slice(1,4).reduce(_*_);
    val k0 = (opts.dim + 1)/2;
    var i = 0;
    while (i < npos) {
    	var j = 0;
    	while (j < idims(0)) {
    	  val ii = i * idims(0);
    	  val ij = j + ii;
    	  var k = 0;
    	  var ss = 0f;
    	  val klim = math.min(opts.dim, idims(0) - j);
    	  while (k < klim) {
    	    ss += inputFMat.data(k + ij);
    	    k += 1;
    	  }
    	  k = j + opts.dim - idims(0) - 1;
    	  while (k >= 0) {
    	  	ss += inputFMat.data(k + ii);
    	    k -= 1;   
    	  }
    	  val k1 = ((k0 + j) % idims(0)) + ii;
    	  val denom0 = 1 + opts.alpha * ss
    	  val denom = math.pow(denom0, opts.beta).toFloat
    	  iderivFMat.data(k1) += derivFMat.data(k1) / denom;
    	  val ddenom = - derivFMat.data(k1) * opts.alpha * opts.beta / (denom * denom0);
    	  k = 0;
    	  while (k < opts.dim && k + j < idims(0)) {
    	    iderivFMat.data(k + ij) += ddenom;
    	    k += 1;
    	  }
    	  k = j + opts.dim - idims(0) - 1;
    	  while (k >= 0) {
    	  	iderivFMat.data(k + ii) += ddenom;
    	    k -= 1;   
    	  }    	  
    		j += 1;
    	} 
      i += 1;
    } 
  }
	
	def forwardGPU = {
		if (cudnnMainHandle.asInstanceOf[AnyRef] == null) initHandles();
		
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
    val lrnK = opts.k;
    
    try {
      xDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(xDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val xSetStatus = cudnnSetTensor4dDescriptor(xDesc, cuTensorFormat, data_type, xdims(3), xdims(0), xdims(2), xdims(1));
      if (xSetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for LRN across channel forward");
      
      yDesc = new cudnnTensorDescriptor();
      if (cudnnCreateTensorDescriptor(yDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
      val ySetStatus = cudnnSetTensor4dDescriptor(yDesc, cuTensorFormat, data_type, ydims(3), ydims(0), ydims(2), ydims(1))
      if (ySetStatus > 0) throw new CUDAException(ySetStatus, "Error creating y tensor for LRN across channel forward");
      
      pDesc = new cudnnLRNDescriptor;
      cudnnCreateLRNDescriptor(pDesc);
      val cstatus = cudnnSetLRNDescriptor(pDesc, dim, alpha, beta, lrnK);
      if (cstatus > 0) throw new RuntimeException("Error setting LRN descriptor %d" format cstatus);

      var err = cudnnLRNCrossChannelForward(cudnnMainHandle, pDesc, lrnMode, pONE, xDesc, inputGMat.pdata, pZERO, yDesc, outputGMat.pdata);
      
      cudaStreamSynchronize(cudnnMainStream);
      if (err == 0) err = cudaGetLastError();
      if (err > 0) throw new CUDAException(err, "Error in CUDNN forward LRN cross channel: " + cudaGetErrorString(err));
          
    } finally {
      if (pDesc != null) cudnnDestroyLRNDescriptor(pDesc);
      if (yDesc != null) cudnnDestroyTensorDescriptor(yDesc);
      if (xDesc != null) cudnnDestroyTensorDescriptor(xDesc);
    }
  }
	  
  def backwardGPU = {
		
		if (inputDeriv.asInstanceOf[AnyRef] != null) {
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


			try {
				xDesc = new cudnnTensorDescriptor()
				if (cudnnCreateTensorDescriptor(xDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
				val xSetStatus = cudnnSetTensor4dDescriptor(xDesc, cuTensorFormat, data_type, xdims(3), xdims(0), xdims(2), xdims(1));
				if (xSetStatus > 0) throw new CUDAException(xSetStatus, "Error creating x tensor for LRN across channel backward");

				yDesc = new cudnnTensorDescriptor()
				if (cudnnCreateTensorDescriptor(yDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
				val ySetStatus = cudnnSetTensor4dDescriptor(yDesc, cuTensorFormat, data_type, ydims(3), ydims(0), ydims(2), ydims(1));
				if (ySetStatus > 0) throw new CUDAException(xSetStatus, "Error creating y tensor for LRN across channel backward");

				dxDesc = new cudnnTensorDescriptor();
				if (cudnnCreateTensorDescriptor(dxDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
				val dxSetStatus = cudnnSetTensor4dDescriptor(dxDesc, cuTensorFormat, data_type, xdims(3), xdims(0), xdims(2), xdims(1));
				if (dxSetStatus > 0) throw new CUDAException(xSetStatus, "Error creating dx tensor for LRN across channel backward");

				dyDesc = new cudnnTensorDescriptor();
				if (cudnnCreateTensorDescriptor(dyDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) throw new OutOfMemoryError();
				val dySetStatus = cudnnSetTensor4dDescriptor(dyDesc, cuTensorFormat, data_type, ydims(3), ydims(0), ydims(2), ydims(1));
				if (dySetStatus > 0) throw new CUDAException(xSetStatus, "Error creating dy tensor for LRN across channel backward");

				pDesc = new cudnnLRNDescriptor;
				cudnnCreateLRNDescriptor(pDesc);
				val cstatus = cudnnSetLRNDescriptor(pDesc, dim, alpha, beta, lrnK);
				if (cstatus > 0) throw new RuntimeException("Error setting LRN descriptor %d" format cstatus);

				var err = cudnnLRNCrossChannelBackward(cudnnMainHandle, pDesc, lrnMode, pONE, yDesc, outputGMat.pdata, 
						dyDesc, derivGMat.pdata, xDesc, inputGMat.pdata, pONE, dxDesc, inputDerivGMat.pdata);

				cudaStreamSynchronize(cudnnMainStream);
				if (err == 0) err = cudaGetLastError();
				if (err > 0) throw new CUDAException(err, "Error in CUDNN backward LRN cross channel: " + cudaGetErrorString(err));
          
			} finally {

				if (pDesc != null) cudnnDestroyLRNDescriptor(pDesc);
				if (dyDesc != null) cudnnDestroyTensorDescriptor(dyDesc);
				if (dxDesc != null) cudnnDestroyTensorDescriptor(dxDesc);
				if (yDesc != null) cudnnDestroyTensorDescriptor(yDesc); 
				if (xDesc != null) cudnnDestroyTensorDescriptor(xDesc);                       
			}
		}
  }
  
  override def clear {
    clearMats;
    cudnnMainHandle = null;
    cudnnMainStream = null;
  }
  
  	  
  override def toString = {
    "LRNacross@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait LRNacrossNodeOpts extends CompoundNodeOpts {
    var dim = 5;
    var alpha = 1f;
    var beta = 0.5f;
    var k = 2f;
    
   def copyOpts(opts:LRNacrossNodeOpts):LRNacrossNodeOpts = {
  		super.copyOpts(opts);
  		opts.dim = dim;
  		opts.alpha = alpha;
  		opts.beta = beta;
  		opts.k = k;
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
