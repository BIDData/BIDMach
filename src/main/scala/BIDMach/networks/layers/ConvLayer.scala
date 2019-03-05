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
import jcuda._
import jcuda.runtime._
import jcuda.runtime.JCuda._
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import scala.util.hashing.MurmurHash3
import java.util.HashMap
import BIDMach.networks._
import java.util.Arrays

@SerialVersionUID(100L)
class ConvLayer(override val net:Net, override val opts:ConvNodeOpts = new ConvNode ) extends ModelLayer(net, opts, 2) {
    var filter:FMat = null; 
    var ffilter:Filter = null;
    var updateFilter:FMat = null;
    var updateFFilter:Filter = null;
    var bias_mat:FMat = null; // it should be size (channel_out*1*1*1), to better broadcast?
    var update_bias_mat:FMat = null;
    var inputDim:IMat = null; // Should be three numbers
    var backwardfiltertime = 0.0;
    var backwarddatatime = 0.0;
    var bwdFilterWS:Mat = null;
//    var outputDim:IMat = null; //Should be three numbers

  var cudnnMainHandle:cudnnHandle = null;
  var cudnnMainStream:cudaStream_t = null;
  
  def initHandles() = { 
    cudnnMainHandle = new cudnnHandle;
    cudnnMainStream = new cudaStream_t;
    var err = cudnnCreate(cudnnMainHandle);
    if (err == 0) err = cudaStreamCreate(cudnnMainStream);
    if (err == 0) err = cudnnSetStream(cudnnMainHandle, cudnnMainStream);
    
    if (err != 0) throw new RuntimeException("Error in CUDNN ConvLayer creation %s" format cudaGetErrorString(err))
  }

  def initModelMats = {
    initHandles();
    inputDim = inputData.dims;
    val channel_in = inputDim(0);
    val filter_h = opts.kernel(0); // 3;0
    val filter_w = opts.kernel(1); // 3;
    val npad = opts.pad(0); //1;
    val nstride = opts.stride(0); // 1;
    val channel_out = opts.noutputs; // actually # of filters;

    val hasInitData = modelmats(imodel).asInstanceOf[AnyRef] != null;
    modelmats(imodel) = if (net.opts.useGPU && Mat.hasCUDA > 0) { 
      val x = GFilter.GFilter2Ddn(filter_h,filter_w,channel_in,channel_out,nstride,npad,modelmats(imodel).asInstanceOf[GMat]);
      x.setTensorFormat(Net.getCUDNNformat(opts.tensorFormat, net.opts.tensorFormat));
      x.convType = Net.getCUDNNconvType(opts.convType, net.opts.convType);
      x;
    } else { 
      FFilter.FFilter2Ddn(filter_h,filter_w,channel_in,channel_out,nstride,npad,modelmats(imodel).asInstanceOf[FMat]);
    }
    filter = modelmats(imodel).asInstanceOf[FMat];
    ffilter = modelmats(imodel).asInstanceOf[Filter];   	
    updatemats(imodel) = ffilter.copy.asInstanceOf[FMat];
    	
    if (!hasInitData) opts.initfn(filter, opts.initv);
    	
    val outDim = Filter.getOutputDims(inputData.dims, ffilter.inDims, ffilter.outDims, ffilter.stride, ffilter.pad, ffilter.outPad);
    val biasDim = irow(outDim(0), 1, 1, 1);
    	
    if (opts.hasBias) { 
      if (modelmats(imodel+1).asInstanceOf[AnyRef] == null) {
	    modelmats(imodel+1) = modelmats(imodel).zeros(biasDim);
	    opts.initbiasfn(modelmats(imodel+1), opts.initbiasv);
      }
      updatemats(imodel+1) = modelmats(imodel).zeros(biasDim); 		    	
    }
    bwdFilterWS = modelmats(0).zeros(filter_h\filter_w\1\channel_out);
    
    if (lr_scales.asInstanceOf[AnyRef] != null) {
    	lr_scales(imodel) = opts.lr_scale;
    	lr_scales(imodel+1) = opts.bias_scale;
    }
    if (output.asInstanceOf[AnyRef] == null) { 
    	val outputBatchDim = Filter.getOutputDims(inputData.dims, ffilter.inDims, ffilter.outDims, ffilter.stride, ffilter.pad, ffilter.outPad);
    	output = filter.zeros(outputBatchDim);
    };
    
    updateFilter = updatemats(imodel).asInstanceOf[FMat];
    updateFFilter = updatemats(imodel).asInstanceOf[Filter];  
    if (updateFilter.asInstanceOf[AnyRef] != null) updateFilter.clear;
    
    bias_mat = modelmats(imodel+1).asInstanceOf[FMat];
    update_bias_mat = updatemats(imodel+1).asInstanceOf[FMat];
    if (opts.hasBias && update_bias_mat.asInstanceOf[AnyRef] != null) update_bias_mat.clear;
  }

  override def forward = {
    val start = toc;    
    // Create filter model, filter update and bias model if needed
    if (inputDim.asInstanceOf[AnyRef] == null) initModelMats;
    inplaceNoConnectSetupDerivs(true);
    
    val workspace = if (Net.getPlacing(opts.inplace, net.opts.inplace) == Net.InPlace) deriv else null;   
    ffilter.convolve(inputData, output, true, workspace);
    if (Net.getPlacing(opts.inplace, net.opts.inplace) == Net.InPlace) deriv.clear;

    if (opts.hasBias) {
      applyBias(bias_mat, output);
    }
    forwardtime += toc - start    
  }

  override def backward = {
    val start = toc;
    inplaceNoConnectGetInputDerivs();
    
    if(opts.hasBias){
      updateBias(deriv, update_bias_mat);
    }
    
    updateFFilter.convolveMfork(inputData, deriv, false, bwdFilterWS);
    backwardfiltertime += toc - start;
    
    if (inputDeriv.asInstanceOf[AnyRef] != null) {  
    	val workspace = if (Net.getPlacing(opts.inplace, net.opts.inplace) == Net.InPlace) output else null;
      ffilter.convolveT(deriv, inputDeriv, false, workspace);
    } 
    backwarddatatime += toc - start;
    
    updateFFilter.convolveMjoin;

    inplaceNoConnectReleaseDeriv()
    backwardtime += toc - start;
  }
  
  override def clear = {
    clearMats;
    filter = null; 
    ffilter= null;
    updateFilter = null;
    updateFFilter = null;
    bias_mat = null; 
    update_bias_mat = null;
    inputDim = null; 
    cudnnMainHandle= null;
    cudnnMainStream = null;
  }

    
  def applyBias(bias:Mat, output:Mat) = {
    (bias, output) match {
      case (gbias:GMat, goutput:GMat) => applyBiasGMat(gbias, goutput);
      case (fbias:FMat, foutput:FMat) => applyBiasFMat(fbias, foutput);
      case _ => throw new RuntimeException("ConvLayer applyBias matrix type not matched");
    }
  }
        
  def applyBiasFMat(bias:FMat, output:FMat) = {
  	val dims = output.dims;
  	val n = dims(3);
  	val h = dims(2);
  	val w = dims(1);
  	val c = dims(0);
  	if (getTensorFormat == Net.TensorNCHW) {
  		var hw = h*w;
  		var i = 0;
  		var iptr = 0;
  		while (i < n) {
  			var j = 0;
  			while (j < c) {
  				val vout = bias.data(j);
  				var k = 0;
  				while (k < hw) {
  					output.data(iptr) += vout;
  					iptr += 1;
  					k += 1;
  				}
  				j += 1;
  			}
  			i += 1;
  		}
  	} else {
  		var hw = h*w;
  		var i = 0;
  		var iptr = 0;
  		while (i < n) {
  			var j = 0;
  			while (j < hw) {
  				var k = 0;
  				while (k < c) {
  					output.data(iptr) += bias.data(k);
  					iptr += 1;
  					k += 1;
  				}
  				j += 1;
  			}
  			i += 1;
  		}        
  	}
  }
  
  def applyBiasGMat(bias:GMat, output:GMat) = {
  	val dims = output.dims;
  	val blen = bias.length;
  	var dataType = cudnnDataType.CUDNN_DATA_FLOAT;
  	val tformat = Net.getCUDNNformat(opts.tensorFormat, net.opts.tensorFormat);
  	
  	val adesc = new cudnnTensorDescriptor;
  	cudnnCreateTensorDescriptor(adesc);
  	val astatus = cudnnSetTensor4dDescriptor(adesc, tformat, dataType, 1, blen, 1, 1);
  	if (astatus > 0) throw new RuntimeException("Error %d creating A tensor for forward bias computation" format astatus);

  	val bdesc = new cudnnTensorDescriptor;
  	cudnnCreateTensorDescriptor(bdesc);
  	val bstatus = cudnnSetTensor4dDescriptor(bdesc, tformat, dataType, dims(3), dims(0), dims(2), dims(1));
  	if (bstatus > 0) throw new RuntimeException("Error %d creating B tensor for forward bias computation" format bstatus);
	  
  	var err = cudnnAddTensor(cudnnMainHandle, GFilter.ONE, adesc, bias.pdata, GFilter.ONE, bdesc, output.pdata);
  	
  	cudaStreamSynchronize(cudnnMainStream);
  	if (err == 0) err = cudaGetLastError();
  	if (err > 0) throw new RuntimeException("Error in forward convolution bias: %s" format cudaGetErrorString(err));
  	
  	cudnnDestroyTensorDescriptor(bdesc);
  	cudnnDestroyTensorDescriptor(adesc);
  }
  
  def updateBias(deriv:Mat, updateBias:Mat) = {
     (deriv, updateBias) match {
      case (gderiv:GMat, gupdateBias:GMat) => updateBiasGMat(gderiv, gupdateBias);
      case (fderiv:FMat, fupdateBias:FMat) => updateBiasFMat(fderiv, fupdateBias);
      case _ => throw new RuntimeException("ConvLayer updateBias matrix type not matched");
     } 	
  }
    
  def updateBiasFMat(deriv:FMat, updateBias:FMat) = {   
  	val dims = deriv.dims;
  	val n = dims(3);
  	val h = dims(2);
  	val w = dims(1);
  	val c = dims(0);
  	if (getTensorFormat == Net.TensorNCHW) {
  		var hw = h*w;
  		var i = 0;
  		var iptr = 0;
  		while (i < n) {
  			var j = 0;
  			while (j < c) {
  				var vsum = 0f;
  				var k = 0;
  				while (k < hw) {
  					vsum += deriv.data(iptr);
  					iptr += 1;
  					k += 1;
  				}
  				updateBias.data(j) += vsum;
  				j += 1;
  			}
  			i += 1;
  		}
  	} else {
  		var hw = h*w;
  		var i = 0;
  		var iptr = 0;
  		while (i < n) {
  			var j = 0;
  			while (j < hw) {
  				var k = 0;
  				while (k < c) {
  					updateBias.data(k) += deriv.data(iptr);
  					iptr += 1;
  					k += 1;
  				}
  				j += 1;
  			}
  			i += 1;
  		}        
  	}      
  }
  
  def updateBiasGMat(deriv:GMat, updateBias:GMat) = {   
  	val dims = deriv.dims;
  	val blen = updateBias.length
  	var dataType = cudnnDataType.CUDNN_DATA_FLOAT;
  	val tformat = Net.getCUDNNformat(opts.tensorFormat, net.opts.tensorFormat);
  	
  	val adesc = new cudnnTensorDescriptor;
  	cudnnCreateTensorDescriptor(adesc);
  	val astatus = cudnnSetTensor4dDescriptor(adesc, tformat, dataType, dims(3), dims(0), dims(2), dims(1));
  	if (astatus > 0) throw new RuntimeException("Error %d creating A tensor for backward bias computation" format astatus);

  	val bdesc = new cudnnTensorDescriptor;
  	cudnnCreateTensorDescriptor(bdesc);
  	val bstatus = cudnnSetTensor4dDescriptor(bdesc, tformat, dataType, 1, blen, 1, 1);
  	if (bstatus > 0) throw new RuntimeException("Error %d creating B tensor for backward bias computation" format bstatus);
	  
  	var err = cudnnConvolutionBackwardBias(cudnnMainHandle, GFilter.ONE, adesc, deriv.pdata, GFilter.ONE, bdesc, updateBias.pdata);

  	cudaStreamSynchronize(cudnnMainStream);
  	if (err == 0) err = cudaGetLastError();
  	if (err > 0) throw new RuntimeException("Error in backward convolution bias: %s" format cudaGetErrorString(err));
  	
  	cudnnDestroyTensorDescriptor(bdesc);
  	cudnnDestroyTensorDescriptor(adesc);
  }
  

  override def toString = {
    "conv@" + Integer.toHexString(hashCode() % 0x10000)
  }
  
}

trait ConvNodeOpts extends ModelNodeOpts {
  var noutputs:Int = 0
  var hasBias:Boolean = true
  var pad:IMat = null
  var kernel:IMat = null
  var stride:IMat = null
  var dilation:IMat = null //was dilation:List[Integer] = Arrays.asList(1)
  var convType:Int = Net.UseNetConvType;
  var initbiasfn:(Mat,Float)=>Mat = Net.constant;
  var initbiasv:Float = 0f;

  def copyOpts(opts:ConvNodeOpts):ConvNodeOpts = {
  		super.copyOpts(opts);
  		opts.noutputs = noutputs;
  		opts.hasBias = hasBias;
  		opts.pad = pad;
  		opts.kernel = kernel;
  		opts.stride = stride;
  		opts.dilation = dilation;
  		opts.convType = convType;
  		opts.initfn = initfn;
  		opts.initv = initv;
  		opts.initbiasfn = initbiasfn;
  		opts.initbiasv = initbiasv;
  		opts;
  }

}

@SerialVersionUID(100L)
class ConvNode extends ModelNode with ConvNodeOpts {

  def copyTo(opts:ConvNode):ConvNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }

  override def clone:ConvNode = {
    copyTo(new ConvNode ).asInstanceOf[ConvNode]
  }
  
  override def create(net:Net):ConvLayer = {
    ConvLayer(net, this)
  }

  override def toString = {
    "conv@" + Integer.toHexString(hashCode() % 0x10000)
  }


}

@SerialVersionUID(100L)
object ConvLayer {
  
  def apply(net:Net) = new ConvLayer(net, new ConvNode);
  
  def apply(net:Net, opts:ConvNodeOpts) = new ConvLayer(net, opts);
  
  def fieldArray(a:FMat, m:Int, n:Int, scale:Float = 100f, border:Int=1, isNCHW:Boolean = true):FMat = {
    val c = a.dims(0);
    val w = a.dims(1);
    val h = a.dims(2);
    val nn = a.dims(3);
    val aa = if (isNCHW) {
      FMat(a).fromNCHWtoNHWC;
    } else {
      FMat(a);
    }
    val bb = aa.reshapeView(c,w,h*n,m).transpose(0\1\3\2).reshapeView(c,w*m,h*n,1) * scale + 128;
    val hinds = (irow(0->(w*m)).reshapeView(w,m) + irow(0->m)*border)(?);
    val vinds = (irow(0->(h*n)).reshapeView(h,n) + irow(0->n)*border)(?);
    val out = zeros(c\((w+border)*m)\((h+border)*n)\1);
    out(?,hinds,vinds,?) = bb;
    out;
  }

}
