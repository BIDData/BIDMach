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
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import scala.util.hashing.MurmurHash3
import java.util.HashMap
import BIDMach.networks._
import java.util.Arrays


class ConvLayer(override val net:Net, override val opts:ConvNodeOpts = new ConvNode ) extends ModelLayer(net, opts, 2) {
    var filter:FMat = null; 
    var ffilter:Filter = null;
    var updateFilter:FMat = null;
    var updateFFilter:Filter = null;
    var bias_mat:FMat = null; // it should be size (channel_out*1*1*1), to better broadcast?
    var update_bias_mat:FMat = null;
    var inputDim:IMat = null; // Should be three numbers
//    var outputDim:IMat = null; //Should be three numbers
    

  def initModelMats = {
    inputDim = inputData.dims;
    val channel_in = inputDim(0);
    val filter_h = opts.kernel(0); // 3;0
    val filter_w = opts.kernel(1); // 3;
    val npad = opts.pad(0); //1;
    val nstride = opts.stride(0); // 1;
    val channel_out = opts.noutputs; // actually # of filters;

    
    if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
    	modelmats(imodel) = if (net.opts.useGPU && Mat.hasCUDA > 0 && Mat.hasCUDNN) {
    		val x = GFilter.GFilter2Ddn(filter_h,filter_w,channel_in,channel_out,nstride,npad); 
    		x.setTensorFormat(Net.getCUDNNformat(opts.tensorFormat, net.opts.tensorFormat));
    		x;
    	} else {
    		FFilter2Ddn(filter_h,filter_w,channel_in,channel_out,nstride,npad);
    	}
    	filter = modelmats(imodel).asInstanceOf[FMat];
    	ffilter = modelmats(imodel).asInstanceOf[Filter];   	
    	ffilter match {
    	  case aa:GFilter => aa.convType = opts.convType;
    	  case _ => {}
    	}
    	updatemats(imodel) = ffilter.copy.asInstanceOf[FMat];
    	
    	opts.initfn(filter, opts.initv);
    	
    	val outDim = Filter.getOutputDims(inputData.dims, ffilter.inDims, ffilter.outDims, ffilter.stride, ffilter.pad, ffilter.outPad);
    	
    	if (opts.hasBias) {
    		val biasDim = irow(outDim(0), outDim(1), outDim(2), 1);
    		modelmats(imodel+1) = modelmats(imodel).zeros(biasDim);
    		updatemats(imodel+1) = modelmats(imodel).zeros(biasDim); 		    	
    		opts.initbiasfn(modelmats(imodel+1), opts.initbiasv);
    	}
    }
    if (lr_scales.asInstanceOf[AnyRef] != null) {
    	lr_scales(imodel) = opts.lr_scale;
    	lr_scales(imodel+1) = opts.bias_scale;
    }
    filter = modelmats(imodel).asInstanceOf[FMat];
    ffilter = modelmats(imodel).asInstanceOf[Filter];
    if (output.asInstanceOf[AnyRef] == null) { 
    	val outputBatchDim = Filter.getOutputDims(inputData.dims, ffilter.inDims, ffilter.outDims, ffilter.stride, ffilter.pad, ffilter.outPad);
    	output = filter.zeros(outputBatchDim);
    };
    
    updateFilter = updatemats(imodel).asInstanceOf[FMat];
    updateFFilter = updatemats(imodel).asInstanceOf[Filter];  
    if (updateFilter.asInstanceOf[AnyRef] != null) updateFilter.clear;
    
    bias_mat = modelmats(imodel+1).asInstanceOf[FMat];
    update_bias_mat = updatemats(imodel+1).asInstanceOf[FMat];
    if (update_bias_mat.asInstanceOf[AnyRef] != null) update_bias_mat.clear;
  }

  override def forward = {
    val start = toc;
    
    // Create filter model, filter update and bias model if needed
    if (inputDim.asInstanceOf[AnyRef] == null) initModelMats;
 
    
    ffilter.convolve(inputData, output, true);
    if (opts.hasBias) output ~ output + bias_mat;

    clearDeriv
    forwardtime += toc - start
  }

  override def backward = {
    val start = toc;
    val ndims = output.dims.length;
    
    if(opts.hasBias){
      update_bias_mat ~ update_bias_mat + deriv.sum(irow(ndims - 1));
    }

    if (inputDeriv.asInstanceOf[AnyRef] != null) {      
        ffilter.convolveT(deriv, inputDeriv, false)
    }

    updateFFilter.convolveM(inputData, deriv, false)

    backwardtime += toc - start;
  }
  
  override def clear = {
    clearMats;
    filter = null; 
    ffilter= null;
    updateFilter = null;
    updateFFilter = null;
    bias_mat = null; // it should be size (channel_out*1*1*1), to better broadcast?
    update_bias_mat = null;
    inputDim = null; 
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
  var tensorFormat:Int = Net.UseNetFormat;
  var convType:Int = cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
  var initfn:(Mat,Float)=>Mat = Net.xavier;
  var initv:Float = 1f;
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
  		opts.tensorFormat = tensorFormat;
  		opts.convType = convType;
  		opts.initfn = initfn;
  		opts.initv = initv;
  		opts.initbiasfn = initbiasfn;
  		opts.initbiasv = initbiasv;
  		opts;
  }

}

class ConvNode extends Node with ConvNodeOpts {

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
    val hinds = (irow(0->(h*n)).reshapeView(h,n) + irow(0->n)*border)(?);
    val vinds = (irow(0->(w*m)).reshapeView(w,m) + irow(0->m)*border)(?);
    val out = zeros(c\((w+border)*m)\((h+border)*n)\1);
    out(?,hinds,vinds,?) = bb;
    out;
  }

}
