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


class LayerNormLayer(override val net:Net, override val opts:LayerNormNodeOpts = new LayerNormNode) extends Layer(net, opts) {
   
  var means:Mat = null;
  var variances:Mat = null;
  var sdevs:Mat = null;
  var batchDim:IMat = null;
  
  var debugMe = false;

  def initModelMats = {
    batchDim = irow(0->(inputData.dims.length-1));
  }

  override def forward = {
    val start = toc;
    inplaceNoConnectGetOutput(true);

    if (batchDim.asInstanceOf[AnyRef] == null) initModelMats;
    
    forwardGeneric;

    forwardtime += toc - start;
  }
  
  override def backward = {
    val start = toc;
    inplaceNoConnectGetInputDerivs();
    
    backwardGeneric
        
    inplaceNoConnectReleaseDeriv()
    backwardtime += toc - start;
  }
  
    
  def forwardGeneric = {
    // Do LayerNorm
    means = mean(inputData)
    variances = variance(inputData) + opts.epsilon;
    sdevs = sqrt(variances);
    output ~ inputData - means;
    output ~ output / sdevs;                       // Works even if output = input;
  }
  
  def backwardGeneric = {
    inputDeriv ~ inputDeriv + (deriv / sdevs);
  }
  
  override def clear = {
    clearMats;  
    means = null;
    variances = null;
    sdevs = null;
    batchDim = null;
  }

  override def toString = {
    "ln@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

trait LayerNormNodeOpts extends NodeOpts {
  var epsilon:Float = 1e-4f;

  def copyOpts(opts:LayerNormNodeOpts):LayerNormNodeOpts = {
		super.copyOpts(opts);
		opts.epsilon = epsilon;
		opts.inplace = inplace;
		opts;
  }
}

class LayerNormNode extends Node with LayerNormNodeOpts {
  
  def copyTo(opts:LayerNormNode):LayerNormNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }
  override def clone:LayerNormNode = copyTo(new LayerNormNode).asInstanceOf[LayerNormNode]

  override def create(net:Net) = LayerNormLayer(net, this)
  
  override def toString = {
    "ln@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

@SerialVersionUID(100L)
object LayerNormLayer {
  
  val ONE = Pointer.to(Array(1.0f));
  val ZERO = Pointer.to(Array(0.0f));

  def apply(net:Net) = new LayerNormLayer(net);
  
  def apply(net:Net, opts:LayerNormNodeOpts) = new LayerNormLayer(net, opts);

}

