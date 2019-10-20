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
import BIDMach.updaters._
import BIDMach.networks._


class AutoNormLayer(override val net:Net, override val opts:AutoNormNodeOpts = new AutoNormNode) extends ModelLayer(net, opts, 2) {
   
  var means:Mat = null;
  var variances:Mat = null;
  var sdevs:Mat = null;
  var moving_means:Mat = null;
  var moving_sdevs:Mat = null;
  var batchDim:IMat = null;
  var batchDim1:IMat = null;
  
  var debugMe = false;

  def initModelMats = {
    if (opts.dims.asInstanceOf[AnyRef] != null) { 
      batchDim = opts.dims;
    } else { 
      batchDim = irow(0->(inputData.dims.length));
    }
    val d = inputData.dims
    if (d.length > 2) { 
      if (d(0) > 128) { 
        batchDim1 = irow(0);
      } else if (d.length > 3 && prod(d(0->(d.length-2))).v > 128) { 
        batchDim1 = irow(0->(inputData.dims.length-2));
      }
    }
    if (modelmats(imodel) == null) {
      val mdim = iones(1, inputData.dims.length);
      modelmats(imodel) = convertMat(ones(mdim));
      modelmats(imodel+1) = convertMat(zeros(mdim));
    }
  }

  override def forward = {
    val start = toc;
    inplaceNoConnectGetOutput(true);

    if (batchDim.asInstanceOf[AnyRef] == null) initModelMats;

    moving_sdevs = modelmats(imodel);
    moving_means = modelmats(imodel+1);
    
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
    // Do AutoNorm
    if (batchDim1.asInstanceOf[AnyRef] != null) { 
      val means1 = inputData.mean(batchDim1);
      means = means1.mean(batchDim);
      val squares1 = (inputData *@ inputData).mean(batchDim1);
	  val squares = squares1.mean(batchDim);
      variances = squares - means *@ means;
    } else { 
      means = inputData.mean(batchDim);
      variances = inputData.variance(batchDim);
    }
    abs(variances, variances);
    variances ~ variances + opts.epsilon;
    sdevs = sqrt(variances);
    output ~ inputData - moving_means;
    output ~ output / moving_sdevs;                       // Works even if output = input;
  }
  
  def backwardGeneric = {
    if (inputDeriv.asInstanceOf[AnyRef] != null ) { 
      val ideriv = deriv / moving_sdevs;
      inputDeriv ~ inputDeriv + ideriv;
    }
    (moving_means, means) match { 
      case (mm:GMat, m:GMat) => { 
        Grad.linComb(mm, opts.decay, m, 1f - opts.decay, mm);
      }
      case _ => moving_means ~ moving_means + ((means - moving_means) * (1f - opts.decay))
    }
    (moving_sdevs, sdevs) match { 
      case (ms:GMat, s:GMat) => { 
        Grad.linComb(ms, opts.decay, s, 1f - opts.decay, ms);
      }
      case _ => moving_sdevs ~ moving_sdevs + ((sdevs - moving_sdevs) * (1f - opts.decay))
    }
  }
  
  override def clear = {
    clearMats;  
    means = null;
    variances = null;
    sdevs = null;
    batchDim = null;
  }

  override def toString = {
    "anorm@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

trait AutoNormNodeOpts extends ModelNodeOpts {
  var epsilon:Float = 1e-4f;
  var decay:Float = 1f
  var dims:IMat = null;

  def copyOpts(opts:AutoNormNodeOpts):AutoNormNodeOpts = {
	super.copyOpts(opts);
	opts.epsilon = epsilon;
	opts.decay = decay;
	opts;
  }
}

class AutoNormNode extends Node with AutoNormNodeOpts {
  
  def copyTo(opts:AutoNormNode):AutoNormNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }
  override def clone:AutoNormNode = copyTo(new AutoNormNode).asInstanceOf[AutoNormNode]

  override def create(net:Net) = AutoNormLayer(net, this)
  
  override def toString = {
    "anorm@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

@SerialVersionUID(100L)
object AutoNormLayer {
  
  val ONE = Pointer.to(Array(1.0f));
  val ZERO = Pointer.to(Array(0.0f));

  def apply(net:Net) = new AutoNormLayer(net);
  
  def apply(net:Net, opts:AutoNormNodeOpts) = new AutoNormLayer(net, opts);

}

