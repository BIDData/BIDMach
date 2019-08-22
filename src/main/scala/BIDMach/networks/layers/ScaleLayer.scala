
package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat,TMat}
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
import scala.util.hashing.MurmurHash3;
import java.util.HashMap;
import BIDMach.networks._ 

/**
 * Scale layer. 
 * Includes model matrices for the scale and bias
 */

@SerialVersionUID(100L)
class ScaleLayer(override val net:Net, override val opts:ScaleNodeOpts = new ScaleNode) extends ModelLayer(net, opts, 2) {
  
  var scaleMat:Mat = null;
  var biasMat:Mat = null;
  var updateScaleMat:Mat = null;
  var updateBiasMat:Mat = null;
  var initDone = false;
  
  def initModelMats = {
    val bdims = inputData.dims.copy;
    bdims(opts.modelDims) = 1
    if (modelmats(imodel) == null) {
    	modelmats(imodel) = convertMat(ones(bdims));
    	modelmats(imodel+1) = convertMat(zeros(bdims));
    	updatemats(imodel) = convertMat(zeros(bdims));
    	updatemats(imodel+1) = convertMat(zeros(bdims));
    }
    scaleMat = modelmats(imodel);
    biasMat = modelmats(imodel+1);
    updateScaleMat = updatemats(imodel);
    updateBiasMat = updatemats(imodel+1);
    initDone = true;
  }

  override def forward = {
    val start = toc;
    if (!initDone) initModelMats;
    inplaceNoConnectGetOutput();
    
    output ~ scaleMat *@ inputData
    if (opts.hasBias) output ~ output + biasMat;
    
    forwardtime += toc - start;
  }

  override def backward(ipass:Int, pos:Long) = {
    val start = toc;
    inplaceNoConnectGetInputDerivs();
    
    inputDeriv ~ inputDeriv + (scaleMat *@ deriv);
    updateScaleMat ~ updateScaleMat + (inputData *@ deriv).sum(opts.modelDims);
    
    if (opts.hasBias) updateBiasMat ~ updateBiasMat + deriv.sum(opts.modelDims);

    inplaceNoConnectReleaseDeriv()
    backwardtime += toc - start;
  }
  
  override def clear = {
    clearMats;
  	scaleMat = null;
  	biasMat = null;
  	updateScaleMat = null;
  	updateBiasMat = null;
  }
  
  override def toString = {
    "scale@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


trait ScaleNodeOpts extends ModelNodeOpts {
  var hasBias:Boolean = true;
  var modelDims = irow(0);
  weight_decay_scale = 0f;
   
  def copyOpts(opts:ScaleNodeOpts):ScaleNodeOpts = {
  		super.copyOpts(opts);
  		opts.hasBias = hasBias;
  		opts.modelDims = modelDims;
   		opts;
  }
}
    
@SerialVersionUID(100L)
class ScaleNode extends ModelNode with ScaleNodeOpts {
  def copyTo(opts:ScaleNode):ScaleNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }
  
  override def toString = {
    "scale@"+Integer.toHexString(hashCode % 0x10000).toString
  }
    
  override def clone:ScaleNode = {
    copyTo(new ScaleNode).asInstanceOf[ScaleNode];
  }
  
  override def create(net:Net):ScaleLayer = {
  	ScaleLayer(net, this);
  }
}

@SerialVersionUID(100L)
object ScaleLayer {  

  def apply(net:Net) = new ScaleLayer(net, new ScaleNode);
  
  def apply(net:Net, opts:ScaleNodeOpts):ScaleLayer = new ScaleLayer(net, opts);
  
}
