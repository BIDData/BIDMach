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
import scala.util.hashing.MurmurHash3;
import java.util.HashMap;
import BIDMach.networks._ 

/**
 * Linear layer. 
 * Includes a model matrix that contains the linear map. 
 */

class ScaleLayer(override val net:Net, override val opts:ScaleNodeOpts = new ScaleNode) extends ModelLayer(net, opts, 2) {
  
	var scaleMat:Mat = null;
  var biasMat:Mat = null;
  var updateScaleMat:Mat = null;
  var updateBiasMat:Mat = null;
  var batchDim:IMat = null;
  
  def initModelMats = {
    val dims = inputData.dims.copy;
    dims(dims.length-1) = 1;
    batchDim = irow(dims.length-1);
    scaleMat = convertMat(ones(dims));
    biasMat = convertMat(zeros(dims));
    updateScaleMat = convertMat(zeros(dims));
    updateBiasMat = convertMat(zeros(dims));
    modelmats(imodel) = scaleMat;
    modelmats(imodel+1) = biasMat;
    updatemats(imodel) = updateScaleMat;
    updatemats(imodel+1) = updateBiasMat;
  }

  override def forward = {
    val start = toc;
    if (scaleMat.asInstanceOf[AnyRef] == null) initModelMats;
    createOutput;
    
    output ~ scaleMat *@ inputData
    if (opts.hasBias) output ~ output + biasMat;
    
    clearDeriv;
    forwardtime += toc - start;
  }

  override def backward(ipass:Int, pos:Long) = {
    val start = toc;
    
    inputDeriv ~ inputDeriv + (scaleMat *@ deriv);
    updateScaleMat ~ updateScaleMat + (inputData dotr deriv);
    
    if (opts.hasBias) updateBiasMat ~ updateBiasMat + deriv.sum(batchDim);

    backwardtime += toc - start;
  }
  
  override def toString = {
    "scale@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


trait ScaleNodeOpts extends ModelNodeOpts {
	var hasBias:Boolean = true;
  
  def copyOpts(opts:ScaleNodeOpts):ScaleNodeOpts = {
  		super.copyOpts(opts);
  		opts.hasBias = hasBias;
  		opts;
  }
}
    
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

object ScaleLayer {  

  def apply(net:Net) = new ScaleLayer(net, new ScaleNode);
  
  def apply(net:Net, opts:ScaleNodeOpts):ScaleLayer = new ScaleLayer(net, opts);
  
}