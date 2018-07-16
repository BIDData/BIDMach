package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
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
 * Variable layer. 
 */


class VariableLayer(override val net:Net, override val opts:VariableNodeOpts = new VariableNode) extends ModelLayer(net, opts, 1) {
  
  def initModelMat(dims:IMat):Mat = {
		  if (lr_scales.asInstanceOf[AnyRef] != null) {
			  lr_scales(imodel) = opts.lr_scale;
		  }
		  zeros(dims)
  }

  override def forward = {
		  val start = toc;
		  if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
			  modelmats(imodel) = convertMat(initModelMat(opts.dims));
			  updatemats(imodel) = modelmats(imodel).zeros(opts.dims);
			  opts.initfn(modelmats(imodel), opts.initv);
		  }
		  val mm = modelmats(imodel);
		  createOutput(mm.dims);
		  inplaceNoConnectGetOutput(true);

		  output <-- modelmats(imodel+1);
		  forwardtime += toc - start;
  }

  override def backward = {
    val start = toc;
    inplaceNoConnectGetInputDerivs();
    
    val mm = modelmats(imodel);

    val um = updatemats(imodel);
    deriv.madd(inputData, um, false, true);

    inplaceNoConnectReleaseDeriv();
    backwardtime += toc - start;
  }
  
  override def clear = {
  		clearMats;
  }

  override def toString = {
    "var@"+Integer.toHexString(hashCode % 0x10000).toString
  }

}

trait VariableNodeOpts extends ModelNodeOpts {
  var dims:IMat = irow(1,1)
  
  def copyOpts(opts:VariableNodeOpts):VariableNodeOpts = {
    super.copyOpts(opts);
    opts.dims = dims;
    opts;
  }
}
    
class VariableNode extends Node with VariableNodeOpts {
  
  def copyTo(opts:VariableNode):VariableNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }
    
  override def clone:VariableNode = {
    copyTo(new VariableNode).asInstanceOf[VariableNode];
  }
  
  override def toString = {
    "var@"+Integer.toHexString(hashCode % 0x10000).toString
  }

  override def create(net:Net):VariableLayer = {
  	VariableLayer(net, this);
  }
}


object VariableLayer {

  def apply(net:Net) = new VariableLayer(net, new VariableNode);

  def apply(net:Net, opts:VariableNodeOpts):VariableLayer = new VariableLayer(net, opts);

}

