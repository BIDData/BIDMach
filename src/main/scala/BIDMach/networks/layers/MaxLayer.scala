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
 * Computes the max of its input layers. 
 */

@SerialVersionUID(100L)
class MaxLayer(override val net:Net, override val opts:MaxNodeOpts = new MaxNode) extends Layer(net, opts) {  
  
	override val _inputs = new Array[LayerTerm](opts.ninputs);

	override def forward = {
    val start = toc;
	  inplaceNoConnectGetOutput();
	  
	  max(inputData, inputDatas(1), output);
	  (2 until inputlength).map((i:Int) => max(output, inputDatas(i), output));

	  forwardtime += toc - start;
	}

	override def backward = {
    val start = toc;
    inplaceNoConnectGetInputDerivs();

    max(inputData, inputDatas(1), output);
    if (inputDerivs(0).asInstanceOf[AnyRef] != null) inputDerivs(0) ~ inputDerivs(0) + squash((output == inputData) ∘ deriv, inputDerivs(0));
    if (inputDerivs(1).asInstanceOf[AnyRef] != null) inputDerivs(1) ~ inputDerivs(1) + squash((output == inputDatas(1)) ∘ deriv, inputDerivs(1));

    (2 until inputlength).map((i:Int) => {
    	if (inputDerivs(i).asInstanceOf[AnyRef] != null) inputDerivs(i) ~ inputDerivs(i) + squash((output == inputDatas(i)) ∘ deriv, inputDerivs(i));
    });
    
    inplaceNoConnectReleaseDeriv()
    backwardtime += toc - start;
	}
  
  override def toString = {
    "max@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait MaxNodeOpts extends NodeOpts {  
	var ninputs = 2;
}

@SerialVersionUID(100L)
class MaxNode extends Node with MaxNodeOpts {
  override val inputs:Array[NodeTerm] = new Array[NodeTerm](ninputs);
  
  def copyTo(opts:MaxNode):MaxNode = {
      super.copyTo(opts);
      opts;
  }

	override def clone:MaxNode = {copyTo(new MaxNode).asInstanceOf[MaxNode];}

	override def create(net:Net):MaxLayer = {MaxLayer(net, this);}
  
  override def toString = {
    "max@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}
  
@SerialVersionUID(100L)
object MaxLayer {  
  
  def apply(net:Net) = new MaxLayer(net, new MaxNode);
  
  def apply(net:Net, opts:MaxNodeOpts) = new MaxLayer(net, opts); 
}
