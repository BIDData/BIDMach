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
 * Sigmoid layer.  
 */

class SigmoidLayer(override val net:Net, override val opts:SigmoidNodeOpts = new SigmoidNode) extends Layer(net, opts) {

  override def forward = {
		val start = toc;
		inplaceNoConnectGetOutput();
		
    LayerFn.applyfwd(inputData, output, LayerFn.SIGMOIDFN);

		forwardtime += toc - start;
}

  override def backward = {
		val start = toc;
		inplaceNoConnectGetInputDerivs();
		
		if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + LayerFn.applyderiv(output, deriv, LayerFn.SIGMOIDFN);
		
		inplaceNoConnectReleaseDeriv()
		backwardtime += toc - start;
  }
  
  override def toString = {
    "sigmoid@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


trait SigmoidNodeOpts extends NodeOpts {  
}

class SigmoidNode extends Node with SigmoidNodeOpts {

	override def clone:SigmoidNode = {copyTo(new SigmoidNode).asInstanceOf[SigmoidNode];}

  override def create(net:Net):SigmoidLayer = {SigmoidLayer(net, this);}
  
  override def toString = {
    "sigmoid@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object SigmoidLayer {  
  
  def apply(net:Net) = new SigmoidLayer(net, new SigmoidNode);
  
  def apply(net:Net, opts:SigmoidNode) = new SigmoidLayer(net, opts);
}
