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
 * Sqrt layer. 
 */

@SerialVersionUID(100L)
class SqrtLayer(override val net:Net, override val opts:SqrtNodeOpts = new SqrtNode) extends Layer(net, opts) {
  
  var half:Mat = null;

	override def forward = {
			val start = toc;
			inplaceNoConnectGetOutput();
			
			if (half.asInstanceOf[AnyRef] == null) half = inputData.ones(1,1) * 0.5f
			sqrt(inputData, output);
	
			forwardtime += toc - start;
	}

	override def backward = {
			val start = toc;
			inplaceNoConnectGetInputDerivs();
			
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv * half / sqrt(inputData) );  
			
			inplaceNoConnectReleaseDeriv()
			backwardtime += toc - start;
	}
  
  override def toString = {
    "sqrt@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


trait SqrtNodeOpts extends NodeOpts {  
}

@SerialVersionUID(100L)
class SqrtNode extends Node with SqrtNodeOpts {

	override def clone:SqrtNode = {copyTo(new SqrtNode).asInstanceOf[SqrtNode];}

  override def create(net:Net):SqrtLayer = {SqrtLayer(net, this);}
  
  override def toString = {
    "sqrt@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

@SerialVersionUID(100L)
object SqrtLayer {  
  
  def apply(net:Net) = new SqrtLayer(net, new SqrtNode);
  
  def apply(net:Net, opts:SqrtNode) = new SqrtLayer(net, opts);
}
