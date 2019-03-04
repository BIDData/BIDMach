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
 * Natural Log layer. 
 */

@SerialVersionUID(100L)
class LnLayer(override val net:Net, override val opts:LnNodeOpts = new LnNode) extends Layer(net, opts) {

	override def forward = {
			val start = toc;
			inplaceNoConnectGetOutput();
			
			ln(inputData, output);

			forwardtime += toc - start;
	}

	override def backward = {
			val start = toc;
			inplaceNoConnectGetInputDerivs();
			
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv/inputData);  
			
			inplaceNoConnectReleaseDeriv()
			backwardtime += toc - start;
	}
  
  override def toString = {
    "ln@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait LnNodeOpts extends NodeOpts {  
}

@SerialVersionUID(100L)
class LnNode extends Node with LnNodeOpts {

	override def clone:LnNode = {copyTo(new LnNode).asInstanceOf[LnNode];}

  override def create(net:Net):LnLayer = {LnLayer(net, this);}
  
  override def toString = {
    "ln@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

@SerialVersionUID(100L)
object LnLayer {  
  
  def apply(net:Net) = new LnLayer(net, new LnNode);
  
  def apply(net:Net, opts:LnNode) = new LnLayer(net, opts);
}
