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
 * Absolute Value layer. 
 */

class AbsLayer(override val net:Net, override val opts:AbsNodeOpts = new AbsNode) extends Layer(net, opts) {

	override def forward = {
			val start = toc;
      inplaceNoConnectGetOutput();
      
			abs(inputData, output);
			clearDeriv;
			forwardtime += toc - start;
	}

	override def backward = {
			val start = toc;
			inplaceNoConnectGetInputDerivs();
			
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv ∘ sign(inputData));
			
			inplaceNoConnectReleaseDeriv()
			backwardtime += toc - start;
	}
  
  override def toString = {
    "abs@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


trait AbsNodeOpts extends NodeOpts {  
}

class AbsNode extends Node with AbsNodeOpts {

	override def clone:AbsNode = {copyTo(new AbsNode).asInstanceOf[AbsNode];}

  override def create(net:Net):AbsLayer = {AbsLayer(net, this);}
  
  override def toString = {
    "abs@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object AbsLayer {  
  
  def apply(net:Net) = new AbsLayer(net, new AbsNode);
  
  def apply(net:Net, opts:AbsNode) = new AbsLayer(net, opts);
}