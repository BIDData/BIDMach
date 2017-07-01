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
 * Mini layer. 
 */

class MiniLayer(override val net:Net, override val opts:MiniNodeOpts = new MiniNode) extends Layer(net, opts) {

	override def forward = {
			val start = toc;
		  createOutput(1 \ inputData.ncols);
		  inplaceNoConnectGetOutput();
		  
			output <-- mini(inputData);

			forwardtime += toc - start;
	}

	override def backward = {
			val start = toc;
			inplaceNoConnectGetInputDerivs();
			
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv ∘ (inputData == mini(inputData)));  
			
			inplaceNoConnectReleaseDeriv()
			backwardtime += toc - start;
	}
  
  override def toString = {
    "mini@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


trait MiniNodeOpts extends NodeOpts {  
}

class MiniNode extends Node with MiniNodeOpts {

	override def clone:MiniNode = {copyTo(new MiniNode).asInstanceOf[MiniNode];}

  override def create(net:Net):MiniLayer = {MiniLayer(net, this);}
  
  override def toString = {
    "mini@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object MiniLayer {  
  
  def apply(net:Net) = new MiniLayer(net, new MiniNode);
  
  def apply(net:Net, opts:MiniNode) = new MiniLayer(net, opts);
}