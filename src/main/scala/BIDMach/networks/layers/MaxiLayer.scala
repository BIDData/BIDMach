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
 * Maxi layer. 
 */

@SerialVersionUID(100L)
class MaxiLayer(override val net:Net, override val opts:MaxiNodeOpts = new MaxiNode) extends Layer(net, opts) {

	override def forward = {
			val start = toc;
		  createOutput(1 \ inputData.ncols);
		  inplaceNoConnectGetOutput();
		  
			output <-- maxi(inputData);
;
			forwardtime += toc - start;
	}

	override def backward = {
			val start = toc;
			inplaceNoConnectGetInputDerivs();
			
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv ∘ (inputData == maxi(inputData)));  
			
			inplaceNoConnectReleaseDeriv()
			backwardtime += toc - start;
	}
  
  override def toString = {
    "maxi@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


trait MaxiNodeOpts extends NodeOpts {  
}

@SerialVersionUID(100L)
class MaxiNode extends Node with MaxiNodeOpts {

  override def clone:MaxiNode = {copyTo(new MaxiNode).asInstanceOf[MaxiNode];}

  override def create(net:Net):MaxiLayer = {MaxiLayer(net, this);}
  
  override def toString = {
    "maxi@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

@SerialVersionUID(100L)
object MaxiLayer {  
  
  def apply(net:Net) = new MaxiLayer(net, new MaxiNode);
  
  def apply(net:Net, opts:MaxiNode) = new MaxiLayer(net, opts);
}
