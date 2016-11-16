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
 * Rectifying Linear Unit layer.
 */

class RectLayer(override val net:Net, override val opts:RectNodeOpts = new RectNode) extends Layer(net, opts) {
	override def forward = {
      val start = toc;
			createOutput;
			output.asMat <-- max(inputData.asMat, 0f);
			clearDeriv;
			forwardtime += toc - start;
	}

	override def backward = {
			val start = toc;
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv ∘ (inputData > 0f));
			backwardtime += toc - start;
	}
  
  override def toString = {
    "rect@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait RectNodeOpts extends NodeOpts {
}
    
class RectNode extends Node with RectNodeOpts {
  def copyTo(opts:RectNode):RectNode = {
    super.copyTo(opts);
    opts;
  }
    
  override def clone:RectNode = {
    copyTo(new RectNode);
  }
  
  override def create(net:Net):RectLayer = {
  	RectLayer(net, this);
  }
  
  override def toString = {
    "rect@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object RectLayer {  
  
  def apply(net:Net) = new RectLayer(net, new RectNode);
  
  def apply(net:Net, opts:RectNodeOpts) = new RectLayer(net, opts);
}
