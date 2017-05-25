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
 * Computes the min of its input layers. 
 */

class MinLayer(override val net:Net, override val opts:MinNodeOpts = new MinNode) extends Layer(net, opts) {  
  
	override val _inputs = new Array[LayerTerm](2);

	override def forward = {
    val start = toc;
	  createOutput(inputData.dims);
	  min(inputData, inputDatas(1), output);
	  clearDeriv;
	  forwardtime += toc - start;
	}
	


	override def backward = {
    val start = toc;

    min(inputData, inputDatas(1), output);
    if (inputDerivs(0).asInstanceOf[AnyRef] != null) inputDerivs(0) ~ inputDerivs(0) + squash((output == inputData) ∘ deriv, inputDerivs(0));
    if (inputDerivs(1).asInstanceOf[AnyRef] != null) inputDerivs(1) ~ inputDerivs(1) + squash((output == inputDatas(1)) ∘ deriv, inputDerivs(1));
    
    backwardtime += toc - start;
	}
  
  override def toString = {
    "min@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait MinNodeOpts extends NodeOpts {  
}

class MinNode extends Node with MinNodeOpts {
  override val inputs:Array[NodeTerm] = new Array[NodeTerm](2);
  
  def copyTo(opts:MinNode):MinNode = {
      super.copyTo(opts);
      opts;
  }

	override def clone:MinNode = {copyTo(new MinNode).asInstanceOf[MinNode];}

	override def create(net:Net):MinLayer = {MinLayer(net, this);}
  
  override def toString = {
    "min@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}
  
object MinLayer {  
  
  def apply(net:Net) = new MinLayer(net, new MinNode);
  
  def apply(net:Net, opts:MinNodeOpts) = new MinLayer(net, opts); 
}
