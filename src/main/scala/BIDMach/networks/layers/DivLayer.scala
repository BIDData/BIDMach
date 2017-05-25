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
 * Computes the quotien of its input layers. 
 */

class DivLayer(override val net:Net, override val opts:DivNodeOpts = new DivNode) extends Layer(net, opts) {  
  
	override val _inputs = new Array[LayerTerm](2);

	override def forward = {
    val start = toc;
	  createOutput(inputData.dims);
	  output <-- inputData;
	  output ~ output / inputDatas(1);
	  clearDeriv;
	  forwardtime += toc - start;
	}
	
	override def backward = {
    val start = toc;

    if (inputDerivs(0).asInstanceOf[AnyRef] != null) inputDerivs(0) ~ inputDerivs(0) + squash(deriv / inputDatas(1), inputDerivs(0));
    if (inputDerivs(1).asInstanceOf[AnyRef] != null) inputDerivs(1) ~ inputDerivs(1) - squash((deriv / inputDatas(1)) ∘ (inputDatas(0) / inputDatas(1)), inputDerivs(1));

    backwardtime += toc - start;
	}
  
  override def toString = {
    "div@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait DivNodeOpts extends NodeOpts {  
}

class DivNode extends Node with DivNodeOpts {
  override val inputs:Array[NodeTerm] = new Array[NodeTerm](2);
  
  def copyTo(opts:DivNode):DivNode = {
      super.copyTo(opts);
      opts;
  }

	override def clone:DivNode = {copyTo(new DivNode).asInstanceOf[DivNode];}

	override def create(net:Net):DivLayer = {DivLayer(net, this);}
  
  override def toString = {
    "div@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}
  
object DivLayer {  
  
  def apply(net:Net) = new DivLayer(net, new DivNode);
  
  def apply(net:Net, opts:DivNodeOpts) = new DivLayer(net, opts); 
}
