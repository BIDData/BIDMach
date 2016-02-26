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
 * Tanh layer. 
 */

class TanhLayer(override val net:Net, override val opts:TanhNodeOpts = new TanhNode) extends Layer(net, opts) {    

	override def forward = {
			val start = toc;
			createOutput;
			tanh(inputData, output);
			clearDeriv;
			forwardtime += toc - start;
	}

	override def backward = {
			val start = toc;
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + LayerFn.applyderiv(output, deriv, LayerFn.TANHFN);
			backwardtime += toc - start;
	}
  
  override def toString = {
    "tanh@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait TanhNodeOpts extends NodeOpts {  
}

class TanhNode extends Node with TanhNodeOpts {

	override def clone:TanhNode = {copyTo(new TanhNode).asInstanceOf[TanhNode];}

  override def create(net:Net):TanhLayer = {TanhLayer(net, this);}
  
  override def toString = {
    "tanh@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object TanhLayer {  
  
  def apply(net:Net) = new TanhLayer(net, new TanhNode);
  
  def apply(net:Net, opts:TanhNode) = new TanhLayer(net, opts);
}
