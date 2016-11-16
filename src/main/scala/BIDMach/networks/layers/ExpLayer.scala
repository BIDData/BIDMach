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
 * Exponential layer. 
 */

class ExpLayer(override val net:Net, override val opts:ExpNodeOpts = new ExpNode) extends Layer(net, opts) {

	override def forward = {
			val start = toc;
			createOutput;
			exp(inputData, output);
			clearDeriv;
			forwardtime += toc - start;
	}

	override def backward = {
			val start = toc;
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv ∘ output);  
			backwardtime += toc - start;
	}
  
  override def toString = {
    "exp@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


trait ExpNodeOpts extends NodeOpts {  
}

class ExpNode extends Node with ExpNodeOpts {

	override def clone:ExpNode = {copyTo(new ExpNode).asInstanceOf[ExpNode];}

  override def create(net:Net):ExpLayer = {ExpLayer(net, this);}
  
  override def toString = {
    "exp@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object ExpLayer {  
  
  def apply(net:Net) = new ExpLayer(net, new ExpNode);
  
  def apply(net:Net, opts:ExpNode) = new ExpLayer(net, opts);
}