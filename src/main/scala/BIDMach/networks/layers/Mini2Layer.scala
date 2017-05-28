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
 * Mini2 layer. 
 */

class Mini2Layer(override val net:Net, override val opts:Mini2NodeOpts = new Mini2Node) extends Layer(net, opts) {

  override val _outputs = new Array[Mat](2);
  override val _derivs = new Array[Mat](2);
	
	override def forward = {
			val start = toc;
			val (mm, ii) = mini2(inputData);
			output = mm;
			setOutput(1, ii);
			clearDeriv;
			forwardtime += toc - start;
	}

	override def backward = {
			val start = toc;
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv ∘ (inputData == mini(inputData)));  
			backwardtime += toc - start;
	}
  
  override def toString = {
    "mini2@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


trait Mini2NodeOpts extends NodeOpts {  
}

class Mini2Node extends Node with Mini2NodeOpts {

	override def clone:Mini2Node = {copyTo(new Mini2Node).asInstanceOf[Mini2Node];}

  override def create(net:Net):Mini2Layer = {Mini2Layer(net, this);}
  
  override def toString = {
    "mini2@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object Mini2Layer {  
  
  def apply(net:Net) = new Mini2Layer(net, new Mini2Node);
  
  def apply(net:Net, opts:Mini2Node) = new Mini2Layer(net, opts);
}