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
 * Maxi2 layer. 
 */

class Maxi2Layer(override val net:Net, override val opts:Maxi2NodeOpts = new Maxi2Node) extends Layer(net, opts) {

  override val _outputs = new Array[Mat](2);
	override val _derivs = new Array[Mat](2);
	
	override def forward = {
			val start = toc;
			val (mm, ii) = maxi2(inputData);
			output = mm;
			setOutput(1, ii);
			clearDeriv;
			forwardtime += toc - start;
	}

	override def backward = {
			val start = toc;		
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv ∘ (inputData == maxi(inputData)));  
			backwardtime += toc - start;
	}
  
  override def toString = {
    "maxi2@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


trait Maxi2NodeOpts extends NodeOpts {  
}

class Maxi2Node extends Node with Maxi2NodeOpts {

	override def clone:Maxi2Node = {copyTo(new Maxi2Node).asInstanceOf[Maxi2Node];}

  override def create(net:Net):Maxi2Layer = {Maxi2Layer(net, this);}
  
  override def toString = {
    "maxi2@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object Maxi2Layer {  
  
  def apply(net:Net) = new Maxi2Layer(net, new Maxi2Node);
  
  def apply(net:Net, opts:Maxi2Node) = new Maxi2Layer(net, opts);
}