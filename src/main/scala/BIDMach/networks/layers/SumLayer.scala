package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,FND,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,GND,ND,SMat,SDMat}
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
 * Sum layer. 
 */

class SumLayer(override val net:Net, override val opts:SumNodeOpts = new SumNode) extends Layer(net, opts) {
	var vmap:ND = null;

  override def forward = {
		  val start = toc;
		  createOutput(1 \ inputData.ncols);
		  output.asMat <-- sum(inputData.asMat);
		  clearDeriv;
		  forwardtime += toc - start;
  }

  override def backward = {
		  val start = toc;
		  if (vmap.asInstanceOf[AnyRef] == null) vmap = deriv.ones(output.nrows, 1);
		  if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (vmap * deriv);  
		  backwardtime += toc - start;
  }
  
  override def toString = {
    "sum@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait SumNodeOpts extends NodeOpts {  
}

class SumNode extends Node with SumNodeOpts {

	override def clone:SumNode = {copyTo(new SumNode).asInstanceOf[SumNode];}

  override def create(net:Net):SumLayer = {SumLayer(net, this);}
  
  override def toString = {
    "sum@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object SumLayer {  
  
  def apply(net:Net) = new SumLayer(net, new SumNode);
  
  def apply(net:Net, opts:SumNode) = new SumLayer(net, opts);
}