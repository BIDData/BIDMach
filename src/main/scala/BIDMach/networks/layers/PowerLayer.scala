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
 * Power layer. 
 */

class PowerLayer(override val net:Net, override val opts:PowerNodeOpts = new PowerNode) extends Layer(net, opts) {

  override val _inputs = new Array[LayerTerm](2);
  var one:Mat = null;

  override def forward = {
		  val start = toc;
		  createOutput;
		  output ~ inputData ^ inputDatas(1);
		  clearDeriv;
		  forwardtime += toc - start;
  }

  override def backward = {
		  val start = toc;
		  
		  if (inputDeriv.asInstanceOf[AnyRef] != null) {
		  	if (one.asInstanceOf[AnyRef] == null) one = inputData.ones(1, 1);  
		    val powm1 = inputData ^ (inputDatas(1) - one);
		    powm1 ~ powm1 *@ inputDatas(1);
		    inputDeriv ~ inputDeriv + (powm1 *@ deriv);  
		  }
		  
		  backwardtime += toc - start;
  }
  
  def clear = {
    clearMats;
    one = null;
  }
  
  override def toString = {
    "power@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait PowerNodeOpts extends NodeOpts {  
}

class PowerNode extends Node with PowerNodeOpts {
	override val inputs:Array[NodeTerm] = new Array[NodeTerm](2);

	override def clone:PowerNode = {copyTo(new PowerNode).asInstanceOf[PowerNode];}

  override def create(net:Net):PowerLayer = {PowerLayer(net, this);}
  
  override def toString = {
    "power@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object PowerLayer {  
  
  def apply(net:Net) = new PowerLayer(net, new PowerNode);
  
  def apply(net:Net, opts:PowerNode) = new PowerLayer(net, opts);
}