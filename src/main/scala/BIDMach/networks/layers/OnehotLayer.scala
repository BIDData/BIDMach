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
/*
 * Designed to map linear integer feature arrays to sparse matrices. Doesnt deal with derivatives.
 */

class OnehotLayer(override val net:Net, override val opts:OnehotNodeOpts = new OnehotNode) extends Layer(net, opts) {

  override def forward = {
		  val start = toc;
		  var hot = if (opts.length != null) {
		    oneHot(inputData, opts.length)
		  } else {
		    oneHot(inputData)
		  }
		  output = if (opts.full) full(hot) else hot
		  forwardtime += toc - start;
  }
  
  override def toString = {
    "onehot@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait OnehotNodeOpts extends NodeOpts {  
  var length:Integer = null
  var full:Boolean = false
}

class OnehotNode extends Node with OnehotNodeOpts {

	override def clone:OnehotNode = {copyTo(new OnehotNode).asInstanceOf[OnehotNode];}

  override def create(net:Net):OnehotLayer = {OnehotLayer(net, this);}
  
  override def toString = {
    "onehot@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object OnehotLayer {  
  
  def apply(net:Net) = new OnehotLayer(net, new OnehotNode);
  
  def apply(net:Net, opts:OnehotNode) = new OnehotLayer(net, opts);
}