package BIDMach.networks.layers

import BIDMat.{Mat,ND,SBMat,CMat,DMat,FMat,FND,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,GND,SMat,SDMat}
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

class SplitVertLayer(override val net:Net, override val opts:SplitVertNodeOpts = new SplitVertNode) extends Layer(net, opts) {
  override val _outputs = new Array[ND](opts.nparts);
  override val _derivs = new Array[ND](opts.nparts);
  var nblock:Int = 0;
  var rowranges = new Array[Mat](opts.nparts);
  
  override def forward = {
		  val start = toc;
		  if (output.asInstanceOf[AnyRef] == null) {
			  nblock = inputData.nrows / opts.nparts;
			  for (i <- 0 until opts.nparts) {
				  rowranges(i) = convertMat(icol((i*nblock)->((i+1)*nblock)));
			  }
		  }
		  for (i <- 0 until opts.nparts) {
			  setOutput(i, inputData(rowranges(i), ?));
		  }
		  clearDerivs;
		  forwardtime += toc - start;
  }

  override def backward = {
		  val start = toc;
		  if (inputDeriv.asInstanceOf[AnyRef] != null) {
			  for (i <- 0 until opts.nparts) {
				  inputDeriv(rowranges(i), ?) = inputDeriv(rowranges(i), ?) + derivs(i);
			  }
		  }
		  backwardtime += toc - start;
  }
  
  override def toString = {
    "splitverte@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait SplitVertNodeOpts extends NodeOpts {  
  var nparts = 1
}

class SplitVertNode extends Node with SplitVertNodeOpts {

	override def clone:SplitVertNode = {copyTo(new SplitVertNode).asInstanceOf[SplitVertNode];}

  override def create(net:Net):SplitVertLayer = {SplitVertLayer(net, this);}
  
  override def toString = {
    "splitverte@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object SplitVertLayer {  
  
  def apply(net:Net) = new SplitVertLayer(net, new SplitVertNode);
  
  def apply(net:Net, opts:SplitVertNode) = new SplitVertLayer(net, opts);
}