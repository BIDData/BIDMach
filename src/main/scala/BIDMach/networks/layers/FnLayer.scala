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



class FnLayer(override val net:Net, override val opts:FnNodeOpts = new FnNode) extends Layer(net, opts) {

  override def forward = {
		  val start = toc;
		  createOutput;
		  if (opts.fwdfn.asInstanceOf[AnyRef] != null) {
		  	output <-- opts.fwdfn(inputData);
		  }
		  clearDeriv;
		  forwardtime += toc - start;
  }

  override def backward = {
		  val start = toc;
		  if (inputDeriv.asInstanceOf[AnyRef] != null &&  opts.bwdfn.asInstanceOf[AnyRef] != null) {
		    inputDeriv ~ inputDeriv + opts.bwdfn(inputData, output, deriv);
		  }
		  backwardtime += toc - start;
  }
  
  override def toString = {
    "fn@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait FnNodeOpts extends NodeOpts { 
  var fwdfn:(Mat)=> Mat = null;
  var bwdfn:(Mat,Mat,Mat)=> Mat = null;
}

class FnNode extends Node with FnNodeOpts {
  	def copyTo(opts:FnNode):FnNode = {
			super.copyTo(opts);
			opts.fwdfn = fwdfn;
			opts.bwdfn = bwdfn;
			opts;
	}

	override def clone:FnNode = {copyTo(new FnNode).asInstanceOf[FnNode];}

  override def create(net:Net):FnLayer = {FnLayer(net, this);}
  
  override def toString = {
    "fn@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object FnLayer {  
  
  def apply(net:Net) = new FnLayer(net, new FnNode);
  
  def apply(net:Net, opts:FnNode) = new FnLayer(net, opts);
}