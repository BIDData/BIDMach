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



class Fn2Layer(override val net:Net, override val opts:Fn2NodeOpts = new Fn2Node) extends Layer(net, opts) {

	override val _inputs = new Array[LayerTerm](2);
    
  override def forward = {
		  val start = toc;
		  createOutput;
		  if (opts.fwdfn.asInstanceOf[AnyRef] != null) {
		  	output <-- opts.fwdfn(inputData, inputDatas(1));
		  }
		  clearDeriv;
		  forwardtime += toc - start;
  }

  override def backward = {
		  val start = toc;
		  if (inputDeriv.asInstanceOf[AnyRef] != null &&  opts.bwdfn1.asInstanceOf[AnyRef] != null) {
		    inputDeriv ~ inputDeriv + opts.bwdfn1(inputData, inputDatas(1), output, deriv);
		  }
		  if (inputDerivs(1).asInstanceOf[AnyRef] != null &&  opts.bwdfn2.asInstanceOf[AnyRef] != null) {
		    inputDeriv ~ inputDeriv + opts.bwdfn2(inputData, inputDatas(1), output, deriv);
		  }
		  backwardtime += toc - start;
  }
  
  override def toString = {
    "fn2@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait Fn2NodeOpts extends NodeOpts { 
  var fwdfn:(Mat,Mat)=> Mat = null;
  var bwdfn1:(Mat,Mat,Mat,Mat)=> Mat = null;
  var bwdfn2:(Mat,Mat,Mat,Mat)=> Mat = null;
}

class Fn2Node extends Node with Fn2NodeOpts {
  	def copyTo(opts:Fn2Node):Fn2Node = {
			super.copyTo(opts);
			opts.fwdfn = fwdfn;
			opts.bwdfn1 = bwdfn1;
			opts.bwdfn2 = bwdfn2;
			opts;
	}

	override def clone:Fn2Node = {copyTo(new Fn2Node).asInstanceOf[Fn2Node];}

  override def create(net:Net):Fn2Layer = {Fn2Layer(net, this);}
  
  override def toString = {
    "fn2@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object Fn2Layer {  
  
  def apply(net:Net) = new Fn2Layer(net, new Fn2Node);
  
  def apply(net:Net, opts:Fn2Node) = new Fn2Layer(net, opts);
}