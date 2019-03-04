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
 * Computes the element-wise difference of input layers. The first argument is added to the output, others are subtracted.
 */

@SerialVersionUID(100L)
class SubLayer(override val net:Net, override val opts:SubNodeOpts = new SubNode) extends Layer(net, opts) { 
  
  override val _inputs = new Array[LayerTerm](opts.ninputs);

	override def forward = {
      val start = toc;
      inplaceNoConnectGetOutput();
      
			output ~ inputData - inputDatas(1);
			(2 until inputlength).map((i:Int) => output ~ output - inputDatas(i));
		
			forwardtime += toc - start;
	}

	override def backward = {
      val start = toc;
      inplaceNoConnectGetInputDerivs();
      
      if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + squash(deriv, inputDeriv);
			(1 until inputlength).map((i:Int) => {
				if (inputDerivs(i).asInstanceOf[AnyRef] != null) inputDerivs(i) ~ inputDerivs(i) - squash(deriv, inputDerivs(i));
			});
			
			inplaceNoConnectReleaseDeriv()
			backwardtime += toc - start;
	}
  
  override def toString = {
    "sub@"+("%04x" format (hashCode % 0x10000));
  }
}

trait SubNodeOpts extends NodeOpts {
	var ninputs = 2;
}

@SerialVersionUID(100L)
class SubNode extends Node with SubNodeOpts {
	 override val inputs:Array[NodeTerm] = new Array[NodeTerm](ninputs);
  
   def copyTo(opts:SubNode):SubNode = {
      super.copyTo(opts);
      opts.ninputs = ninputs;
      opts;
  }

	override def clone:SubNode = {copyTo(new SubNode).asInstanceOf[SubNode];}

	override def create(net:Net):SubLayer = {SubLayer(net, this);}
  
  override def toString = {
   "sub@"+("%04x" format (hashCode % 0x10000));
  }
}

@SerialVersionUID(100L)
object SubLayer { 
  
  def apply(net:Net) = new SubLayer(net, new SubNode);
  
  def apply(net:Net, opts:SubNodeOpts) = new SubLayer(net, opts); 
}
