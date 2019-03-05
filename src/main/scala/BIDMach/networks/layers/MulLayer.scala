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
 * Computes the product of its input layers. 
 */

@SerialVersionUID(100L)
class MulLayer(override val net:Net, override val opts:MulNodeOpts = new MulNode) extends Layer(net, opts) {  
  
	override val _inputs = new Array[LayerTerm](opts.ninputs);
  val qeps = 1e-40f;
  
  def guardSmall(a:Mat, eps:Float):Mat = {
    a + (abs(a) < eps) * (2*eps);
  }

	override def forward = {
    val start = toc;
    inplaceNoConnectGetOutput();
	        
	  output ~ inputData ∘ inputDatas(1);
	  (2 until inputlength).map((i:Int) => output ~ output ∘ inputDatas(i));
	
	  forwardtime += toc - start;
	}

	override def backward = {
    val start = toc;
    inplaceNoConnectGetInputDerivs();
    
    if (_inputs.length == 2) {
      if (inputDerivs(0).asInstanceOf[AnyRef] != null) inputDerivs(0) ~ inputDerivs(0) + squash(deriv ∘ inputDatas(1), inputDerivs(0));
      if (inputDerivs(1).asInstanceOf[AnyRef] != null) inputDerivs(1) ~ inputDerivs(1) + squash(deriv ∘ inputDatas(0), inputDerivs(1));
    } else {
			val doutput = deriv ∘ output;
			(0 until inputlength).map((i:Int) => {
				if (inputDerivs(i).asInstanceOf[AnyRef] != null) inputDerivs(i) ~ inputDerivs(i) + squash(doutput / guardSmall(inputDatas(i), qeps), inputDerivs(i));
			});
    }
    
    inplaceNoConnectReleaseDeriv()
    backwardtime += toc - start;
	}
  
  override def toString = {
    "mul@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait MulNodeOpts extends NodeOpts {  
	var ninputs = 2;
}

@SerialVersionUID(100L)
class MulNode extends Node with MulNodeOpts {
  override val inputs:Array[NodeTerm] = new Array[NodeTerm](ninputs);
  
  def copyTo(opts:MulNode):MulNode = {
      super.copyTo(opts);
      opts.ninputs = ninputs;
      opts;
  }

	override def clone:MulNode = {copyTo(new MulNode).asInstanceOf[MulNode];}

	override def create(net:Net):MulLayer = {MulLayer(net, this);}
  
  override def toString = {
    "mul@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}
  
@SerialVersionUID(100L)
object MulLayer {  
  
  def apply(net:Net) = new MulLayer(net, new MulNode);
  
  def apply(net:Net, opts:MulNodeOpts) = new MulLayer(net, opts); 
}
