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
 * Dropout layer with fraction to keep "frac". Deletes the same neurons in forward and backward pass. 
 * Assumes that "randmat" is not changed between forward and backward passes. 
 */

class DropoutLayer(override val net:Net, override val opts:DropoutNodeOpts = new DropoutNode) extends Layer(net, opts) {  
	var randmat:Mat = null;

  override def forward = {
    val start = toc;
		inplaceNoConnect();
		
		if (randmat.asInstanceOf[AnyRef] == null) {
		  randmat = inputData.zeros(inputData.dims)
		} 
		if (net.predicting) {
			output ~ inputData * opts.frac;
		} else {
			rand(randmat);
			randmat ~ randmat < opts.frac
			output ~ inputData ∘ randmat;
		}
		forwardtime += toc - start;
  }

  override def backward = {
		val start = toc;
		inplaceGetInputDerivs;
		
		if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv ∘ randmat);
		
		inplaceReturnDeriv;
		backwardtime += toc - start;
  }
  
  override def clear = {
    clearMats;
    randmat = null;
  } 
  
  override def toString = {
    "dropout@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait DropoutNodeOpts extends NodeOpts {
    var frac = 1f;
}
    
    
class DropoutNode extends Node with DropoutNodeOpts { 
	def copyTo(opts:DropoutNode):DropoutNode = {
			super.copyTo(opts);
			opts.frac = frac;
			opts;
	}

	override def clone:DropoutNode = {copyTo(new DropoutNode);}

	override def create(net:Net):DropoutLayer = {DropoutLayer(net, this);}

	override def toString = {
			"dropout@"+Integer.toHexString(hashCode % 0x10000).toString
	}
}
  
object DropoutLayer { 
  
  def apply(net:Net) = new DropoutLayer(net, new DropoutNode);
  
  def apply(net:Net, opts:DropoutNodeOpts) = new DropoutLayer(net, opts);
}
