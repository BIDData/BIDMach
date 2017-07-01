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
 * Softmax layer. Output = exp(input) / sum(exp(input))
 */

class SoftmaxLayer(override val net:Net, override val opts:SoftmaxNodeOpts = new SoftmaxNode) extends Layer(net, opts) {
  var one:Mat = null;

  override def forward = {
			val start = toc;
			if (one.asInstanceOf[AnyRef] == null) one = inputData.ones(1,1);
			inplaceNoConnectGetOutput();
			
			output ~ inputData - maxi(inputData)
			exp(output, output);  // ensures sum(exps) is between 1 and nfeats
			output ~ output / sum(output);
		
			forwardtime += toc - start;
	}

	override def backward = {
			val start = toc;
			inplaceNoConnectGetInputDerivs();
			
			if (inputDeriv.asInstanceOf[AnyRef] != null) {
				val exps = exp(inputData - maxi(inputData));
				val smax = exps / sum(exps);
				inputDeriv ~ inputDeriv + ((smax ∘ deriv) - (smax ∘ (smax ∙ deriv)));
			}
			
			inplaceNoConnectReleaseDeriv();
			backwardtime += toc - start;
	}
	
	override def clear = {
	  clearMats;
	  one = null;
	}

  override def toString = {
    "softmax@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait SoftmaxNodeOpts extends NodeOpts {

}

class SoftmaxNode extends Node with SoftmaxNodeOpts {

	override def clone:SoftmaxNode = {copyTo(new SoftmaxNode).asInstanceOf[SoftmaxNode];};

  override def create(net:Net):SoftmaxLayer = {SoftmaxLayer(net, this);}

  override def toString = {
    "softmax@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object SoftmaxLayer {

  def apply(net:Net) = new SoftmaxLayer(net, new SoftmaxNode);

  def apply(net:Net, opts:SoftmaxNode) = new SoftmaxLayer(net, opts);
}
