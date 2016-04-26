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
  var coloffsets:Mat = null;

	override def forward = {
			val start = toc;
			createOutput;
			val exps = exp(inputData.asMat - maxi(inputData.asMat));  // ensures sum(exps) is between 1 and nfeats
			output.asMat ~ exps / sum(exps);
			clearDeriv;
			forwardtime += toc - start;
	}

	override def backward = {
			val start = toc;
			val exps = exp(inputData.asMat - maxi(inputData.asMat));
			val sumexps = sum(exps);
			val isum = 1f / (sumexps ∘ sumexps);
			if (inputDeriv.asInstanceOf[AnyRef] != null) 
        inputDeriv.asMat ~ inputDeriv.asMat + (((exps / sumexps) ∘ deriv.asMat) - (exps ∘ (isum ∘ (exps ∙ deriv.asMat))));
			backwardtime += toc - start;
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
