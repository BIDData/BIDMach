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
 * GLMLayer implements linear, logistic and hinge-loss SVM. 
 * Commonly used as an output layer so includes a score method.
 */

class GLMLayer(override val net:Net, override val opts:GLMNodeOpts = new GLMNode) extends Layer(net, opts) {
	var ilinks:Mat = null;
	var totflops = 0L;

	override def forward = {
			val start = toc;
			createoutput;
			if (ilinks.asInstanceOf[AnyRef] == null) {
			  ilinks = convertMat(opts.links);
			  for (i <- 0 until opts.links.length) {
			  	totflops += GLM.linkArray(opts.links(i)).fnflops
			  }
			}
			output <-- GLM.preds(inputData, ilinks, totflops);
			clearDeriv;
			forwardtime += toc - start;
	}

	override def backward = {
			val start = toc;
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv ∘ GLM.derivs(output, target, ilinks, totflops));
			backwardtime += toc - start;
	}

	override def score:FMat = { 
			val v = GLM.llfun(output, target, ilinks, totflops);
			FMat(mean(v, 2));
	}
}

trait GLMNodeOpts extends NodeOpts { 
	var links:IMat = null;
}
 
class GLMNode extends Node with GLMNodeOpts {  
	def copyTo(opts:GLMNode) = {
		super.copyTo(opts);
		opts.links = links;
		opts;
	}

	override def clone:GLMNode = {copyTo(new GLMNode);}   

	override def create(net:Net):GLMLayer = {GLMLayer(net, this);}
}
  
object GLMLayer {
  
  def apply(net:Net) = new GLMLayer(net, new GLMNode);
  
  def apply(net:Net, opts:GLMNodeOpts) = new GLMLayer(net, opts); 

}
