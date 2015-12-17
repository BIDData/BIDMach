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
 * Normalization layer adds a downward-propagating derivative term whenever its norm 
 * is different from the optsified value (targetNorm).
 */

class NormLayer(override val net:Net, override val opts:NormNodeOpts = new NormNode) extends Layer(net, opts) {
	var sconst:Mat = null;

  override def forward = {
		val start = toc;
		createOutput;
		output <-- inputData;
		clearDeriv;
		forwardtime += toc - start;
  }

  override def backward = {
		val start = toc;
    if (inputDeriv.asInstanceOf[AnyRef] != null) {
    	if (sconst.asInstanceOf[AnyRef] == null) sconst = output.zeros(1,1);
    	sconst.set(math.min(0.1f, math.max(-0.1f, (opts.targetNorm - norm(output)/output.length).toFloat * opts.weight)));
    	inputDeriv = output ∘ sconst;
    	inputDeriv ~ inputDeriv + deriv;
    }
    backwardtime += toc - start;
  }
}

trait NormNodeOpts extends NodeOpts {
	var targetNorm = 1f;
	var weight = 1f;
	
	def copyOpts(opts:NormNodeOpts):NormNodeOpts = {
  		super.copyOpts(opts);
  		opts.targetNorm = targetNorm;
  		opts.weight = weight;
  		opts;
    }
}

class NormNode extends Node with NormNodeOpts {  
    
   def copyTo(opts:NormNode):NormNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }
    
    override def clone:NormNode = {copyTo(new NormNode).asInstanceOf[NormNode];}
    
    override def create(net:Net):NormLayer = {NormLayer(net, this);}
  }
  
object NormLayer {  
  
  def apply(net:Net) = new NormLayer(net, new NormNode);
  
  def apply(net:Net, opts:NormNode) = new NormLayer(net, opts);  
}
