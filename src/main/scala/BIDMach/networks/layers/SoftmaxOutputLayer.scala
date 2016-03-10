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

class SoftmaxOutputLayer(override val net:Net, override val opts:SoftmaxOutputNodeOpts = new SoftmaxOutputNode) extends Layer(net, opts) with OutputLayer { 
  var coloffsets:Mat = null;
  var zero:Mat = null;

  override def forward = {
		  val start = toc;
      createOutput;
      output.asMat ~ inputData.asMat - maxi(inputData.asMat)
      exp(output.asMat, output.asMat);  // ensures sum(exps) is between 1 and nfeats
      output.asMat ~ output.asMat / sum(output.asMat);
      clearDeriv;
      forwardtime += toc - start;
  }

  override def backward = {
		  val start = toc;
		  if (coloffsets.asInstanceOf[AnyRef] == null) coloffsets = convertMat(irow(0->output.ncols)*output.nrows);
		  if (inputDeriv.asInstanceOf[AnyRef] != null) {
		    if (zero.asInstanceOf[AnyRef] == null) zero = convertMat(row(0f));
        deriv.asMat ~ zero - output.asMat;
        val inds = target + coloffsets;
			  deriv.asMat(inds) = deriv.asMat(inds) + 1f;               // deriv = target - preds
	    //deriv.asMat~deriv.asMat/max(output.asMat(inds),0.1f)
        inputDeriv ~ inputDeriv + deriv; 
      }
		  backwardtime += toc - start;
  }
  
  override def score:FMat = {
    if (coloffsets.asInstanceOf[AnyRef] == null) coloffsets = convertMat(irow(0->output.ncols)*output.nrows);
    val inds = target + coloffsets;
    if (opts.scoreType == 1) {
      FMat(mean(output(inds) == maxi(output.asMat)));
    } else 
    if (opts.scoreType == 2) {
      FMat(mean(output(inds) >= (sortdown(output.asMat)(4,?))));
        }
    else {
    	FMat(mean(ln(output(inds))));   
    }
  }
  
  override def toString = {
    "softmaxout@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait SoftmaxOutputNodeOpts extends NodeOpts {
	var scoreType = 0;
		
	def copyOpts(opts:SoftmaxOutputNodeOpts):SoftmaxOutputNodeOpts = {
			super.copyOpts(opts);
			opts.scoreType = scoreType;
			opts;
	}
}

class SoftmaxOutputNode extends Node with SoftmaxOutputNodeOpts {
  
  def copyTo(opts:SoftmaxOutputNode):SoftmaxOutputNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }

	override def clone:SoftmaxOutputNode = {copyTo(new SoftmaxOutputNode).asInstanceOf[SoftmaxOutputNode];}

	override def create(net:Net):SoftmaxOutputLayer = {SoftmaxOutputLayer(net, this);}
  
   override def toString = {
    "softmaxout@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}
  
object SoftmaxOutputLayer { 
  
  def apply(net:Net) = new SoftmaxOutputLayer(net, new SoftmaxOutputNode);
  
  def apply(net:Net, opts:SoftmaxOutputNode) = new SoftmaxOutputLayer(net, opts);
}
