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
      createoutput;
      output ~ inputData - maxi(inputData)
      exp(output, output);  // ensures sum(exps) is between 1 and nfeats
      output ~ output / sum(output);
      clearDeriv;
      forwardtime += toc - start;
  }

  override def backward = {
		  val start = toc;
		  if (coloffsets.asInstanceOf[AnyRef] == null) coloffsets = convertMat(irow(0->output.ncols)*output.nrows);
		  if (inputDeriv.asInstanceOf[AnyRef] != null) {
		    if (zero.asInstanceOf[AnyRef] == null) zero = convertMat(row(0f));
        deriv ~ zero - output;
        val inds = target + coloffsets;
			  deriv(inds) = deriv(inds) + 1f;               // deriv = target - preds
        inputDeriv ~ inputDeriv + deriv; 
      }
		  backwardtime += toc - start;
  }
  
  override def score:FMat = {
    if (coloffsets.asInstanceOf[AnyRef] == null) coloffsets = convertMat(irow(0->output.ncols)*output.nrows);
    val inds = target + coloffsets;
    if (opts.scoreType == 1) {
      FMat(mean(output(inds) == maxi(output)));
    } else {
    	FMat(mean(ln(output(inds))));   
    }
  }
}

trait SoftmaxOutputNodeOpts extends NodeOpts {
	var scoreType = 0;
		
	def copyOpts(opts:SoftmaxOutputNodeOpts):SoftmaxOutputNodeOpts = {
			super.copyOpts(opts);
			opts.scoreType = scoreType;
			opts;
	}
	
	 def copyTo(opts:SoftmaxOutputNodeOpts):SoftmaxOutputNodeOpts = {
    this.asInstanceOf[NodeOpts].copyTo(opts);
    copyOpts(opts);
    opts
  }
}

class SoftmaxOutputNode extends Node with SoftmaxOutputNodeOpts {

	override def clone:SoftmaxOutputNode = {copyTo(new SoftmaxOutputNode).asInstanceOf[SoftmaxOutputNode];}

	override def create(net:Net):SoftmaxOutputLayer = {SoftmaxOutputLayer(net, this);}
}
  
object SoftmaxOutputLayer { 
  
  def apply(net:Net) = new SoftmaxOutputLayer(net, new SoftmaxOutputNode);
  
  def apply(net:Net, opts:SoftmaxOutputNode) = new SoftmaxOutputLayer(net, opts);
}
