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
  var coloffsets:IMat = null;
  var zero:Mat = null;

  override def forward = {
		  val start = toc;
      createOutput;
      output ~ inputData - maxi(inputData)
      exp(output, output);  // ensures sum(exps) is between 1 and nfeats
      output ~ output / sum(output);
      clearDeriv;
      forwardtime += toc - start;
  }

  override def backward = {
		  val start = toc;
		  if (coloffsets.asInstanceOf[AnyRef] == null) coloffsets = int(convertMat(irow(0->output.ncols)*output.nrows));
		  if (inputDeriv.asInstanceOf[AnyRef] != null) {
		    if (zero.asInstanceOf[AnyRef] == null) zero = convertMat(row(0f));
        deriv ~ zero - output;
        val inds = int(target) + coloffsets;
			  deriv(inds) = deriv(inds) + 1f;               // deriv = target - preds
        inputDeriv ~ inputDeriv + deriv; 
      }
		  backwardtime += toc - start;
  }
  
  override def score:FMat = {
    if (coloffsets.asInstanceOf[AnyRef] == null) coloffsets = int(convertMat(irow(0->output.ncols)*output.nrows));
    val inds = int(target) + coloffsets;
    if (opts.scoreType == SoftmaxOutputLayer.AccuracyLoss) {
      if (opts.doVariance) {
        val matches = (output(inds) == maxi(output));
        FMat(mean(matches)) on FMat(variance(matches));
      } else {
      	FMat(mean(output(inds) == maxi(output)));
      }
    } else {
    	if (opts.doVariance) {
    	  val out = ln(output(inds));
    	  FMat(mean(out)) on FMat(variance(out));
    	} else {
    		FMat(mean(ln(output(inds))));   
    	}
    }
  }
  
  override def toString = {
    "softmaxout@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait SoftmaxOutputNodeOpts extends NodeOpts {
	var scoreType = 0;
	var doVariance = false;
		
	def copyOpts(opts:SoftmaxOutputNodeOpts):SoftmaxOutputNodeOpts = {
			super.copyOpts(opts);
			opts.scoreType = scoreType;
			opts.doVariance = doVariance;
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
  
  final val CrossEntropyLoss = 0;
  final val AccuracyLoss = 1;
  
  def apply(net:Net) = new SoftmaxOutputLayer(net, new SoftmaxOutputNode);
  
  def apply(net:Net, opts:SoftmaxOutputNode) = new SoftmaxOutputLayer(net, opts);
}
