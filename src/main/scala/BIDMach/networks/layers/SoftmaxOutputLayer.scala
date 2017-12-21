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
  var one:Mat = null;
  var eps:Mat = null;

  override def forward = {
		  val start = toc;
      inplaceNoConnectGetOutput();
      
      output ~ inputData - maxi(inputData);  // ensures sum(exps) is between 1 and nfeats
      exp(output, output); 
      output ~ output / sum(output);

      forwardtime += toc - start;
  }

  override def backward = {
		  val start = toc;
		  inplaceNoConnectGetInputDerivs();
		  
		  if (coloffsets.asInstanceOf[AnyRef] == null) {
		    coloffsets = int(convertMat(irow(0->output.ncols)*output.nrows));
		  }
		  if (inputDeriv.asInstanceOf[AnyRef] != null) {
		    if (zero.asInstanceOf[AnyRef] == null) {
		      zero = convertMat(row(0f));
		      one = convertMat(row(1f));
		      eps = convertMat(row(opts.eps));
		    }		    
		    val inds = int(target) + coloffsets;
		    output ~ inputData - maxi(inputData);
		    exp(output, output);
				output ~ output / sum(output);
		    opts.lossType match {
		      case SoftmaxOutputLayer.LogTargetProbs => {
		      	inputDeriv ~ inputDeriv - output;
		      	inputDeriv(inds) = inputDeriv(inds) + one; 
		      }
		      case SoftmaxOutputLayer.TargetProbs => {
		      	val oderiv = output ∘ output(inds);
		      	inputDeriv ~ inputDeriv - oderiv;
		      	inputDeriv(inds) = inputDeriv(inds) + output(inds); 		      
		      }
		      case SoftmaxOutputLayer.CrossEntropyLoss => {
		      	val oneMinusP = one - output;
		      	max(oneMinusP, eps, oneMinusP);
		      	val invOneMinusP = oneMinusP;
		      	invOneMinusP ~ one / oneMinusP;
		      	val logit = output ∘ invOneMinusP;
		      	val slogit = sum(logit) - invOneMinusP;
		      	slogit ~ slogit - invOneMinusP(inds);
		      	slogit ~ slogit ∘ output;
		      	inputDeriv ~ inputDeriv + slogit;
		      	inputDeriv(inds) = inputDeriv(inds) + invOneMinusP(inds);
		      }
		    }
      }
		  
		  inplaceNoConnectReleaseDeriv();
		  backwardtime += toc - start;
  }
  
  override def score:FMat = {
    if (coloffsets.asInstanceOf[AnyRef] == null) coloffsets = int(convertMat(irow(0->output.ncols)*output.nrows));
    val inds = int(target) + coloffsets;
    if (opts.scoreType == SoftmaxOutputLayer.AccuracyScore) {
    	FMat(output(inds) == maxi(output));
    } else {
    	FMat(ln(output(inds)));
    }
  }
  
  override def clear = {
  		clearMats;
  		coloffsets = null;
  		zero = null;
  		one = null;
  		eps = null;
  }
  
  override def toString = {
    "softmaxout@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait SoftmaxOutputNodeOpts extends NodeOpts {
	var scoreType = 0;
	var lossType = 0;
	var eps = 1e-5f;
		
	def copyOpts(opts:SoftmaxOutputNodeOpts):SoftmaxOutputNodeOpts = {
			super.copyOpts(opts);
			opts.scoreType = scoreType;
			opts.lossType = lossType;
			opts.eps = eps;
			opts;
	}
}

class SoftmaxOutputNode extends Node with OutputNode with SoftmaxOutputNodeOpts {
  
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
  
  final val CrossEntropyScore = 0;
  final val AccuracyScore = 1;
 
  final val CrossEntropyLoss = 0;
  final val MultinomialLogisticLoss = 0;
  final val CaffeMultinomialLogisticLoss = 1;
  final val LogTargetProbs = 1;
  final val MultinomialLoss = 2;
  final val TargetProbs = 2;
  
  def apply(net:Net) = new SoftmaxOutputLayer(net, new SoftmaxOutputNode);
  
  def apply(net:Net, opts:SoftmaxOutputNode) = new SoftmaxOutputLayer(net, opts);
}
