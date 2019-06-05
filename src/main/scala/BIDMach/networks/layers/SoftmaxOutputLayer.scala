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
 * Softmax output layer = softmax + loss + max row select. 
 * 
 */

@SerialVersionUID(100L)
class SoftmaxOutputLayer(override val net:Net, override val opts:SoftmaxOutputNodeOpts = new SoftmaxOutputNode) extends Layer(net, opts) with OutputLayer { 
  var coloffsets:IMat = null;
  var probs:Mat = null;
  var eps:Mat = null;
  
  def getInds() = {
	if (coloffsets.asInstanceOf[AnyRef] == null) coloffsets = int(convertMat(irow(0->inputData.ncols)*inputData.nrows));
	int(target) + coloffsets;
  }
  
  def computeProbs() {
	probs = inputData.reshapeView(inputData.nrows, inputData.ncols) - maxi(inputData);  // ensures sum(exps) is between 1 and nfeats
	exp(probs, probs); 
	probs ~ probs / sum(probs); 
  }

  override def forward = {
	val start = toc;
	createOutput(1 \ inputData.ncols);
	inplaceNoConnectSetupDerivs();

    computeProbs();
    
    val inds = getInds();
    opts.lossType match {
      case SoftmaxOutputLayer.TargetProbs => {
        output <-- probs(inds);
      }
      case SoftmaxOutputLayer.CrossEntropyLoss => {
        if (eps.asInstanceOf[AnyRef] == null) eps = convertMat(row(opts.eps));
    	ln(probs(inds)+eps, output);
      }
    }

    forwardtime += toc - start;
  }

  override def backward = {
	val start = toc;
	inplaceNoConnectGetInputDerivs();

	if (inputDeriv.asInstanceOf[AnyRef] != null) {	    
	  val inds = getInds();
	  
	  computeProbs();
	  
	  opts.lossType match {
		case SoftmaxOutputLayer.TargetProbs => {
		  val probinds = probs(inds) ∘ deriv;
		  val oderiv = probs ∘ probinds;
		  inputDeriv ~ inputDeriv - oderiv.reshapeView(inputDeriv.dims)
		  inputDeriv(inds) = inputDeriv(inds) + probinds; 		      
		}
		case SoftmaxOutputLayer.CrossEntropyLoss => {
		  probs ~ probs ∘ deriv;
		  inputDeriv ~ inputDeriv - probs.reshapeView(inputDeriv.dims);
		  inputDeriv(inds) = inputDeriv(inds) + deriv;
		}
	  }
    }
	
	inplaceNoConnectReleaseDeriv();
	backwardtime += toc - start;
  }
  
  override def score:FMat = {
    val inds = getInds();
    if (opts.scoreType == SoftmaxOutputLayer.AccuracyScore) {
      FMat(probs(inds) == maxi(probs));
    } else {
      FMat(ln(probs(inds)));
    }
  }
  
  override def clear = {
  	clearMats;
  	coloffsets = null;
  	eps = null;
  }
  
  override def toString = {
    "softmaxout@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait SoftmaxOutputNodeOpts extends NodeOpts {
  var scoreType = 1;
  var lossType = 1;
  var eps = 1e-6f;
  
  def copyOpts(opts:SoftmaxOutputNodeOpts):SoftmaxOutputNodeOpts = {
	super.copyOpts(opts);
	opts.scoreType = scoreType;
	opts.lossType = lossType;
	opts.eps = eps;
	opts;
  }
}

@SerialVersionUID(100L)
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
  
  final val CrossEntropyLoss = 1;
  final val MultinomialLogisticLoss = 1;
  final val CaffeMultinomialLogisticLoss = 1;
  final val LogTargetProbs = 1;
  final val MultinomialLoss = 2;
  final val TargetProbs = 2;
  
  def apply(net:Net) = new SoftmaxOutputLayer(net, new SoftmaxOutputNode);
  
  def apply(net:Net, opts:SoftmaxOutputNode) = new SoftmaxOutputLayer(net, opts);
}
