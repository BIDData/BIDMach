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
 * Softmaxx layer. Output = exp(input) / sum(exp(input))
 */

@SerialVersionUID(100L)
class SoftmaxxLayer(override val net:Net, override val opts:SoftmaxxNodeOpts = new SoftmaxxNode) extends Layer(net, opts) {
  var one:Mat = null;

  override def forward = {
	val start = toc;
	if (one.asInstanceOf[AnyRef] == null) one = inputData.ones(1,1);
	inplaceNoConnectGetOutput();
	
	output ~ inputData - inputData.maxi(opts.inds)
	exp(output, output);  // ensures sum(exps) is between 1 and nfeats
	output ~ output / output.sum(opts.inds);
	
	forwardtime += toc - start;
  }

  override def backward = {
	val start = toc;
	inplaceNoConnectGetInputDerivs();
	
	if (inputDeriv.asInstanceOf[AnyRef] != null) {
	  val exps = exp(inputData - inputData.maxi(opts.inds));
	  val smax = exps / exps.sum(opts.inds);
      val sderiv = smax ∘ deriv
	  inputDeriv ~ inputDeriv + (sderiv - (smax ∘ sderiv.sum(opts.inds)));
	}
	
	inplaceNoConnectReleaseDeriv();
	backwardtime += toc - start;
  }
  
  override def clear = {
	clearMats;
	one = null;
  }

  override def toString = {
    "softmaxx@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait SoftmaxxNodeOpts extends NodeOpts {
  var inds:IMat = irow(0);
}

@SerialVersionUID(100L)
class SoftmaxxNode extends Node with SoftmaxxNodeOpts {

  override def clone:SoftmaxxNode = {copyTo(new SoftmaxxNode).asInstanceOf[SoftmaxxNode];};

  override def create(net:Net):SoftmaxxLayer = {SoftmaxxLayer(net, this);}

  override def toString = {
    "softmaxx@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

@SerialVersionUID(100L)
object SoftmaxxLayer {

  def apply(net:Net) = new SoftmaxxLayer(net, new SoftmaxxNode);

  def apply(net:Net, opts:SoftmaxxNode) = new SoftmaxxLayer(net, opts);
}
