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


@SerialVersionUID(100L)
class TransposeLayer(override val net:Net, override val opts:TransposeNodeOpts = new TransposeNode) extends Layer(net, opts) {

  override def forward = {
		  val start = toc;
		  if (output.asInstanceOf[AnyRef] == null) {
			  output = inputData.transpose(opts.perm);
		  }
		  val tmp = inputData.transpose(opts.perm);
		  if (tmp.GUID != output.GUID) {
		    output <-- tmp;
		  }
		  inplaceNoConnectSetupDerivs();
      
		  forwardtime += toc - start;
  }

  override def backward = {
		  val start = toc;
		  inplaceNoConnectGetInputDerivs();
		  
		  if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + deriv.transpose(invperm(opts.perm))
		  
		  inplaceNoConnectReleaseDeriv();
		  backwardtime += toc - start;
  }

  override def toString = {
    "transpose@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait TransposeNodeOpts extends NodeOpts {
  var perm:IMat = null;                  // Permutation matrix

  def copyOpts(opts:TransposeNodeOpts):TransposeNodeOpts = {
	super.copyOpts(opts);
	opts.perm = perm;           
	opts;
  }
}

@SerialVersionUID(100L)
class TransposeNode extends Node with TransposeNodeOpts {
  
	def copyTo(opts:TransposeNode):TransposeNode = {
	  this.asInstanceOf[Node].copyTo(opts);
	  copyOpts(opts);
	  opts
	}

	override def clone:TransposeNode = {copyTo(new TransposeNode).asInstanceOf[TransposeNode];}

	override def create(net:Net):TransposeLayer = {TransposeLayer(net, this);};

	override def toString = {
			"transpose@"+Integer.toHexString(hashCode % 0x10000).toString
	}
}

@SerialVersionUID(100L)
object TransposeLayer {

  def apply(net:Net) = new TransposeLayer(net, new TransposeNode);

  def apply(net:Net, opts:TransposeNode) = new TransposeLayer(net, opts);
}
