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



class ReshapeLayer(override val net:Net, override val opts:ReshapeNodeOpts = new ReshapeNode) extends Layer(net, opts) {

  override def forward = {
    val start = toc;
    if (output.asInstanceOf[AnyRef] == null) {
      val io = inputData;
      val newdims = if (opts.addBatchDim) {
	if (io.nrows != opts.dims.data.reduce(_*_)) {
	  throw new RuntimeException("ReshapeLayer input and output dims not compatible")
	}
	opts.dims \ io.ncols;
      } else {
	if (io.length != opts.dims.data.reduce(_*_)) {
	  throw new RuntimeException("ReshapeLayer input and output dims not compatible")
	}
	opts.dims;
      }
      output = io match {
	case s:IMat => io.izeros(newdims)
	case m:Mat => io.zeros(newdims)
      }
    }
    output <-- inputData.reshapeView(output.dims);
    inplaceNoConnectSetupDerivs();
    
    forwardtime += toc - start;
  }

  override def backward = {
    val start = toc;
    inplaceNoConnectGetInputDerivs();
    
    if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + deriv.reshapeView(inputDeriv.dims)
    
    inplaceNoConnectReleaseDeriv();
    backwardtime += toc - start;
  }

  override def toString = {
    "reshape@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait ReshapeNodeOpts extends NodeOpts {
  var dims:IMat = null;                  // New tensor dimensions.
  var addBatchDim:Boolean = true;        // If true, last input dimension (batch size) is appended to dims. If false, dims are used directly for output.
  
  def copyOpts(opts:ReshapeNodeOpts):ReshapeNodeOpts = {
		super.copyOpts(opts);
		opts.dims = dims;           
		opts.addBatchDim = addBatchDim;
		opts;
  }
}

class ReshapeNode extends Node with ReshapeNodeOpts {
  
	def copyTo(opts:ReshapeNode):ReshapeNode = {
			this.asInstanceOf[Node].copyTo(opts);
			copyOpts(opts);
			opts
	}

	override def clone:ReshapeNode = {copyTo(new ReshapeNode).asInstanceOf[ReshapeNode];}

	override def create(net:Net):ReshapeLayer = {ReshapeLayer(net, this);};

	override def toString = {
			"reshape@"+Integer.toHexString(hashCode % 0x10000).toString
	}
}

object ReshapeLayer {

  def apply(net:Net) = new ReshapeLayer(net, new ReshapeNode);

  def apply(net:Net, opts:ReshapeNode) = new ReshapeLayer(net, opts);
}
