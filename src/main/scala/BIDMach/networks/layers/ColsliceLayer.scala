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
class ColsliceLayer(override val net:Net, override val opts:ColsliceNodeOpts = new ColsliceNode) extends Layer(net, opts) {

  override def forward = {
	val start = toc;
	if (output.asInstanceOf[AnyRef] == null) {
      val odims = inputData(0).dims.copy;
      odims(odims.length-1) = opts.b - opts.a;
	  output = inputData.zeros(odims);
	}
	inplaceNoConnectGetOutput();
		  
    inputData.colslice(opts.a, opts.b, output, 0);

	forwardtime += toc - start;
  }

  override def backward = {
	val start = toc;
	inplaceNoConnectGetInputDerivs();
	
	val odims = output.dims;
	if (inputDeriv.asInstanceOf[AnyRef] != null) {
      val olddata = inputDeriv.colslice(opts.a, opts.b);
      olddata ~ olddata + deriv;
      olddata.colslice(0, olddata.ncols, inputDeriv, opts.a);
	}  
	inplaceNoConnectReleaseDeriv();
	backwardtime += toc - start;
  }
  
  override def clear = {
    clearMats;
  }
  
  override def toString = {
    "cols@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


trait ColsliceNodeOpts extends NodeOpts {  
  var a = 0;
  var b = -1;

  def copyOpts(opts:ColsliceNodeOpts):ColsliceNodeOpts = {
	super.copyOpts(opts);
	opts.a = a;
    opts.b = b;
	opts;
  }

}

@SerialVersionUID(100L)
class ColsliceNode extends Node with ColsliceNodeOpts {
  
  override def clone:ColsliceNode = {copyTo(new ColsliceNode).asInstanceOf[ColsliceNode];}

  override def create(net:Net):ColsliceLayer = {ColsliceLayer(net, this);}
  
  override def toString = {
    "cols@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

@SerialVersionUID(100L)
object ColsliceLayer {  
  
  def apply(net:Net) = new ColsliceLayer(net, new ColsliceNode);
  
  def apply(net:Net, opts:ColsliceNode) = new ColsliceLayer(net, opts);
}
