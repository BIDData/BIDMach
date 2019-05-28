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
class HcatLayer(override val net:Net, override val opts:HcatNodeOpts = new HcatNode) extends Layer(net, opts) {
  override val _inputs = new Array[LayerTerm](opts.ninputs);

  var colranges:IMat = null;
  var cumcols:IMat = null;
  
  override def forward = {
	val start = toc;
	if (output.asInstanceOf[AnyRef] == null) {
      colranges = izeros(1, opts.ninputs);
	  for (i <- 0 until opts.ninputs) {
        colranges(i) = inputDatas(i).ncols;
	  }
      val cumcols = 0 \ cumsum(colranges);
      val odims = inputData(0).dims.copy;
      odims(odims.length-1) = cumcols(opts.ninputs);
	  output = inputData.zeros(odims);
	}
	inplaceNoConnectGetOutput();
		  
	val odims = output.dims;
	for (i <- 0 until opts.ninputs) {
      inputDatas(i).colslice(0, colranges(i), output, cumcols(i));
	}
	forwardtime += toc - start;
  }

  override def backward = {
	val start = toc;
	inplaceNoConnectGetInputDerivs();
	
	val odims = output.dims;
	for (i <- 0 until opts.ninputs) {
	  if (inputDerivs(i).asInstanceOf[AnyRef] != null) {
		inputDerivs(i) ~ inputDerivs(i) + deriv.colslice(cumcols(i), cumcols(i) + colranges(i));
	  }
	}  
	inplaceNoConnectReleaseDeriv();
	backwardtime += toc - start;
  }
  
  override def clear = {
    clearMats;
    colranges = null;
    cumcols = null;
  }
  
  override def toString = {
    "hcat@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


trait HcatNodeOpts extends NodeOpts {  
  var ninputs = 2;

  def copyOpts(opts:HcatNodeOpts):HcatNodeOpts = {
	super.copyOpts(opts);
	opts.ninputs = ninputs;
	opts;
  }

}

@SerialVersionUID(100L)
class HcatNode extends Node with HcatNodeOpts {
  override val inputs = new Array[NodeTerm](ninputs);
  
  def copyTo(opts:HcatNode):HcatNode = {
	super.copyTo(opts);
	copyOpts(opts);
	opts;
  }

  override def clone:HcatNode = {copyTo(new HcatNode);}

  override def create(net:Net):HcatLayer = {HcatLayer(net, this);}
  
  override def toString = {
    "hcat@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

@SerialVersionUID(100L)
object HcatLayer {  
  
  def apply(net:Net) = new HcatLayer(net, new HcatNode);
  
  def apply(net:Net, opts:HcatNode) = new HcatLayer(net, opts);
}
