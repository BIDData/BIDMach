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



class CopyLayer(override val net:Net, override val opts:CopyNodeOpts = new CopyNode) extends Layer(net, opts) {

  override def forward = {
		  val start = toc;
		  if (output.asInstanceOf[AnyRef] == null) {
			  val io = inputData;
			  output = io match {
			    case s:SMat => io.zeros(io.nrows, io.ncols, io.nnz)
			    case s:IMat => io.izeros(io.dims)
			    case m:Mat => io.zeros(io.dims)
			  }
		  }
		  output <-- inputData;
		  clearDeriv;
		  forwardtime += toc - start;
  }

  override def backward = {
		  val start = toc;
		  if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + deriv;
		  backwardtime += toc - start;
  }

  override def toString = {
    "copy@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait CopyNodeOpts extends NodeOpts {
}

class CopyNode extends Node with CopyNodeOpts {

    override def clone:CopyNode = {copyTo(new CopyNode).asInstanceOf[CopyNode];};

    override def create(net:Net):CopyLayer = {CopyLayer(net, this);};
	
    override def toString = {
	"copy@"+Integer.toHexString(hashCode % 0x10000).toString
    }
}

object CopyLayer {

  def apply(net:Net) = new CopyLayer(net, new CopyNode);

  def apply(net:Net, opts:CopyNode) = new CopyLayer(net, opts);
}
