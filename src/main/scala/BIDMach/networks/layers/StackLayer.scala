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

class StackLayer(override val net:Net, override val opts:StackNodeOpts = new StackNode) extends Layer(net, opts) {
  override val _inputs = new Array[LayerTerm](opts.ninputs);

  var colranges = new Array[Mat](opts.ninputs);
  
  override def forward = {
		  val start = toc;
		  if (output.asInstanceOf[AnyRef] == null) {
			  var orows = 0;
			  for (i <- 0 until opts.ninputs) {
				  val thisrow = inputDatas(i).nrows;
				  colranges(i) = convertMat(irow(orows -> (orows + thisrow)));
				  orows += thisrow;
			  }
			  output = convertMat(zeros(orows \ inputData.ncols));
		  }
		  for (i <- 0 until opts.ninputs) {
			  output.asMat(colranges(i), ?) = inputDatas(i).asMat;
		  }
		  clearDeriv;
		  forwardtime += toc - start;
  }

  override def backward = {
		  val start = toc;
		  for (i <- 0 until opts.ninputs) {
			  if (inputDerivs(i).asInstanceOf[AnyRef] != null) {
				  inputDerivs(i) <-- deriv(colranges(i), ?)
			  }
		  }  
		  backwardtime += toc - start;
  }
  
  override def toString = {
    "stack@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


trait StackNodeOpts extends NodeOpts {  
  var ninputs = 2;
}

class StackNode extends Node with StackNodeOpts {
	override val inputs = new Array[NodeTerm](ninputs);

	override def clone:StackNode = {copyTo(new StackNode).asInstanceOf[StackNode];}

  override def create(net:Net):StackLayer = {StackLayer(net, this);}
  
  override def toString = {
    "stack@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object StackLayer {  
  
  def apply(net:Net) = new StackLayer(net, new StackNode);
  
  def apply(net:Net, opts:StackNode) = new StackLayer(net, opts);
}