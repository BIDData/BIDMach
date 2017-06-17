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

  var colranges = new Array[IMat](opts.ninputs);
  var ndims = 0;
  var ocols = 0;
  var tensorFormat = Net.TensorNCHW; 
  
  override def forward = {
		  val start = toc;
		  if (output.asInstanceOf[AnyRef] == null) {
		    if (net.asInstanceOf[AnyRef] != null) tensorFormat = net.opts.tensorFormat;
		    ndims = inputData.dims.length;
			  ocols = 0;
			  for (i <- 0 until opts.ninputs) {
				  val thiscol = inputDatas(i).dims(0);
				  colranges(i) = inputData.izeros(1,thiscol).asInstanceOf[IMat];
				  colranges(i) <-- irow(ocols -> (ocols + thiscol));
				  ocols += thiscol;
			  }
			  val odims = inputData.dims.copy;
			  odims(0) = ocols;
			  output = inputData.zeros(odims);
		  }
		  val odims = output.dims;
		  for (i <- 0 until opts.ninputs) {
		  	if (tensorFormat == Net.TensorNCHW) {
		  		ndims match {
		  		case 2 => output(colranges(i), ?) = inputDatas(i);
		  		case 3 => {
		  		  val out = output.reshapeView(odims(1), odims(0), odims(2));
		  		  out(?, colranges(i), ?) = inputDatas(i).reshapeView(odims(1), inputDatas(i).dims(0), odims(2));
		  		}
		  		case 4 => {
		  			val out = output.reshapeView(odims(1), odims(2), odims(0), odims(3));
		  		  output(?, ?, colranges(i), ?) = inputDatas(i).reshapeView(odims(1), odims(2), inputDatas(i).dims(0), odims(3));
		  		}
		  		case 5 => {
		  			val out = output.reshapeView(odims(1), odims(2), odims(3), odims(0), odims(4));
		  		  output(?, ?, ?, colranges(i), ?) = inputDatas(i).reshapeView(odims(1), odims(2), odims(3), inputDatas(i).dims(0), odims(4));
		  		}
		  		}
		  	} else {
		  	  ndims match {
		  		case 2 => output(colranges(i), ?) = inputDatas(i);
		  		case 3 => output(colranges(i), ?, ?) = inputDatas(i);
		  		case 4 => output(colranges(i), ?, ?, ?) = inputDatas(i);
		  		case 5 => output(colranges(i), ?, ?, ?, ?) = inputDatas(i);
		  		}
		  	}
		  }
		  clearDeriv;
		  forwardtime += toc - start;
  }

  override def backward = {
		  val start = toc;
		  val odims = output.dims;
		  for (i <- 0 until opts.ninputs) {
		  	if (inputDerivs(i).asInstanceOf[AnyRef] != null) {
		  		if (tensorFormat == Net.TensorNCHW) {
		  			ndims match {
		  			case 2 => inputDerivs(i) ~ inputDerivs(i) + deriv(colranges(i), ?);
		  			case 3 => {
		  			  val oderiv = deriv.reshapeView(odims(1), odims(0), odims(2));
		  			  inputDerivs(i) ~ inputDerivs(i) + oderiv(?, colranges(i), ?).reshapeView(inputDerivs(i).dims);
		  			}
		  			case 4 => {
		  				val oderiv = deriv.reshapeView(odims(1), odims(2), odims(0), odims(3));
		  			  inputDerivs(i) ~ inputDerivs(i) + oderiv(?, ?, colranges(i), ?).reshapeView(inputDerivs(i).dims);
		  			}
		  			case 5 => {
		  				val oderiv = deriv.reshapeView(odims(1), odims(2), odims(3), odims(0), odims(4));
		  			  inputDerivs(i) ~ inputDerivs(i) + oderiv(?, ?, ?, colranges(i), ?).reshapeView(inputDerivs(i).dims);
		  			}
		  			}
		  		} else {
		  			ndims match {
		  			case 2 => inputDerivs(i) ~ inputDerivs(i) + deriv(colranges(i), ?);
		  			case 3 => inputDerivs(i) ~ inputDerivs(i) + deriv(colranges(i), ?, ?);
		  			case 4 => inputDerivs(i) ~ inputDerivs(i) + deriv(colranges(i), ?, ?, ?);
		  			case 5 => inputDerivs(i) ~ inputDerivs(i) + deriv(colranges(i), ?, ?, ?, ?);
		  			}
		  		}
			  }
		  }  
		  backwardtime += toc - start;
  }
  
  override def clear = {
    clearMats;
    colranges = null;
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