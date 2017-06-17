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

class SplitVertLayer(override val net:Net, override val opts:SplitVertNodeOpts = new SplitVertNode) extends Layer(net, opts) {
  override val _outputs = new Array[Mat](opts.nparts);
  override val _derivs = new Array[Mat](opts.nparts);
  var nblock:Int = 0;
  var colranges = new Array[IMat](opts.nparts);
  var ndims = 0;
  var tensorFormat = Net.TensorNCHW; 
  
  override def forward = {
		  val start = toc;
		  val dims = inputData.dims;
		  if (output.asInstanceOf[AnyRef] == null) {
		  	if (net.asInstanceOf[AnyRef] != null) tensorFormat = net.opts.tensorFormat;
		    ndims = dims.length;
			  nblock = dims(0) / opts.nparts;
			  for (i <- 0 until opts.nparts) {		    
				  colranges(i) = inputData.izeros(1, nblock).asInstanceOf[IMat];
				  colranges(i) <-- icol((i*nblock)->((i+1)*nblock));
			  }
		  }
		  for (i <- 0 until opts.nparts) {
		  	if (tensorFormat == Net.TensorNCHW) {
		  		ndims match {
		  		case 2 =>	setOutput(i, inputData(colranges(i), ?));
		  		case 3 =>	{
		  			val indat = inputData.reshapeView(dims(1), dims(0), dims(2));
		  		  setOutput(i, indat(?, colranges(i), ?).reshapeView(nblock, dims(1), dims(2)));
		  		}
		  		case 4 =>	{
		  			val indat = inputData.reshapeView(dims(1), dims(2), dims(0), dims(3));
		  		  setOutput(i, indat(?, ?, colranges(i), ?).reshapeView(nblock, dims(1), dims(2), dims(3)));	  		
		  		}
		  		case 5 =>	{
		  			val indat = inputData.reshapeView(dims(1), dims(2), dims(3), dims(0), dims(4));
		  		  setOutput(i, indat(?, ?, ?, colranges(i), ?).reshapeView(nblock, dims(1), dims(2), dims(3), dims(4)));
		  		}
		  		}
		  	} else {
		  		ndims match {
		  		case 2 =>	setOutput(i, inputData(colranges(i), ?));
		  		case 3 =>	setOutput(i, inputData(colranges(i), ?, ?));
		  		case 4 =>	setOutput(i, inputData(colranges(i), ?, ?, ?));
		  		case 5 =>	setOutput(i, inputData(colranges(i), ?, ?, ?, ?));
		  		}		    
		  	}
		  }
		  clearDerivs;
		  forwardtime += toc - start;
  }

  override def backward = {
		  val start = toc;
		  val dims = inputData.dims;
		  if (inputDeriv.asInstanceOf[AnyRef] != null) {
			  for (i <- 0 until opts.nparts) {
			  	if (tensorFormat == Net.TensorNCHW) {
			  		ndims match {
			  		case 2 => inputDeriv(colranges(i), ?) = inputDeriv(colranges(i), ?) + derivs(i);
			  		case 3 => {
			  			val inderiv = inputDeriv.reshapeView(dims(1), dims(0), dims(2));
			  		  inderiv(?, colranges(i), ?) = inderiv(?, colranges(i), ?) + derivs(i).reshapeView(dims(1), nblock, dims(2));
			  		}
			  		case 4 => {
			  			val inderiv = inputDeriv.reshapeView(dims(1), dims(2), dims(0), dims(3));
			  		  inderiv(?, ?, colranges(i), ?) = inderiv(?, ?, colranges(i), ?) + derivs(i).reshapeView(dims(1), dims(2), nblock, dims(3));
			  		}
			  		case 5 => {
			  			val inderiv = inputDeriv.reshapeView(dims(1), dims(2), dims(3), dims(0), dims(4));
			  		  inderiv(?, ?, ?, colranges(i), ?) = inderiv(?, ?, ?, colranges(i), ?) + derivs(i).reshapeView(dims(1), dims(2), dims(3), nblock, dims(4));
			  		}
			  		}
			  	} else {
			  		ndims match {
			  		case 2 => inputDeriv(colranges(i), ?) = inputDeriv(colranges(i), ?) + derivs(i);
			  		case 3 => inputDeriv(colranges(i), ?, ?) = inputDeriv(colranges(i), ?, ?) + derivs(i);
			  		case 4 => inputDeriv(colranges(i), ?, ?, ?) = inputDeriv(colranges(i), ?, ?, ?) + derivs(i);
			  		case 5 => inputDeriv(colranges(i), ?, ?, ?, ?) = inputDeriv(colranges(i), ?, ?, ?, ?) + derivs(i);
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
    "splitvert@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait SplitVertNodeOpts extends NodeOpts {  
  var nparts = 1
}

class SplitVertNode extends Node with SplitVertNodeOpts {

	override def clone:SplitVertNode = {copyTo(new SplitVertNode).asInstanceOf[SplitVertNode];}

  override def create(net:Net):SplitVertLayer = {SplitVertLayer(net, this);}
  
  override def toString = {
    "splitvert@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object SplitVertLayer {  
  
  def apply(net:Net) = new SplitVertLayer(net, new SplitVertNode);
  
  def apply(net:Net, opts:SplitVertNode) = new SplitVertLayer(net, opts);
}