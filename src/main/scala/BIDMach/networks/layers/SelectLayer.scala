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
/*
 * Designed to map linear integer feature arrays to sparse matrices. Doesnt deal with derivatives.
 */

class SelectLayer(override val net:Net, override val opts:SelectNodeOpts = new SelectNode) extends Layer(net, opts) {
  var colindx:IMat = null;
  var fullindx:IMat = null;

  override val _inputs = new Array[LayerTerm](2);
    
  override def forward = {
		  val start = toc;
		  
		  val indx = inputDatas(1).asInstanceOf[IMat];
		  if (colindx.asInstanceOf[AnyRef] == null || colindx.ncols != indx.ncols) {
		    colindx = indx + 0;
		    colindx <-- irow(0->indx.ncols) *@ inputData.nrows;
		  }
		  fullindx = colindx + indx;
		  
		  output = inputData(fullindx);
		  
			clearDeriv;
		  forwardtime += toc - start;
  }
  
  override def backward = {
  		val start = toc;
  		
  		if (inputDeriv.asInstanceOf[AnyRef] != null){
  		  inputDeriv(fullindx) = inputDeriv(fullindx) + deriv;
  		}
  		
  		backwardtime += toc - start;
  }
  
  
  override def toString = {
    "select@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait SelectNodeOpts extends NodeOpts {  
}

class SelectNode extends Node with SelectNodeOpts {
	override val inputs:Array[NodeTerm] = new Array[NodeTerm](2);

	override def clone:SelectNode = {copyTo(new SelectNode).asInstanceOf[SelectNode];}

  override def create(net:Net):SelectLayer = {SelectLayer(net, this);}
  
  override def toString = {
    "select@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object SelectLayer {  
  
  def apply(net:Net) = new SelectLayer(net, new SelectNode);
  
  def apply(net:Net, opts:SelectNode) = new SelectLayer(net, opts);
}