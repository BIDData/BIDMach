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
 * Computes the column-wise dot products of two input layers.
 */

class DotLayer(override val net:Net, override val opts:DotNodeOpts = new DotNode) extends Layer(net, opts) { 
  
  override val _inputs = new Array[LayerTerm](2);

	override def forward = {
      val start = toc;
			createOutput(1 \ inputData.ncols);
			inplaceNoConnectGetOutput();
			
			(output ~ inputData) dot inputDatas(1);
			
			forwardtime += toc - start;
	}

	override def backward = {
      val start = toc;
      inplaceNoConnectGetInputDerivs();
      
      if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv *@ inputDatas(1));
      if (inputDerivs(1).asInstanceOf[AnyRef] != null) inputDerivs(1) ~ inputDerivs(1) + (deriv *@ inputData);

      inplaceNoConnectReleaseDeriv()
			backwardtime += toc - start;
	}
  
  override def toString = {
    "dot@"+("%04x" format (hashCode % 0x10000));
  }
}

trait DotNodeOpts extends NodeOpts {

}

class DotNode extends Node with DotNodeOpts {
	 override val inputs:Array[NodeTerm] = new Array[NodeTerm](2);
  
   def copyTo(opts:DotNode):DotNode = {
      super.copyTo(opts);
      opts;
  }

	override def clone:DotNode = {copyTo(new DotNode).asInstanceOf[DotNode];}

	override def create(net:Net):DotLayer = {DotLayer(net, this);}
  
  override def toString = {
   "dot@"+("%04x" format (hashCode % 0x10000));
  }
}

object DotLayer { 
  
  def apply(net:Net) = new DotLayer(net, new DotNode);
  
  def apply(net:Net, opts:DotNodeOpts) = new DotLayer(net, opts); 
}
