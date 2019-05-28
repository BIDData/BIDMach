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
class ColpermLayer(override val net:Net, override val opts:ColpermNodeOpts = new ColpermNode) extends Layer(net, opts) {

  override val _inputs = new Array[LayerTerm](2);

  override def forward = {
	val start = toc;
	if (output.asInstanceOf[AnyRef] == null) {
      val odims = irow(inputData.nrows,inputDatas(1).length)
	  output = inputData.zeros(odims);
    }
	inplaceNoConnectGetOutput();
		  
    output <-- inputData(?, inputDatas(1).asInstanceOf[IMat]);

	forwardtime += toc - start;
  }

  override def backward = {
	val start = toc;
	inplaceNoConnectGetInputDerivs();
	
	val inds = inputDatas(1).asInstanceOf[IMat];
	if (inputDeriv.asInstanceOf[AnyRef] != null) {
      inputDeriv(?, inds) = inputDeriv(?, inds) + deriv;
	}  
	inplaceNoConnectReleaseDeriv();
	backwardtime += toc - start;
  }
  
  override def clear = {
    clearMats;
  }
  
  override def toString = {
    "colperm@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


trait ColpermNodeOpts extends NodeOpts {  
}

@SerialVersionUID(100L)
class ColpermNode extends Node with ColpermNodeOpts {
  override val inputs:Array[NodeTerm] = new Array[NodeTerm](2);
  
  override def clone:ColpermNode = {copyTo(new ColpermNode).asInstanceOf[ColpermNode];}

  override def create(net:Net):ColpermLayer = {ColpermLayer(net, this);}
  
  override def toString = {
    "colperm@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

@SerialVersionUID(100L)
object ColpermLayer {  
  
  def apply(net:Net) = new ColpermLayer(net, new ColpermNode);
  
  def apply(net:Net, opts:ColpermNode) = new ColpermLayer(net, opts);
}



