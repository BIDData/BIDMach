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



class ForwardLayer(override val net:Net, override val opts:ForwardNodeOpts = new ForwardNode) extends Layer(net, opts) {

  override def forward = {
		  val start = toc;
		  inplaceNoConnectGetOutput();
		  
		  output <-- inputData;
//		  clearDeriv;
		  forwardtime += toc - start;
  }

  override def backward = {
  }
  
  override def toString = {
    "forward@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait ForwardNodeOpts extends NodeOpts {  
}

class ForwardNode extends Node with ForwardNodeOpts {

	override def clone:ForwardNode = {copyTo(new ForwardNode).asInstanceOf[ForwardNode];}

  override def create(net:Net):ForwardLayer = {ForwardLayer(net, this);}
  
  override def toString = {
    "forward@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object ForwardLayer {  
  
  def apply(net:Net) = new ForwardLayer(net, new ForwardNode);
  
  def apply(net:Net, opts:ForwardNode) = new ForwardLayer(net, opts);
}