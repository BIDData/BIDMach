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
 * Sign layer. 
 */

class SignLayer(override val net:Net, override val opts:SignNodeOpts = new SignNode) extends Layer(net, opts) {

	override def forward = {
			val start = toc;
			inplaceNoConnectGetOutput();
					  
			sign(inputData, output);

			forwardtime += toc - start;
	}

	override def backward = {
			val start = toc; 
			
			backwardtime += toc - start;
	}
  
  override def toString = {
    "exp@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


trait SignNodeOpts extends NodeOpts {  
}

class SignNode extends Node with SignNodeOpts {

	override def clone:SignNode = {copyTo(new SignNode).asInstanceOf[SignNode];}

  override def create(net:Net):SignLayer = {SignLayer(net, this);}
  
  override def toString = {
    "exp@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object SignLayer {  
  
  def apply(net:Net) = new SignLayer(net, new SignNode);
  
  def apply(net:Net, opts:SignNode) = new SignLayer(net, opts);
}