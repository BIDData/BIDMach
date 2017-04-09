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
 * Constant layer outputs its value argument
 */

class ConstantLayer(override val net:Net, override val opts:ConstantNodeOpts = new ConstantNode) extends Layer(net, opts) {
  
  override def forward = {
  		val start = toc;
			
			if (output.asInstanceOf[AnyRef] == null) {
			  output = net.convertMat(opts.value);
			}
			
			forwardtime += toc - start;
	}
  
  override def toString = {
    "const@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait ConstantNodeOpts extends NodeOpts {
  var value:Mat = null;
}

class ConstantNode extends Node with ConstantNodeOpts {

  def copyTo(opts:ConstantNode):ConstantNode = {
    super.copyTo(opts);
    opts.value = value;
    opts;
  }
    
  override def clone:ConstantNode = {copyTo(new ConstantNode)}
  	
  override def create(net:Net):ConstantLayer = {
    ConstantLayer(net, this);
  }
  
  override def toString = {
    "const@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


  
object ConstantLayer {
  
  def apply(net:Net) = new ConstantLayer(net, new ConstantNode);
  
  def apply(net:Net, opts:ConstantNodeOpts) = new ConstantLayer(net, opts);
  
}
