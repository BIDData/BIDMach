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
 * Input layer is currently just a placeholder.
 */

@SerialVersionUID(100L)
class InputLayer(override val net:Net, override val opts:InputNodeOpts = new InputNode) extends Layer(net, opts) {
    override def toString = {
    "input@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait InputNodeOpts extends NodeOpts {}

@SerialVersionUID(100L)
class InputNode extends Node with InputNodeOpts {
  def copyTo(opts:InputNode):InputNode = {
    super.copyTo(opts);
    opts;
  }
    
  override def clone:InputNode = {copyTo(new InputNode)}
  	
  override def create(net:Net):InputLayer = {
    InputLayer(net, this);
  }
  
  override def toString = {
    "input@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}


@SerialVersionUID(100L)
object InputLayer {
  
  def apply(net:Net) = new InputLayer(net, new InputNode);
  
  def apply(net:Net, opts:InputNodeOpts) = new InputLayer(net, opts);
  
}
