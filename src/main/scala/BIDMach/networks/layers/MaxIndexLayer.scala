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


class MaxIndexLayer(override val net:Net, override val opts:MaxIndexNodeOpts = new MaxIndexNode) extends Layer(net, opts) {

  override def forward = {
    val start = toc;
    output = maxi2(inputData, 1)._2;
    forwardtime += toc - start;
  }

  override def backward = {
    val start = toc;
    backwardtime += toc - start;
  }

  override def toString = {
    "copy@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait MaxIndexNodeOpts extends NodeOpts {
}

class MaxIndexNode extends Node with MaxIndexNodeOpts {

  override def clone:MaxIndexNode = {copyTo(new MaxIndexNode).asInstanceOf[MaxIndexNode];}

  override def create(net:Net):MaxIndexLayer = {MaxIndexLayer(net, this);}

  override def toString = {
    "maxidx@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object MaxIndexLayer {

  def apply(net:Net) = new MaxIndexLayer(net, new MaxIndexNode);

  def apply(net:Net, opts:MaxIndexNode) = new MaxIndexLayer(net, opts);
}
