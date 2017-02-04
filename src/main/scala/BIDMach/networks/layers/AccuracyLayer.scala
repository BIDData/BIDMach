package BIDMach.networks.layers

import BIDMach.networks._

// TODO: Implement this better
class AccuracyLayer(override val net:Net, override val opts:AccuracyNodeOpts = new AccuracyNode) extends Layer(net, opts) {

  override def toString = {
    "accuracy@"+Integer.toHexString(hashCode % 0x10000).toString
  }

}

trait AccuracyNodeOpts extends NodeOpts {
}

class AccuracyNode extends Node with AccuracyNodeOpts {

  override def clone:AccuracyNode = copyTo(new AccuracyNode).asInstanceOf[AccuracyNode]
  
  override def create(net:Net) = AccuracyLayer(net, this)
  
  override def toString = {
    "accuracy@"+Integer.toHexString(hashCode % 0x10000).toString
  }

}

object AccuracyLayer {
  
  def apply(net:Net) = new AccuracyLayer(net, new AccuracyNode)
  
  def apply(net:Net, opts:AccuracyNodeOpts) = new AccuracyLayer(net, opts)

}