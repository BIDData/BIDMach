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
import scala.util.hashing.MurmurHash3
import java.util.HashMap
import BIDMach.networks._


/**
 * Softplus layer.  
 */

class SoftplusLayer(override val net:Net, override val opts:SoftplusNodeOpts = new SoftplusNode) extends Layer(net, opts) {
  var totflops = 0L

  override def forward = {
      val start = toc
      createOutput
      LayerFn.applyfwd(inputData, output, LayerFn.SOFTPLUSFN)
      clearDeriv
      forwardtime += toc - start
  }

  override def backward = {
      val start = toc
      if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + LayerFn.applyderiv(inputData, deriv, LayerFn.SOFTPLUSFN)
      backwardtime += toc - start
  }
  
   override def toString = {
    "softplus@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait SoftplusNodeOpts extends NodeOpts {  
}

class SoftplusNode extends Node with SoftplusNodeOpts {

  override def clone:SoftplusNode = {copyTo(new SoftplusNode).asInstanceOf[SoftplusNode];}

  override def create(net:Net):SoftplusLayer = {SoftplusLayer(net, this);}
  
  override def toString = {
    "softplus@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object SoftplusLayer {  
  
  def apply(net:Net) = new SoftplusLayer(net, new SoftplusNode)
  
  def apply(net:Net, opts:SoftplusNode) = new SoftplusLayer(net, opts)
}

