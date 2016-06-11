package BIDMach.networks.layers

import BIDMat.{Mat,ND,SBMat,CMat,DMat,FMat,FND,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,GND,SMat,SDMat}
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

class SplitHorizLayer(override val net:Net, override val opts:SplitHorizNodeOpts = new SplitHorizNode) extends Layer(net, opts) {
  override val _outputs = new Array[ND](opts.nparts)
  override val _derivs = new Array[ND](opts.nparts)
  var nblock:Int = 0
  var colranges = new Array[Mat](opts.nparts)
  
  override def forward = {
      val start = toc
      if (output.asInstanceOf[AnyRef] == null) {
        nblock = inputData.ncols / opts.nparts
        for (i <- 0 until opts.nparts) {
          colranges(i) = convertMat(irow((i*nblock)->((i+1)*nblock)))
        }
      }
      for (i <- 0 until opts.nparts) {
        setOutput(i, inputData.colslice(i*nblock, (i+1)* nblock))
      }
      clearDerivs
      forwardtime += toc - start
  }

  override def backward = {
      val start = toc
      if (inputDeriv.asInstanceOf[AnyRef] != null) {
        for (i <- 0 until opts.nparts) {
          inputDeriv(?, colranges(i)) = inputDeriv(?, colranges(i)) + derivs(i)
        }
      }  
      backwardtime += toc - start
  }
  
  override def toString = {
    "splithorize@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait SplitHorizNodeOpts extends NodeOpts { 
  var nparts = 1
}

class SplitHorizNode extends Node with SplitHorizNodeOpts {

  override def clone:SplitHorizNode = {copyTo(new SplitHorizNode).asInstanceOf[SplitHorizNode];}

  override def create(net:Net):SplitHorizLayer = {SplitHorizLayer(net, this);}
  
  override def toString = {
    "splithorize@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object SplitHorizLayer {  
  
  def apply(net:Net) = new SplitHorizLayer(net, new SplitHorizNode)
  
  def apply(net:Net, opts:SplitHorizNode) = new SplitHorizLayer(net, opts)
}