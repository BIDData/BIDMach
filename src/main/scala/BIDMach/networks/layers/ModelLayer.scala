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


class ModelLayer(override val net:Net, override val opts:ModelNodeOpts = new ModelNode) extends Layer(net, opts) {
  var imodel = 0
  
  override def getModelMats(net:Net):Unit = {
    imodel = if (net.opts.nmodelmats > 0) {   // If explicit model numbers are given, use them. 
      opts.imodel
    } else if (opts.modelName.length > 0) {               // If this is a named layer, look it up. 
      if (net.modelMap.containsKey(opts.modelName)) {
        net.modelMap.get(opts.modelName)
      } else {
        val len = net.modelMap.size
        net.modelMap.put(opts.modelName, len + net.opts.nmodelmats);  
        len
      }
    } else {                                         // Otherwise return the next available int
      net.imodel += 1
      net.imodel - 1
    }
  }
}

trait ModelNodeOpts extends NodeOpts {
  var modelName = ""
  var imodel = 0
  
  def copyOpts(opts:ModelNodeOpts):ModelNodeOpts = {
    super.copyOpts(opts)
    opts.modelName = modelName
    opts.imodel = imodel
    opts
  }
}
    
class ModelNode extends Node with ModelNodeOpts {
  
  def copyTo(opts:ModelNode):ModelNode = {
    this.asInstanceOf[Node].copyTo(opts)
    copyOpts(opts)
    opts
  }
    
  override def clone:ModelNode = {
    copyTo(new ModelNode).asInstanceOf[ModelNode]
  }
}
