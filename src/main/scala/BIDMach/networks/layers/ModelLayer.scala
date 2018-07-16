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
 * Assign an integer id (imodel) which points to the model matrix (or matrices) used by this layer. 
 * nmats is the number of model matrices use by this layer. When multiple model matrices are used, 
 * they are assumed to be consecutive. 
 * If using named model layers, the model layer name is augmented with "_i" for matrices >= 1. 
 */


class ModelLayer(override val net:Net, override val opts:ModelNodeOpts = new ModelNode, val nmats:Int = 1) extends Layer(net, opts) {
	var imodel = 0;
  
  override def getModelMats(net:Net):Unit = {
		imodel = if (net.opts.nmodelmats > 0) {             // If explicit model numbers are given, use them. 
			opts.imodel;
		} else if (opts.modelName.length > 0) {             // If this is a named layer, look it up. 
			if (net.modelMap.containsKey(opts.modelName)) {
				net.modelMap.get(opts.modelName);
			} else {
				val len = net.modelMap.size;
				net.modelMap.put(opts.modelName, len + net.opts.nmodelmats); 	
				for (i <- 1 until nmats) {
				  net.modelMap.put(opts.modelName+"_%d" format i, len + i + net.opts.nmodelmats);
				}
				len;
			}
		} else {                                            // Otherwise return the next available int
			net.imodel += nmats;
			net.imodel - nmats;
		};
  }
}

trait ModelNodeOpts extends NodeOpts {
  var modelName = "";
  var imodel = 0;
  var lr_scale = 1f;
  var bias_scale = 1f;
  var initfn:(Mat,Float)=>Mat = Net.xavier;
  var initv:Float = 1f;
  
  def copyOpts(opts:ModelNodeOpts):ModelNodeOpts = {
    super.copyOpts(opts);
    opts.modelName = modelName;
    opts.imodel = imodel;
    opts.lr_scale = lr_scale;
    opts.bias_scale = bias_scale;
    opts.initfn = initfn;
    opts.initv = initv;
    opts;
  }
}
    
class ModelNode extends Node with ModelNodeOpts {
  
  def copyTo(opts:ModelNode):ModelNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }
    
  override def clone:ModelNode = {
    copyTo(new ModelNode).asInstanceOf[ModelNode];
  }
}
