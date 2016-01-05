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


class CompoundLayer(override val net:Net, override val opts:CompoundNode = new CompoundNode) extends ModelLayer(net, opts) {
	 
  override def setInput(i:Int, v:LayerTerm):CompoundLayer = {               // Assumes the inputs are the first k layers in internal_layers
    _inputs(i) = v;
    internal_layers(i).setInput(0, v);
    this
  }
	
	var internal_layers:Array[Layer] = null;
	
	var grid:LayerMat = null;
	
	override def forward = {
			val start = toc;
			if (net.opts.debug == 0) {
				internal_layers.map(_.forward);
			} else {
				for (i <- 0 until internal_layers.length) {
					if (net.opts.debug > 0) println("  compound layer forward %d %s" format (i, internal_layers(i).getClass));
					internal_layers(i).forward;
				}
			}
			for (i <- 0 until opts.outputNumbers.length) {
				_outputs(i) = internal_layers(opts.outputNumbers(i)).output
						if (_derivs(i).asInstanceOf[AnyRef] == null){
							_derivs(i) = internal_layers(opts.outputNumbers(i)).deriv;
						}
			}
			forwardtime += toc - start;
	}
	
	override def backward(ipass:Int, pos:Long) = {
		val start = toc;
		if (net.opts.debug == 0) {
	  internal_layers.reverse.map(_.backward(ipass, pos));
		} else {
	    for (i <- internal_layers.length until 1 by -1) {
	      if (net.opts.debug > 0) println("  compound layer backward %d" format (i-1, internal_layers(i-1).getClass));
	      internal_layers(i-1).backward(ipass, pos);
	    }
	  }
    backwardtime += toc - start;
	}
		
	override def getModelMats(net:Net) = {
	  internal_layers.map(_.getModelMats(net));
	}

	def construct = {
		internal_layers = new Array[Layer](opts.lopts.length);
	  for (i <- 0 until internal_layers.length) {
	  	internal_layers(i) = opts.lopts(i).create(net);
	  	opts.lopts(i).myLayer = internal_layers(i);
	  	internal_layers(i).parent = this;
	  }
	  for (i <- 0 until internal_layers.length) {
	  	for (j <- 0 until opts.lopts(i).inputs.length) {
    		if (opts.lopts(i).inputs(j) != null) {
          val nodeTerm = opts.lopts(i).inputs(j);      
          internal_layers(i).setInput(j, new LayerTerm(nodeTerm.node.myLayer, nodeTerm.term));
        }
    	}
      internal_layers(i) match {
        case aa:LinLayer => aa.opts.aopts = opts.aopts;
        case _ =>
      }
	  }
	}
  
  override def toString = {
    "compound@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait CompoundNodeOpts extends ModelNodeOpts {
  var aopts:ADAGrad.Opts = null;
  var prefix = "";
}

class CompoundNode extends ModelNode with CompoundNodeOpts {
	var grid:NodeMat = null;
  var lopts:Array[Node] = null;
  
  override def toString = {
    "compound@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}
