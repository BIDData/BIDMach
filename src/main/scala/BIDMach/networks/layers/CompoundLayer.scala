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


@SerialVersionUID(100L)
class CompoundLayer(override val net:Net, override val opts:CompoundNode = new CompoundNode) extends ModelLayer(net, opts) {
	 
  override def setInput(i:Int, v:LayerTerm):CompoundLayer = {               // Assumes the inputs are the first k layers in internal_layers
    _inputs(i) = v;
    internal_layers(i).setInput(0, v);
    this
  }
	
	var grid:LayerMat = null;
	
	def internal_layers:Array[Layer] = grid.data;
	
	override def forward = {
			val start = toc;
			for (i <- 0 until grid.ncols) {
			  for (j <- 0 until grid.nrows) {
			  	val layer = grid(j, i);
			    if (layer != null) {
			    	if (net.opts.debug != 0) {
			    		println("  compound layer forward (%d,%d) %s" format (j, i, layer.getClass));
			    	}
			      layer.forward;
			    }
			  }
			}

			for (i <- 0 until opts.outputNumbers.length) {
				_outputs(i) = grid(opts.outputNumbers(i)).output;
				if (_derivs(i).asInstanceOf[AnyRef] == null){
					_derivs(i) = grid(opts.outputNumbers(i)).deriv;
				}
			}
			forwardtime += toc - start;
	}
	
	override def backward(ipass:Int, pos:Long) = {
		val start = toc;
		for (i <- (grid.ncols - 1) to 0 by -1) {
		  for (j <- (grid.nrows -1) to 0 by -1) {
		  	val layer = grid(j, i);
		  	if (layer != null) {
		  		if (net.opts.debug != 0) {
		  			println("  compound layer backward (%d,%d) %s" format (j, i, layer.getClass));
		  		}
		  		layer.backward(ipass, pos);
		  	}
			}
		}
    backwardtime += toc - start;
	}
		
	override def getModelMats(net:Net) = {
		for (i <- 0 until grid.ncols) {
			for (j <- 0 until grid.nrows) {
				val layer = grid(j, i);
				if (layer != null) {
					layer.getModelMats(net);
				}
			}
		}
	}

	def construct = {
//		internal_layers = new Array[Layer](opts.lopts.length);
		grid = LayerMat(opts.grid.nrows, opts.grid.ncols);
		for (i <- 0 until grid.ncols) {
		  for (j <- 0 until grid.nrows) {
		  	val node = opts.grid(j, i);
		    if (node != null) {
		    	grid(j, i) = node.create(net);
		    	node.myLayer = grid(j, i);
		    	grid(j, i).parent = this;
		    }
		  }
		}
		for (i <- 0 until grid.ncols) {
			for (j <- 0 until grid.nrows) {
			  val node = opts.grid(j, i);
			  if (node != null) {
			  	for (k <- 0 until node.inputs.length) {
			  		if (node.inputs(k) != null) {
			  			val nodeTerm = node.inputs(k);      
			  			grid(j, i).setInput(k, new LayerTerm(nodeTerm.node.myLayer, nodeTerm.term));
			  		}
			  	}
			  	grid(j, i) match {
			  	case aa:LinLayer => aa.opts.aopts = opts.aopts;
			  	case _ =>
			  	}
			  }
			}
	  }
	}
	
	override def clear = {
	  clearMats;
	  for (i <- 0 until internal_layers.length) {
	    if (internal_layers(i).asInstanceOf[AnyRef] != null) {
	      internal_layers(i).clear;
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

@SerialVersionUID(100L)
class CompoundNode extends ModelNode with CompoundNodeOpts {
	var grid:NodeMat = null;
//  var lopts:Array[Node] = null;
  
  override def toString = {
    "compound@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}
