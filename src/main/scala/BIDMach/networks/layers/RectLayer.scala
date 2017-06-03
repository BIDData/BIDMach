package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,ND,SMat,SDMat}
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
 * Rectifying Linear Unit layer.
 */

class RectLayer(override val net:Net, override val opts:RectNodeOpts = new RectNode) extends Layer(net, opts) {
	override def forward = {
      val start = toc;
      if (opts.inplace) {
        output = inputData;
      } else {
      	createOutput;
      }
			max(inputData, 0f, output);
			clearDeriv;
			forwardtime += toc - start;
	}

	override def backward = {
			val start = toc;
			if (inputDeriv.asInstanceOf[AnyRef] != null) {
			  if (opts.inplace) {
			  	RectLayer.rectHelper(output, deriv, deriv);
			    inputDeriv ~ inputDeriv + deriv; 
			  } else {
			  	inputDeriv ~ inputDeriv + (deriv ∘ (inputData > 0f));
			  }
			}
			backwardtime += toc - start;
	}
  
  override def toString = {
    "rect@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait RectNodeOpts extends NodeOpts {
  var inplace:Boolean = false;
}
    
class RectNode extends Node with RectNodeOpts {
  def copyTo(opts:RectNode):RectNode = {
    super.copyTo(opts);
    opts;
  }
    
  override def clone:RectNode = {
    copyTo(new RectNode);
  }
  
  override def create(net:Net):RectLayer = {
  	RectLayer(net, this);
  }
  
  override def toString = {
    "rect@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object RectLayer {  
  
  def apply(net:Net) = new RectLayer(net, new RectNode);
  
  def apply(net:Net, opts:RectNodeOpts) = new RectLayer(net, opts);
  
  def rectHelper(a:Mat, b:Mat, c:Mat):Mat = {
    ND.checkDims("RectLayer", a.dims.data, b.dims.data);
    ND.checkDims("RectLayer", a.dims.data, c.dims.data);
    (a, b, c) match {
      case (aa:GMat, bb:GMat, cc:GMat) => {
        aa.gOp(bb, cc, GMat.BinOp.op_ifpos);
      }
      case (aa:GDMat, bb:GDMat, cc:GDMat) => {
        aa.gOp(bb, cc, GMat.BinOp.op_ifpos);
      }
      case (aa:FMat, bb:FMat, cc:FMat) => {
        var i = 0;
        while (i < a.length) {
          cc.data(i) = (if (aa.data(i) > 0f) bb.data(i) else 0f);
          i += 1;
        }
      }
      case (aa:DMat, bb:DMat, cc:DMat) => {
        var i = 0;
        while (i < a.length) {
          cc.data(i) = (if (aa.data(i) > 0.0) bb.data(i) else 0.0);
          i += 1;
        }
      }
    }
    c;
  }
}
