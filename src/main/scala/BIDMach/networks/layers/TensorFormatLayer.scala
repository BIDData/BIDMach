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
 * TensorFormat layer. Can convert from NHWC to NCHW or vice-versa, or do an identity map.
 * Assumes the input type is known. If conversion is set to "auto" will convert from the input type to the net type. 
 */

@SerialVersionUID(100L)
class TensorFormatLayer(override val net:Net, override val opts:TensorFormatNodeOpts = new TensorFormatNode) extends Layer(net, opts) {

	override def forward = {
			val start = toc;

			opts.conversion match {
			  case TensorFormatLayer.NHWCtoNCHW => output = inputData.asInstanceOf[FMat].fromNHWCtoNCHW;
			  case TensorFormatLayer.NCHWtoNHWC => output = inputData.asInstanceOf[FMat].fromNCHWtoNHWC;
			  case TensorFormatLayer.AUTO => {
			    (opts.inputFormat, net.opts.tensorFormat) match {
			      case (Net.TensorNHWC, Net.TensorNCHW) => output = inputData.asInstanceOf[FMat].fromNHWCtoNCHW;
			      case (Net.TensorNCHW, Net.TensorNHWC) => output = inputData.asInstanceOf[FMat].fromNCHWtoNHWC;
			      case _ => output = inputData;
			    }
			  }
			}
			
			clearDerivLazy;

			forwardtime += toc - start;
	}
	
	 override def backward = {
  		val start = toc;
  		
  		if (inputDeriv.asInstanceOf[AnyRef] != null) {
  			opts.conversion match {
  			case TensorFormatLayer.NHWCtoNCHW => inputDeriv ~ inputDeriv + deriv.asInstanceOf[FMat].fromNCHWtoNHWC;
  			case TensorFormatLayer.NCHWtoNHWC => inputDeriv ~ inputDeriv + deriv.asInstanceOf[FMat].fromNHWCtoNCHW;
  			case TensorFormatLayer.AUTO => {
  				(opts.inputFormat, net.opts.tensorFormat) match {
  				case (Net.TensorNHWC, Net.TensorNCHW) => inputDeriv ~ inputDeriv + deriv.asInstanceOf[FMat].fromNCHWtoNHWC;
  				case (Net.TensorNCHW, Net.TensorNHWC) => inputDeriv ~ inputDeriv + deriv.asInstanceOf[FMat].fromNHWCtoNCHW;
  				case _ => inputDeriv ~ inputDeriv + deriv;
  				}
  			}
  			}
  		}  		
  		
  		backwardtime += toc - start;
	 }
  
  override def toString = {
    "format@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait TensorFormatNodeOpts extends NodeOpts {
  var conversion:Int = TensorFormatLayer.AUTO;
  var inputFormat:Int = Net.TensorNHWC;
  
  def copyOpts(opts:TensorFormatNodeOpts):TensorFormatNodeOpts = {
  		super.copyOpts(opts);
  		opts.conversion = conversion;
  		opts.inputFormat = inputFormat;
  		opts;
  }
}

@SerialVersionUID(100L)
class TensorFormatNode extends Node with TensorFormatNodeOpts {
  def copyTo(opts:TensorFormatNode):TensorFormatNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }

	override def clone:TensorFormatNode = {copyTo(new TensorFormatNode).asInstanceOf[TensorFormatNode];}

  override def create(net:Net):TensorFormatLayer = {TensorFormatLayer(net, this);}
  
  override def toString = {
    "format@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

@SerialVersionUID(100L)
object TensorFormatLayer {  
  final val AUTO = 0;
  final val NHWCtoNCHW = 1;
  final val NCHWtoNHWC = 2;
  
  def apply(net:Net) = new TensorFormatLayer(net, new TensorFormatNode);
  
  def apply(net:Net, opts:TensorFormatNode) = new TensorFormatLayer(net, opts);
}
