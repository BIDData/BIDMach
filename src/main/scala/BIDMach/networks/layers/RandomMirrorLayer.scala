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
 * Random Mirror layer. Randomly Mirrors the input with probability p.  
 * Does not pass derivatives.
 */

class RandomMirrorLayer(override val net:Net, override val opts:RandomMirrorNodeOpts = new RandomMirrorNode) extends Layer(net, opts) {
  var mirrorInds:IMat = null;
  var randomSelector:Mat = null;
  
  def setupInds = {
    val dims = inputData.dims;
    val w = dims(1);
    mirrorInds = inputData.izeros(1, w).asInstanceOf[IMat];
    mirrorInds <-- irow((w-1) to 0 by -1);
    randomSelector = inputData.zeros(1\1\1\dims(3));
  }

	override def forward = {
  		val start = toc;
  		
  		val dims = inputData.dims;
			if (mirrorInds.asInstanceOf[AnyRef] == null) setupInds;
			
			val tformat = Net.getCUDNNformat(opts.tensorFormat, net.opts.tensorFormat);
			val mirrored = if (tformat == Net.TensorNHWC) {
				inputData(?, mirrorInds, ?, ?);
			} else {
			  val realshape = inputData.reshapeView(dims(1), dims(2), dims(0), dims(3));
			  realshape(mirrorInds, ?, ?, ?).reshapeView(dims);
			}
			mirrored ~ mirrored - inputData;
			rand(randomSelector);
			randomSelector ~ randomSelector < opts.prob;
			mirrored ~ mirrored *@ randomSelector;
			output = inputData + mirrored;

			forwardtime += toc - start;
	}
	
	override def clear = {
	  clearMats;
	  mirrorInds = null;
    randomSelector = null;
	}
  
  override def toString = {
    "randmirror@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait RandomMirrorNodeOpts extends NodeOpts {
  var prob = 0.5f;
  var tensorFormat:Int = Net.UseNetFormat;
    
  def copyOpts(opts:RandomMirrorNodeOpts):RandomMirrorNodeOpts = {
  		super.copyOpts(opts);
  		opts.prob = prob;
  		opts.tensorFormat = tensorFormat;
  		opts;
  }
}

class RandomMirrorNode extends Node with RandomMirrorNodeOpts {
  def copyTo(opts:RandomMirrorNode):RandomMirrorNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }

	override def clone:RandomMirrorNode = {copyTo(new RandomMirrorNode).asInstanceOf[RandomMirrorNode];}

  override def create(net:Net):RandomMirrorLayer = {RandomMirrorLayer(net, this);}
  
  override def toString = {
    "randmirror@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object RandomMirrorLayer {  
  
  def apply(net:Net) = new RandomMirrorLayer(net, new RandomMirrorNode);
  
  def apply(net:Net, opts:RandomMirrorNode) = new RandomMirrorLayer(net, opts);
}