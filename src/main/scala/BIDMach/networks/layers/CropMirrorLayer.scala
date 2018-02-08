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
import java.util.Random;


/**
 * Crop+Mirror layer. Crops the input according to the dimensions specified in the "sizes" option, and randomly mirrors in X. 
 * A zero size leaves that dimension untouched. 
 * Does not pass derivatives
 */

class CropMirrorLayer(override val net:Net, override val opts:CropMirrorNodeOpts = new CropMirrorNode) extends Layer(net, opts) {
  var blockInds:Array[IMat] = null;
  var baseInds:Array[IMat] = null;
  var baseMirrorInds:IMat = null;
  var roffset:IMat = null;
  var offsets:IMat = null;
  var sizes:IMat = null;
  var random:Random = null;
  
  def setupInds = {
    val dims = inputData.dims;
    if (dims.length != opts.sizes.length) throw new RuntimeException("CropLayer sizes param doesnt match input dimension");
    random = new Random;
    blockInds = new Array[IMat](dims.length);
    baseInds = new Array[IMat](dims.length);
    roffset = inputData.izeros(1,1).asInstanceOf[IMat];
    offsets = izeros(1, dims.length)
    for (i <- 0 until dims.length) {
      blockInds(i) = if (opts.sizes(i) <= 0) {
        ? 
      } else {
        val gap = dims(i) - opts.sizes(i);
        offsets(i) = if (opts.offsets.asInstanceOf[AnyRef] != null && opts.offsets(i) >= 0) {
          math.min(opts.offsets(i), gap);
        } else {
          gap/2;
        }
        net.convertMat(irow(offsets(i)->(offsets(i) + opts.sizes(i)))).asInstanceOf[IMat];
      } 
      if (opts.sizes(i) > 0) {
        baseInds(i) = net.convertMat(irow(0->opts.sizes(i))).asInstanceOf[IMat];
        if (i == 1) {
          baseMirrorInds = net.convertMat(irow((opts.sizes(i)-1) to 0 by -1)).asInstanceOf[IMat];
        }
      }
    }
  }
  
  def updateInds = {
  	val dims = inputData.dims;
  	for (i <- 0 until dims.length) {
  	  val gap = dims(i) - opts.sizes(i);
      if (opts.randoffsets(i) > 0 && opts.sizes(i) > 0 && gap > 0) { 
        if (net.predicting || opts.randoffsets.asInstanceOf[AnyRef] == null) {
        	roffset.set(offsets(i));
        } else {
          val offset_magnitude = math.min(gap, opts.randoffsets(i));
          val ioff = math.max(0, math.min(gap-1, gap/2 + (offset_magnitude * (random.nextFloat() -0.5f)).toInt)); 
          roffset.set(ioff);
        }
        if (i == 1 && !net.predicting && random.nextFloat() < 0.5f) {
        	blockInds(i) ~ baseMirrorInds + roffset;
        } else {
        	blockInds(i) ~ baseInds(i) + roffset; 
        }
      }
  	}
  }

	override def forward = {
  		val start = toc;
  		val dims = inputData.dims;
  		if (sizes.asInstanceOf[AnyRef] == null) {
  		  sizes = opts.sizes.copy;
  		}
  		sizes(sizes.length-1) = inputData.ncols;
			if (blockInds.asInstanceOf[AnyRef] == null) setupInds;
			updateInds;
			if (net.opts.tensorFormat == Net.TensorNHWC) {
				blockInds.length match {
				case 2 => output = inputData(blockInds(0), blockInds(1));
				case 3 => output = inputData(blockInds(0), blockInds(1), blockInds(2));
				case 4 => output = inputData(blockInds(0), blockInds(1), blockInds(2), blockInds(3));
				case 5 => output = inputData(blockInds(0), blockInds(1), blockInds(2), blockInds(3), blockInds(4));
				}
			} else {
				blockInds.length match {
				case 2 => output = inputData(blockInds(0), blockInds(1));
				case 3 => {
				  val reshaped = inputData.reshapeView(dims(1), dims(0), dims(2));
				  val cropped = reshaped(blockInds(1), blockInds(0), blockInds(2));
				  output = cropped.reshapeView(sizes(0), sizes(1), sizes(2));
				}
				case 4 => {
				  val reshaped = inputData.reshapeView(dims(1), dims(2), dims(0), dims(3));
				  val cropped = reshaped(blockInds(1), blockInds(2), blockInds(0), blockInds(3));
				  output = cropped.reshapeView(sizes(0), sizes(1), sizes(2), sizes(3));
				}
				case 5 => {
				  val reshaped = inputData.reshapeView(dims(1), dims(2), dims(3), dims(0), dims(4));
				  val cropped = reshaped(blockInds(1), blockInds(2), blockInds(3), blockInds(0), blockInds(4));
				  output = cropped.reshapeView(sizes(0), sizes(1), sizes(2), sizes(3), sizes(4));
				}
				}
			}
      if (net.opts.compute_input_gradient)
        inplaceNoConnectSetupDerivs()
			forwardtime += toc - start;
	}
	
  override def backward = {
    val start = toc;    
    if (net.opts.compute_input_gradient && inputDeriv.asInstanceOf[AnyRef] != null && deriv.asInstanceOf[AnyRef] != null){
        if (net.opts.tensorFormat == Net.TensorNHWC) {
            inputDeriv(blockInds(0), blockInds(1), blockInds(2), blockInds(3)) = deriv
        }
        else{
            val dims = inputData.dims;
            val reshaped = inputDeriv.reshapeView(dims(1), dims(2), dims(0), dims(3));
            val odims = output.dims
            reshaped(blockInds(1), blockInds(2), blockInds(0), blockInds(3)) = deriv.reshapeView(odims(1), odims(2), odims(0), odims(3))
        }
    }
    backwardtime += toc - start;
  }
  
	override def clear = {
	  clearMats;
	  blockInds = null;
    sizes = null;
    baseInds= null;
    roffset = null;
    offsets = null;
	}
  
  override def toString = {
    "cropMirror@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait CropMirrorNodeOpts extends CropNodeOpts {
}

class CropMirrorNode extends Node with CropMirrorNodeOpts {
  def copyTo(opts:CropMirrorNode):CropMirrorNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }

	override def clone:CropMirrorNode = {copyTo(new CropMirrorNode).asInstanceOf[CropMirrorNode];}

  override def create(net:Net):CropMirrorLayer = {CropMirrorLayer(net, this);}
  
  override def toString = {
    "cropMirror@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object CropMirrorLayer {  
  
  def apply(net:Net) = new CropMirrorLayer(net, new CropMirrorNode);
  
  def apply(net:Net, opts:CropMirrorNode) = new CropMirrorLayer(net, opts);
}