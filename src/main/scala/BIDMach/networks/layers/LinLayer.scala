package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat,TMat}
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
 * Linear layer. 
 * Includes a model matrix that contains the linear map. 
 */

class LinLayer(override val net:Net, override val opts:LinNodeOpts = new LinNode) extends ModelLayer(net, opts) {
  var vexp:Mat = null;
  var texp:Mat = null;
  var lrate:Mat = null;
  var modelcols:IMat  = null;
//  var sumsq:Mat = null;
  var mask:Mat = null;
  var dprod:Mat = null;
  var firststep = -1f;
  var waitsteps = 0;
  var epsilon = 0f;
  var ADAinitialized = false;
  
  def initModelMat(nr:Int, nc:Int):Mat = {
    if (opts.tmatShape != null) {
      val (y, x, h, w) = opts.tmatShape(nr, nc);
      val out = TMat(nr, nc, y, x, h, w, zeros(1,1));
      out.tiles.foreach((x:Mat) => {rand(x); x ~ x - 0.5f; x ~ x *@ opts.initv})
      out;
    } else {
    	(rand(nr, nc) - 0.5f) *@ opts.initv;
    }
  }

  override def forward = {
    val start = toc;
    val modelcols = inputData.nrows;
    if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
      val outdim = if (opts.outdim == 0) inputData.nrows else opts.outdim;
      modelmats(imodel) = convertMat(initModelMat(outdim, modelcols + (if (opts.hasBias) 1 else 0)));
      updatemats(imodel) = modelmats(imodel).zeros(modelmats(imodel).nrows, modelmats(imodel).ncols);  
    }
    if (opts.aopts != null && !ADAinitialized) initADAGrad;
    val mm = if (opts.hasBias) modelmats(imodel).view(modelmats(imodel).nrows, modelcols) else modelmats(imodel);
    createOutput(mm.nrows \ inputData.ncols);
    if (opts.withInteractions) {
      GLM.pairMult(mm, inputData, output);
    } else {
    	output ~ mm * inputData; 
    }
    if (opts.hasBias) output ~ output + modelmats(imodel).colslice(modelcols, modelcols+1);
    clearDeriv;
    forwardtime += toc - start;
  }

  override def backward(ipass:Int, pos:Long) = {
    val start = toc;
    if (modelcols.asInstanceOf[AnyRef] == null) {
    	modelcols = irow(inputData.nrows);
    }
    val mm = if (opts.hasBias) modelmats(imodel).view(modelmats(imodel).nrows, modelcols.v) else modelmats(imodel);
    if (inputDeriv.asInstanceOf[AnyRef] != null) {
      mm.madd(deriv, inputDeriv, true, false);
    }
    if (opts.aopts != null) {
      if (firststep <= 0) firststep = pos.toFloat;
      val step = (pos + firststep)/firststep;
      if (opts.withInteractions) {
      	ADAGrad.pairMultUpdate(deriv, inputData, modelmats(imodel), updatemats(imodel), mask, lrate, vexp, texp, epsilon, step, waitsteps, opts.hasBias);        
      } else {
      	ADAGrad.multUpdate(deriv, inputData, modelmats(imodel), updatemats(imodel), mask, lrate, vexp, texp, epsilon, step, waitsteps, opts.hasBias);
      }
    } else {
    	val um = if (opts.hasBias) updatemats(imodel).view(updatemats(imodel).nrows, modelcols.v) else updatemats(imodel);
    	deriv.madd(inputData, um, false, true);
      if (opts.hasBias) updatemats(imodel)(?,modelcols) = updatemats(imodel)(?,modelcols) + sum(deriv,2)
    }
    backwardtime += toc - start;
  }


  def initADAGrad {
    val aopts = opts.aopts;
    val mm = modelmats(imodel); 
    val d = mm.nrows;
    val m = mm.ncols;
    firststep = -1f;
    lrate = convertMat(aopts.lrate);
    texp = convertMat(aopts.texp);
    vexp = convertMat(aopts.vexp);
//    sumsq = convertMat(zeros(d, m));
    updatemats(imodel).set(aopts.initsumsq);
    waitsteps = aopts.waitsteps;
    epsilon = aopts.epsilon;
    mask = aopts.mask;
    ADAinitialized = true;
  }
  
  override def toString = {
    "linear@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait LinNodeOpts extends ModelNodeOpts {
	var hasBias:Boolean = false;
  var aopts:ADAGrad.Opts = null;
  var outdim = 0;
  var tmatShape:(Int, Int) => (Array[Int], Array[Int], Array[Int], Array[Int]) = null;
  var withInteractions = false;
  var initv:Float = 1f;
  
  def copyOpts(opts:LinNodeOpts):LinNodeOpts = {
  		super.copyOpts(opts);
  		opts.hasBias = hasBias;
  		opts.aopts = aopts;
  		opts.outdim = outdim;
  		opts.tmatShape = tmatShape;
  		opts.withInteractions = withInteractions;
  		opts.initv = initv;
  		opts;
  }
}
    
class LinNode extends ModelNode with LinNodeOpts {
  def copyTo(opts:LinNode):LinNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }
  
  override def toString = {
    "linear@"+Integer.toHexString(hashCode % 0x10000).toString
  }
    
  override def clone:LinNode = {
    copyTo(new LinNode).asInstanceOf[LinNode];
  }
  
  override def create(net:Net):LinLayer = {
  	LinLayer(net, this);
  }
}

object LinLayer {  

  def apply(net:Net) = new LinLayer(net, new LinNode);
  
  def apply(net:Net, opts:LinNodeOpts):LinLayer = new LinLayer(net, opts);
  
}