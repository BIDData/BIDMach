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

class LinLayer(override val net:Net, override val opts:LinNodeOpts = new LinNode) extends ModelLayer(net, opts, 2) {
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
  var ngroups = 0;
  

  def initModelMat(nr:Int, nc:Int, initv:Float):Mat = {
    if (lr_scales.asInstanceOf[AnyRef] != null) {
    	lr_scales(imodel) = opts.lr_scale;
    	lr_scales(imodel+1) = opts.bias_scale;
    }
    ngroups = math.max(opts.ngroups, 1);
    if (opts.tmatShape != null) {
      val (y, x, h, w) = opts.tmatShape(nr, nc);
      val out = TMat(nr, nc, y, x, h, w, zeros(1,1));
      out;
    } else {
      if (opts.ngroups <= 1) {
      	zeros(nr, nc);
      } else {
        zeros(nr \ nc \ ngroups);
      }
    }
  }

  override def forward = {
  	val start = toc;
  	val modelcols = inputData.nrows;
  	if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
  		val outdim = if (opts.outdim == 0) inputData.nrows else opts.outdim;
  		modelmats(imodel) = convertMat(initModelMat(outdim, modelcols, opts.initv));
  		updatemats(imodel) = convertMat(modelmats(imodel).copy);
  		opts.initfn(modelmats(imodel), opts.initv);
  		if (opts.hasBias) {
  			modelmats(imodel+1) = convertMat(zeros(outdim, ngroups));
  			updatemats(imodel+1) = convertMat(zeros(outdim, ngroups));
  			opts.initbiasfn(modelmats(imodel+1), opts.initbiasv);
  		}
  	}
  	if (opts.aopts != null && !ADAinitialized) initADAGrad;
  	val mm = modelmats(imodel);
  	if (opts.ngroups <= 1) {
  		createOutput(mm.nrows \ inputData.ncols);
  	} else {
  	  createOutput(mm.nrows \ inputData.ncols \ ngroups);
  	}
  	inplaceNoConnectGetOutput(true);
  	
  	if (opts.withInteractions) {
  		GLM.pairMult(mm, inputData, output);
  	} else {
  	  if (opts.ngroups <= 1) {
  	  	output ~ mm * inputData;
  	  } else {
  	    output.blockmult(mm, inputData, opts.ngroups);
  	  }
  	}
  	if (opts.hasBias) {
  		output ~ output + modelmats(imodel+1);
  	}
  	forwardtime += toc - start;
  }

  override def backward(ipass:Int, pos:Long) = {
    val start = toc;
    inplaceNoConnectGetInputDerivs();
    
    val mm = modelmats(imodel);
    if (inputDeriv.asInstanceOf[AnyRef] != null) {
      if (opts.ngroups <= 1) {
      	mm.madd(deriv, inputDeriv, true, false);
      } else {
        mm.blockmadd(deriv, inputDeriv, opts.ngroups, true, false);
      }
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
    	val um = updatemats(imodel);
    	if (opts.ngroups <= 1) {
    		deriv.madd(inputData, um, false, true);
    	} else {
    	  mm.blockmadd(deriv, inputDeriv, opts.ngroups, true, false);
    	}
      if (opts.hasBias) updatemats(imodel+1) ~ updatemats(imodel+1) + sum(deriv,2);
    }    
    inplaceNoConnectReleaseDeriv();
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
  
  override def clear = {
  		clearMats;
  		vexp = null;
  		texp = null;
  		lrate = null;
  		modelcols = null;

  		mask = null;
  		dprod = null;
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
  var initfn:(Mat,Float)=>Mat = Net.xavier;
  var initv:Float = 1f;
  var initbiasfn:(Mat,Float)=>Mat = Net.constant;
  var initbiasv:Float = 0f;
  var ngroups = 0;
  
  def copyOpts(opts:LinNodeOpts):LinNodeOpts = {
  		super.copyOpts(opts);
  		opts.hasBias = hasBias;
  		opts.aopts = aopts;
  		opts.outdim = outdim;
  		opts.tmatShape = tmatShape;
  		opts.withInteractions = withInteractions;
  		opts.initfn = initfn;
  		opts.initv = initv;
  		opts.initbiasfn = initbiasfn;
  		opts.initbiasv = initbiasv;
  		opts.ngroups = ngroups;
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
