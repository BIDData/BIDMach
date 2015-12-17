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


class NegsampOutputLayer(override val net:Net, override val opts:NegsampOutputNodeOpts = new NegsampOutputNode) extends ModelLayer(net, opts) with OutputLayer { 
  var vexp:Mat = null;
  var texp:Mat = null;
  var lrate:Mat = null;
  var iexpt:Mat = null;
  var cfact:Mat = null;
  var cexpt:Mat = null;  
//  var sumsq:Mat = null;
  var mask:Mat = null;
  var firststep = -1f;
  var waitsteps = 0;
  var epsilon = 0f;
  var ADAinitialized = false;
  var randwords:Mat = null;
  var onerow:Mat = null;
  var prods:Mat = null;
  var inputMat:Mat = null;
  var targMat:Mat = null;
  var irange:Mat = null;
  var coloffsets:Mat = null;
  var correction = 1f;

  override def forward = {
		val start = toc;
		val modelrows = inputData.nrows;
		val nfeats = if (opts.outdim == 0) inputData.nrows else opts.outdim;
		if (correction.asInstanceOf[AnyRef] == null) correction = 1f * nfeats / opts.nsamps;
    if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
      modelmats(imodel) = convertMat(normrnd(0, 1, modelrows + (if (opts.hasBias) 1 else 0), nfeats));
      updatemats(imodel) = modelmats(imodel).zeros(modelmats(imodel).nrows, nfeats);  
    }
    if (opts.aopts != null && !ADAinitialized) initADAGrad;
    if (randwords.asInstanceOf[AnyRef] == null) randwords = convertMat(zeros(opts.nsamps + 1, inputData.ncols));
    if (iexpt.asInstanceOf[AnyRef] == null) iexpt = convertMat(row(1f/(1f-opts.expt)));
    if (onerow.asInstanceOf[AnyRef] == null) onerow = convertMat(ones(1, inputData.ncols));
    val mm = modelmats(imodel); 
    inputMat = if (opts.hasBias) (inputData on onerow) else inputData;
    
    rand(randwords);                                                          // Compute some random negatives
    val irandwords = min(nfeats-2, int((nfeats - 1) * (randwords ^ iexpt)));  // produce power-law values with exponent expt
    irandwords ~ irandwords + (irandwords >= target);                         // remove targets as possible negative samples
    irandwords(opts.nsamps, ?) = target;
    
    val indmat = nHot(irandwords, nfeats);
    prods = DDS(mm, inputMat, indmat);
    output = prods.contents.view(opts.nsamps+1, inputData.ncols);

    output ~ output - maxi(output)
    exp(output, output);  // ensures sum(exps) is between 1 and nfeats
    if (opts.docorrect) {
      output(opts.nsamps, ?) = output(opts.nsamps, ?) * (1/correction);
    }
    val sout = sum(output);
    output ~ output / sout;
    forwardtime += toc - start;
  }

  override def backward = {
		val start = toc;
		val modelrows = inputData.nrows;
		val nfeats = if (opts.outdim == 0) inputData.nrows else opts.outdim;
		if (targMat.asInstanceOf[AnyRef] == null) targMat = convertMat(zeros(opts.nsamps, inputData.ncols) on ones(1, inputData.ncols));
		val mm = modelmats(imodel); 
		val um = updatemats(imodel);
		
		deriv = targMat - output;
		prods.contents <-- deriv.contents;  
		inputMat.madd(prods, um, false, true);
		if (inputDeriv.asInstanceOf[AnyRef] != null) {
			if (opts.hasBias) {
				inputMat ~ mm * prods;
				if (irange.asInstanceOf[AnyRef] == null) irange = convertMat(icol(0->inputData.nrows));
				inputDeriv ~ inputDeriv + inputMat(irange, ?);
			} else {
				mm.madd(prods, inputDeriv);
			}
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
  
  override def score:FMat = {
    if (opts.scoreType < 2) {
      opts.scoreType match {
        case 0 => FMat(mean(ln(output(opts.nsamps, ?))));
        case 1 => FMat(mean(output(opts.nsamps, ?) == maxi(output)));
      }    	   
    } else {
      val mprod = modelmats(imodel) ^* inputMat;
      mprod ~ mprod - maxi(mprod);
      exp(mprod, mprod);
      mprod ~ mprod / sum(mprod);
      if (coloffsets.asInstanceOf[AnyRef] == null) coloffsets = convertMat(irow(0->mprod.ncols)*mprod.nrows);
      val inds = target + coloffsets;
      opts.scoreType match {
        case 2 => FMat(mean(ln(mprod(inds))));
        case 3 => FMat(mean(mprod(inds) == maxi(mprod)));        
      }
    }
  }
}

trait NegsampOutputNodeOpts extends ModelNodeOpts {  
 
    var nsamps = 100;
    var hasBias:Boolean = false;
    var aopts:ADAGrad.Opts = null;
    var outdim = 0; 
    var scoreType = 0;
    var expt = 0.5;
    var docorrect = true;
    
   def copyOpts(opts:NegsampOutputNodeOpts):NegsampOutputNodeOpts = {
			super.copyOpts(opts);
			opts.nsamps = nsamps;
			opts.hasBias = hasBias;
			opts.aopts = aopts;
			opts.outdim = outdim;
			opts.expt = expt;
			opts.scoreType = scoreType;
			opts;
	}
}

class NegsampOutputNode extends ModelNode with NegsampOutputNodeOpts {
  
  def copyTo(opts:NegsampOutputNode):NegsampOutputNode = {
    this.asInstanceOf[ModelNode].copyTo(opts);
    copyOpts(opts);
    opts
  }

	override def clone:NegsampOutputNode = {copyTo(new NegsampOutputNode).asInstanceOf[NegsampOutputNode];}

	override def create(net:Net):NegsampOutputLayer = {NegsampOutputLayer(net, this);}
}
  
object NegsampOutputLayer {
  
  def apply(net:Net) = new NegsampOutputLayer(net, new NegsampOutputNode);
  
  def apply(net:Net, opts:NegsampOutputNode) = new NegsampOutputLayer(net, opts);
}