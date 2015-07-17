package BIDMach.networks

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach._
import scala.util.hashing.MurmurHash3;
import scala.collection.mutable.HashMap;

/**
 * Basic DNN class. Learns a supervised map from input blocks to output (target) data blocks. There are currently 15 layer types:
 - InputLayer: just a placeholder for the first layer which is loaded with input data blocks. No learnable params. 
 - FCLayer: Fully-Connected Linear layer. Has a matrix of learnable params which is the input-output map. 
 - RectLayer: Rectifying one-to-one layer. No params.
 - GLMLayer: a one-to-one layer with GLM mappings (linear, logistic, abs-logistic and SVM). No learnable params. 
 - NormLayer: normalizing layer that adds a derivative term based on the difference between current layer norm and a target norm. 
   No learnable params. The target norm and weight of this derivative term can be specified. 
 - DropoutLayer: A layer that implements random dropout. No learnable params, but dropout fraction can be specified. 
 - AddLayer: adds input layers element-wise.
 - MulLayer: multiplies input layers element-wise. Will also perform edge operations (one input can be a scalar). 
 - Softmax: a softmax (normalized exponential) layer.
 - Tanh: Hyperbolic tangent non-linearity.
 - Sigmoid: Logistic function non-linearity.
 - Softplus: smooth ReLU unit. 
 - Ln: natural logarithm
 - Exp: exponential
 - Sum: column-wise sum
 *
 * The network topology is specified by opts.layers which is a sequence of "LayerSpec" objects. There is a LayerSpec
 * Class for each Layer class, which holds the params for defining that layer. Currently only four LayerSpec types need params:
 - FC: "outside" holds the output dimensions of the FClayer (input dimension set by previous layer). 
 - GLM: "links" holds the links matrix (integer specs for loss types, see GLM), for the output of that layer. Its size should match the number of targets.
 - Norm: "targetNorm" holds a target per-element norm, and "weight" is the weight for this term in the derivative calculation.
 - Dropout: "frac" holds the fraction of neurons to retain.
 *
 * Each LayerSpec instance has up to two inputs which are other LayerSpec instances (or null). This graph structure can be cyclic. 
 * When the model is created, the Layer structure mimics the LayerSpec structure. 
 */

class DNN(override val opts:DNN.Opts = new DNN.Options) extends Model(opts) {
  var layers:Array[Layer] = null;
  var targmap:Mat = null;
  var mask:Mat = null;
  var bufmat:Mat = null;
  var modelMap:HashMap[String,Int] = null;
  var batchSize = -1;
  var imodel = 0;

  override def init() = {
	  mats = datasource.next;
	  var nfeats = mats(0).nrows;
	  datasource.reset;
	  targmap = if (opts.targmap.asInstanceOf[AnyRef] != null) convertMat(opts.targmap) else null;
	  mask = if (opts.dmask.asInstanceOf[AnyRef] != null) convertMat(opts.dmask) else null;
	  layers = new Array[Layer](opts.layers.length);
	  modelMap = HashMap();
	  imodel = 0;
	  if (refresh) {
	    val nm = opts.nmodelmats + opts.layers.map(_ match {case x:DNN.ModelLayerSpec => x.nmodels; case _ => 0}).reduce(_+_);
	    setmodelmats(new Array[Mat](nm));
	  }
	  for (i <- 0 until opts.layers.length) {
	  	layers(i) = opts.layers(i).factory(this);
	  	opts.layers(i).myLayer = layers(i);
	  }
	  updatemats = new Array[Mat](modelmats.length);
	  for (i <- 0 until modelmats.length) {
	    if (modelmats(i).asInstanceOf[AnyRef] != null) modelmats(i) = convertMat(modelmats(i));
	  } 		
	  for (i <- 0 until opts.layers.length) {
	  	if (opts.layers(i).input.asInstanceOf[AnyRef] != null) layers(i).input = opts.layers(i).input.myLayer.asInstanceOf[DNN.this.Layer];
	  	if (opts.layers(i).inputs != null) {
	  	  (0 until opts.layers(i).inputs.length).map((j:Int) => layers(i).inputs(j) = opts.layers(i).inputs(j).myLayer.asInstanceOf[DNN.this.Layer]);
	  	} 
	  }
	  if (useGPU) copyMats(mats, gmats);
	  val pb = putBack;
	  putBack = -1;
    evalbatch(gmats, 0, 0);
    putBack = pb;
	  datasource.reset;
  }
  
  
  def dobatch(gmats:Array[Mat], ipass:Int, pos:Long):Unit = {
    if (batchSize < 0) batchSize = gmats(0).ncols;
    if (batchSize == gmats(0).ncols) {                                    // discard odd-sized minibatches
    	layers(0).data = gmats(0);
    	if (targmap.asInstanceOf[AnyRef] != null) {
    		layers(layers.length-1).target = targmap * gmats(0);
    	} else {
    		layers(layers.length-1).target = gmats(1);
    	}
    	if (mask.asInstanceOf[AnyRef] != null) {
    		modelmats(0) ~ modelmats(0) ∘ mask;
    	}
    	var i = 1;
    	while (i < layers.length) {
    		layers(i).forward;
    		i += 1;
    	}
      layers(layers.length-1).deriv.set(1);   	
    	for (j <- 0 until updatemats.length) updatemats(j).clear;
    	while (i > 1) {
    		i -= 1;
    		layers(i).backward(ipass, pos);
    	}
    	if (mask.asInstanceOf[AnyRef] != null) {
    		updatemats(0) ~ updatemats(0) ∘ mask;
    	}
    }
  }
  
  def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat = {  
  	if (batchSize < 0) batchSize = gmats(0).ncols;
    layers(0).data = extendData(gmats(0), batchSize);
    val targ = extendData(if (targmap.asInstanceOf[AnyRef] != null && putBack < 0) {
    	targmap * gmats(0);
    } else {
    	gmats(1);
    }, batchSize);
    layers(layers.length-1).target = targ;
    if (mask.asInstanceOf[AnyRef] != null) {
    	modelmats(0) ~ modelmats(0) ∘ mask;
    }
    var i = 1;
    while (i < layers.length) {
    	layers(i).forward;
    	i += 1;
    }
    if (putBack >= 0) {
    	layers(layers.length-1).data.colslice(0, gmats(0).ncols, gmats(1));
    }
    layers(layers.length-1).score
  }
  
  /* 
   * Deal with annoying sub-sized minibatches
   */
  
  def extendData(mat:Mat, batchSize:Int):Mat = {
    val nrows = mat.nrows;
    val ncols = mat.ncols;
    val bsize = batchSize - ncols;
    if (bsize > 0) {
    	val newGUID = MurmurHash3.mix(MurmurHash3.mix((mat.GUID >> 32).toInt, mat.GUID.toInt),"extendData".##);
    	mat match {
    	case a:FMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = zeros(nrows, bsize); a \ bufmat}
    	case a:DMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = dzeros(nrows, bsize); a \ bufmat}
    	case a:IMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = izeros(nrows, bsize); a \ bufmat}
    	case a:LMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = lzeros(nrows, bsize); a \ bufmat}
    	case a:GMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = gzeros(nrows, bsize); a \ bufmat}
    	case a:GDMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = gdzeros(nrows, bsize); a \ bufmat}
    	case a:GIMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = gizeros(nrows, bsize); a \ bufmat}   
    	case a:GLMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = glzeros(nrows, bsize); a \ bufmat}
    	case a:SMat => {val b = new SMat(nrows, ncols, a.nnz, a.ir, a.jc, a.data); b.setGUID(newGUID); b}
    	case a:SDMat => {val b = new SDMat(nrows, ncols, a.nnz, a.ir, a.jc, a.data); b.setGUID(newGUID); b}
    	case a:GSMat => {val b = new GSMat(nrows, ncols, a.nnz, a.ir, a.ic, a.jc, a.data, a.realnnz); b.setGUID(newGUID); b}
    	case a:GSDMat => {val b = new GSDMat(nrows, ncols, a.nnz, a.ir, a.ic, a.jc, a.data, a.realnnz); b.setGUID(newGUID); b}
    	}
    } else {
      mat;
    }
  }
  
  
  class Layer {
    var data:Mat = null;
    var target:Mat = null;
    var deriv:Mat = null;
    var input:Layer = null;
    var inputs:Array[Layer] = null;
    def forward = {};
    def backward:Unit = {};
    def backward(ipass:Int, pos:Long):Unit = backward;
    def score:FMat = zeros(1,1);
    
    def createData = {
      if (data.asInstanceOf[AnyRef] == null) data = input.data.zeros(input.data.nrows, input.data.ncols);
    }
    
    def createData(nrows:Int, ncols:Int) = {
      if (data.asInstanceOf[AnyRef] == null) data = input.data.zeros(nrows, ncols);
    }
    
    def clearDeriv = {
    	if (deriv.asInstanceOf[AnyRef] == null) deriv = data.zeros(data.nrows, data.ncols);
    	deriv.clear;
    }
  }
  
  /**
   * Full connected layer. 
   * Includes a model matrix that contains the linear map. 
   */
  
  class FCLayer(val imodel:Int, val constFeat:Boolean, val aopts:ADAGrad.Opts, val outdim:Int) extends Layer {
  	var vexp:Mat = null;
    var texp:Mat = null;
    var lrate:Mat = null;
    var sumsq:Mat = null;
    var firststep = -1f;
    var waitsteps = 0;
    var epsilon = 0f;
    
    override def forward = {
      if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
        modelmats(imodel) = convertMat(normrnd(0, 1, outdim, input.data.nrows + (if (constFeat) 1 else 0)));
        updatemats(imodel) = modelmats(imodel).zeros(modelmats(imodel).nrows, modelmats(imodel).ncols);
        if (aopts != null) initADAGrad;  
      }
    	val mm = if (constFeat) {
    		modelmats(imodel).colslice(1, modelmats(imodel).ncols);
    	} else {
    		modelmats(imodel);
    	}
    	createData(mm.nrows, input.data.ncols);
    	data ~ mm * input.data;
    	if (constFeat) data ~ data + modelmats(imodel).colslice(0, 1);
    	clearDeriv;
    }
    
    override def backward(ipass:Int, pos:Long) = {
    	val mm = if (constFeat && imodel > 0) {
    		modelmats(imodel).colslice(1, modelmats(imodel).ncols);
    	} else {
    		modelmats(imodel);
    	}
      if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + (mm ^* deriv);
      if (aopts != null) {
        if (firststep <= 0) firststep = pos.toFloat;
        val istep = (pos + firststep)/firststep;
      	ADAGrad.multUpdate(deriv, input.data, modelmats(imodel), sumsq, mask, lrate, texp, vexp, epsilon, istep, waitsteps);
      } else {
      	val dprod = deriv *^ input.data;
      	updatemats(imodel) ~ updatemats(imodel) + (if (constFeat) (sum(deriv,2) \ dprod) else dprod);
      }
    }

    
    def initADAGrad {
      val mm = modelmats(imodel); 
      val d = mm.nrows;
      val m = mm.ncols;
    	firststep = -1f;
    	lrate = convertMat(aopts.lrate);
    	texp = convertMat(aopts.texp);
    	vexp = convertMat(aopts.vexp);
    	sumsq = convertMat(zeros(d, m));
    	sumsq.set(aopts.initsumsq);
    	waitsteps = aopts.waitsteps;
    	epsilon = aopts.epsilon;
    }
  }
  
  /**
   * Rectifying Linear Unit layer.
   */
  
  class ReLULayer extends Layer {
    override def forward = {
      createData;
      data <-- max(input.data, 0f);
      clearDeriv;
    }
    
    override def backward = {
      if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + (deriv ∘ (input.data > 0f));
    }
  }
  
  /**
   * Input layer is currently just a placeholder.
   */
  
  class InputLayer extends Layer {
  }
  
  /**
   * GLMLayer implements linear, logistic and hinge-loss SVM. 
   * Commonly used as an output layer so includes a score method.
   */
  
  class GLMLayer(val links:IMat) extends Layer {
    val ilinks = if (useGPU) GIMat(links) else links;
    var totflops = 0L;
    for (i <- 0 until links.length) {
      totflops += GLM.linkArray(links(i)).fnflops
    }
    
    override def forward = {
      createData;
      data <-- GLM.preds(input.data, ilinks, totflops);
      clearDeriv;
    }
    
    override def backward = {
      if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + (deriv ∘ GLM.derivs(data, target, ilinks, totflops));
    }
    
    override def score:FMat = { 
      val v = GLM.llfun(data, target, ilinks, totflops);
      FMat(mean(v, 2));
    }
  }
  
  /**
   * Normalization layer adds a downward-propagating derivative term whenever its norm 
   * is different from the specified value (targetNorm).
   */
    
  class NormLayer(val targetNorm:Float, val weight:Float) extends Layer {
    var sconst:Mat = null
    
    override def forward = {
      createData;
      data <-- input.data;
      clearDeriv;
    }
    
    override def backward = {
      if (input.deriv.asInstanceOf[AnyRef] != null) {
      	if (sconst.asInstanceOf[AnyRef] == null) sconst = data.zeros(1,1);
      	sconst.set(math.min(0.1f, math.max(-0.1f, (targetNorm - norm(data)/data.length).toFloat * weight)));
      	input.deriv = data ∘ sconst;
      	input.deriv ~ input.deriv + deriv;
      }
    }
  }
  
  /**
   * Dropout layer with fraction to keep "frac". Deletes the same neurons in forward and backward pass. 
   * Assumes that "randmat" is not changed between forward and backward passes. 
   */
  
  class DropoutLayer(val frac:Float) extends Layer {  
    var randmat:Mat = null;
    
    override def forward = {
      createData;
      randmat = input.data + 20f;   // Hack to make a cached container to hold the random data
      if (opts.predict) {
      	data ~ input.data * frac;
      } else {
      	if (useGPU) {
      		grand(randmat.asInstanceOf[GMat]); 
      	} else {
      		rand(randmat.asInstanceOf[FMat]);
      	}
      	randmat ~ (randmat < frac)
      	data ~ input.data ∘ randmat;
      }
      clearDeriv;
    }
    
    override def backward = {
      if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + deriv ∘ randmat;
    }
  }
  
  /**
   * Computes the sum of input layers. 
   */
  
  class AddLayer extends Layer { 
    
    override def forward = {
      createData(inputs(0).data.nrows, inputs(0).data.ncols);
      data <-- inputs(0).data;
      (1 until inputs.length).map((i:Int) => data ~ data + inputs(i).data);
      clearDeriv;
    }
    
    override def backward = {
      (0 until inputs.length).map((i:Int) => {
      	if (inputs(i).deriv.asInstanceOf[AnyRef] != null) inputs(i).deriv ~ inputs(i).deriv + deriv
      });
    }
  }
  
  /**
   * Computes the product of its input layers. 
   */
  
  class MulLayer extends Layer {  
    
    override def forward = {
    	createData(inputs(0).data.nrows, inputs(0).data.ncols);
      data <-- inputs(0).data;
      (1 until inputs.length).map((i:Int) => data ~ data ∘ inputs(i).data);
      clearDeriv;
    }
    
    override def backward = {
      val ddata = deriv ∘ data;
      (0 until inputs.length).map((i:Int) => {
      	if (inputs(i).deriv.asInstanceOf[AnyRef] != null) inputs(i).deriv ~ inputs(i).deriv + ddata / inputs(i).data;
      });
    }
  }
  
  /**
   * Softmax layer. Output = exp(input) / sum(exp(input))
   */
  
  class SoftmaxLayer extends Layer {    
    
    override def forward = {
      createData;
      val exps = exp(input.data);
      data ~ exps / sum(exps);
      clearDeriv;
    }
    
    override def backward = {
      val exps = exp(input.data);
      val sumexps = sum(exps);
      val isum = 1f / (sumexps ∘ sumexps);
      if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~
        input.deriv + ((exps / sumexps) ∘ deriv) - (exps ∘ (isum ∘ (exps ∙ deriv))) ;
    }
  }
  
  /**
   * Tanh layer. 
   */
  
  class TanhLayer extends Layer {    
    
    override def forward = {
      createData;
      tanh(input.data, data);
      clearDeriv;
    }
    
    override def backward = {
      val tmp = tanh(input.data);
      if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + (1 - tmp ∘ tmp) ∘ deriv;
    }
  }
  
  /**
   * Sigmoid layer. Uses GLM implementations of logistic functions for performance. 
   */
  
  class SigmoidLayer extends Layer {
    var ilinks:Mat = null;
    var totflops = 0L;
    
    override def forward = {
      createData;
      if (ilinks.asInstanceOf[AnyRef] == null) {
        ilinks = izeros(input.data.nrows, 1)
        ilinks.set(GLM.logistic);
        if (useGPU) ilinks = GIMat(ilinks);
      }
      if (totflops == 0L) totflops = input.data.nrows * GLM.linkArray(1).fnflops
      data <-- GLM.preds(input.data, ilinks, totflops)
      clearDeriv;
    }
    
    override def backward = {
      val tmp = data - (data ∘ data);
      if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + tmp ∘ deriv;
    }
  }
  
  /**
   * Softplus layer.  
   */
  
  class SoftplusLayer extends Layer {
  	var ilinks:Mat = null;
    var totflops = 0L;
   
    override def forward = {
      createData;
      val big = input.data > 60f;      
      data ~ ((1 - big) ∘ ln(1f + exp(min(60f, input.data)))) + big ∘ input.data;
      clearDeriv;
    }
    
    override def backward = {
      if (ilinks.asInstanceOf[AnyRef] == null) {
        ilinks = izeros(input.data.nrows, 1)
        ilinks.set(GLM.logistic);
      }
      if (totflops == 0L) totflops = input.data.nrows * GLM.linkArray(1).fnflops;
      if (useGPU) ilinks = GIMat(ilinks);
      if (input.deriv.asInstanceOf[AnyRef] != null) {
      	val tmp = GLM.preds(input.data, ilinks, totflops);
      	input.deriv ~ input.deriv + tmp ∘ deriv;
      }
    }
  }
  
  /**
   * Natural Log layer. 
   */
  
  class LnLayer extends Layer {
    
    override def forward = {
      createData;
      ln(input.data, data);
      clearDeriv;
    }
    
    override def backward = {
      if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + deriv/input.data;    
    }
  }
  
  /**
   * Exponential layer. 
   */
  
  class ExpLayer extends Layer {
    
    override def forward = {
      createData;
      exp(input.data, data);
      clearDeriv;
    }
    
    override def backward = {
      if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + deriv ∘ data;    
    }
  }
  
  /**
   * Sum layer. 
   */
  
  class SumLayer extends Layer {
    var vmap:Mat = null;
    
    override def forward = {
      createData(1, input.data.ncols);
      data <-- sum(input.data);
      clearDeriv;
    }
    
    override def backward = {
      if (vmap.asInstanceOf[AnyRef] == null) vmap = deriv.ones(data.nrows, 1);
      if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + vmap * deriv;    
    }
  }
  /**
   * Creates a copy of the input grown by a small piece of the last minibatch to support lagged updates
   * e.g. for word2vec
   */
  
  class LagLayer(siz:Int) extends Layer {
    var lastBatch:Mat = null;
    
    override def forward = {
      createData(input.data.nrows, input.data.ncols + siz);
      data.colslice(input.data.ncols, input.data.ncols+siz, data);
      input.data.colslice(0, input.data.ncols, data, siz);
    }
  }
  
  /**
   * Block matrix-matrix multiply. Each output block is nr x nc. nc needs to be a submultiple of the minibatch size. 
   * Each element of the block moves by step in the corresponding matrix. 
   */
  
  class blockGemmLayer(nr:Int, nc:Int, step:Int, reps:Int, inshift:Int) extends Layer {
  	val aspect = nr / nc;
  	val astep = if (step == 1) nc else 1;
  	val shift0 = if (inshift >= 0) inshift else 0;
  	val shift1 = if (inshift < 0) - inshift else 0;
  	
    override def forward = {
      val nrows = inputs(0).data.nrows;
      val nrowsa = nrows * aspect;
      if (data.asInstanceOf[AnyRef] == null) data = inputs(0).data.zeros(nr, nc * reps);
      
      inputs(0).data.blockGemm(1, 0, nr, nc, reps, 
                          shift0*nrowsa, step*nrowsa, astep*nrowsa, 
      		inputs(1).data, shift1*nrows,  step*nrows,  astep*nrows, 
      		data,           0,             step*nr,     astep*nr);      
    }
    
    override def backward = {
      val nrows = inputs(0).data.nrows;
      val nrowsa = nrows * aspect;
      if (inputs(0).deriv.asInstanceOf[AnyRef] == null) inputs(0).deriv = inputs(0).data.zeros(nrows, inputs(0).data.ncols);
      if (inputs(1).deriv.asInstanceOf[AnyRef] == null) inputs(1).deriv = inputs(1).data.zeros(nrows, inputs(1).data.ncols);
      
      inputs(1).data.blockGemm(0, 1, nrows, nc, reps, 
      		                 shift1*nrows,  step*nrows,  astep*nrows, 
      		deriv,           0,             step*nr,     astep*nr, 
      		inputs(0).deriv, shift0*nrowsa, step*nrowsa, astep*nrowsa);
      
      inputs(0).data.blockGemm(0, 0, nrows, nr, reps, 
      		                 shift0*nrowsa, step*nrowsa, astep*nrowsa, 
      		deriv,           0,             step*nr,     astep*nr, 
      		inputs(1).deriv, shift1*nrows,  step*nrows,  astep*nrows);
    
    }
  }
}

object DNN  {
  trait Opts extends Model.Opts {
	  var layers:Seq[LayerSpec] = null;
    var links:IMat = null;
    var nweight:Float = 0.1f;
    var dropout:Float = 0.5f;
    var predict:Boolean = false;
    var targetNorm:Float = 1f;
    var targmap:Mat = null;
    var dmask:Mat = null;
    var constFeat:Boolean = false;
    var aopts:ADAGrad.Opts = null;
    var nmodelmats = 0;
  }
  
  class Options extends Opts {}
  
  abstract class LayerSpec(var input:LayerSpec, val inputs:Array[LayerSpec]) {
    var myLayer:DNN#Layer = null;
    def factory(net:DNN):net.Layer
  }
  
  abstract class ModelLayerSpec(input:LayerSpec, val modelName:String, val imodel:Int, val nmodels:Int) extends LayerSpec(input, null) {}
  
  class FC(input:LayerSpec, val outsize:Int, override val modelName:String, val constFeat:Boolean = false, val aopts:ADAGrad.Opts = null, override val imodel:Int = 0) extends ModelLayerSpec(input, modelName, imodel, 1) {
  	override def factory(net:DNN) = {
  		val thismodel = if (net.opts.nmodelmats > 0) {   // If explicit model numbers are given, use them. 
  			imodel 
  		} else if (modelName.length > 0) {               // If this is a named layer, look it up. 
  			if (net.modelMap.contains(modelName)) {
  				net.modelMap(modelName);
  			} else {
  				val len = net.modelMap(modelName).length;
  				net.modelMap(modelName) = len + net.opts.nmodelmats; 	
  				len;
  			}
  		} else {                                         // Otherwise return the next available int
  		  net.imodel += 1;
  		  net.imodel - 1;
  		}
  		val fclayer = new net.FCLayer(thismodel, constFeat, aopts, outsize);
  		fclayer;
  	}
  }
  
  class ReLU(input:LayerSpec) extends LayerSpec(input, null) {
    override def factory(net:DNN) = new net.ReLULayer;
  }
  
  class Input extends LayerSpec(null, null) {
    override def factory(net:DNN) = new net.InputLayer;
  }
  
  class GLM(input:LayerSpec, val links:IMat) extends LayerSpec(input, null) {
    override def factory(net:DNN) = new net.GLMLayer(links);
  }
  
  class Norm(input:LayerSpec, val targetNorm:Float, val weight:Float) extends LayerSpec(input, null) {
    override def factory(net:DNN) = new net.NormLayer(targetNorm, weight);
  }
  
  class Dropout(input:LayerSpec, val frac:Float) extends LayerSpec(input, null) {
    override def factory(net:DNN) = new net.DropoutLayer(frac);
  }
  
  class Add(inputs:Array[LayerSpec]) extends LayerSpec(null, inputs) {
    override def factory(net:DNN) = new net.AddLayer;
  }
  
  class Mul(inputs:Array[LayerSpec]) extends LayerSpec(null, inputs) {
    override def factory(net:DNN) = new net.MulLayer;
  }
  
  class Softmax(input:LayerSpec) extends LayerSpec(input, null) {
    override def factory(net:DNN) = new net.SoftmaxLayer;
  }
  
  class Tanh(input:LayerSpec) extends LayerSpec(input, null) {
    override def factory(net:DNN) = new net.TanhLayer;
  }
  
  class Sigmoid(input:LayerSpec) extends LayerSpec(input, null) {
    override def factory(net:DNN) = new net.SigmoidLayer;
  }
  
  class Softplus(input:LayerSpec) extends LayerSpec(input, null) {
    override def factory(net:DNN) = new net.SoftplusLayer;
  }
  
  class Ln(input:LayerSpec) extends LayerSpec(input, null) {
  	override def factory(net:DNN) = new net.LnLayer;
  }
  
  class Exp(input:LayerSpec) extends LayerSpec(input, null) {
    override def factory(net:DNN) = new net.ExpLayer;
  }
  
  class Sum(input:LayerSpec) extends LayerSpec(input, null) {
    override def factory(net:DNN) = new net.SumLayer;
  }
  
  class Lag(input:LayerSpec, val siz:Int) extends LayerSpec(input, null) {
    override def factory(net:DNN) = new net.LagLayer(siz);
  }
  
  /**
   * Build a stack of layer specs. layer(0) is an input layer, layer(n-1) is a GLM layer. 
   * Intermediate layers are FC alternating with ReLU, starting and ending with FC. 
   * First FC layer width is given as an argument, then it tapers off by taper.
   */
  
  def dlayers(depth0:Int, width:Int, taper:Float, ntargs:Int, opts:Opts, nonlin:Int = 1):Array[LayerSpec] = {
    val depth = (depth0/2)*2 + 1;              // Round up to an odd number of layers 
    val layers = new Array[LayerSpec](depth);
    var w = width;
    layers(0) = new Input;
    for (i <- 1 until depth - 2) {
    	if (i % 2 == 1) {
    		layers(i) = new FC(layers(i-1), w, "", opts.constFeat, opts.aopts);
    		w = (taper*w).toInt;
    	} else {
    	  nonlin match {
    	    case 1 => layers(i) = new Tanh(layers(i-1));
    	    case 2 => layers(i) = new Sigmoid(layers(i-1));
    	    case 3 => layers(i) = new ReLU(layers(i-1));
    	    case 4 => layers(i) = new Softplus(layers(i-1));
    	  }
    	}
    }
    layers(depth-2) = new FC(layers(depth-3), ntargs, "", opts.constFeat, opts.aopts);
    layers(depth-1) = new GLM(layers(depth-2), opts.links);
    opts.layers = layers
    layers
  }
  
  /**
   * Build a stack of layer specs. layer(0) is an input layer, layer(n-1) is a GLM layer. 
   * Intermediate layers are FC, ReLU, Norm, starting and ending with FC. 
   * First FC layer width is given as an argument, then it tapers off by taper.
   */
  
  def dlayers3(depth0:Int, width:Int, taper:Float, ntargs:Int, opts:Opts, nonlin:Int = 1):Array[LayerSpec] = {
    val depth = (depth0/3)*3;              // Round up to a multiple of 3 
    val layers = new Array[LayerSpec](depth);
    var w = width;
    layers(0) = new Input;
    for (i <- 1 until depth - 2) {
    	if (i % 3 == 1) {
    		layers(i) = new FC(layers(i-1), w, "", opts.constFeat, opts.aopts);
    		w = (taper*w).toInt;
    	} else if (i % 3 == 2) {
    	  nonlin match {
    	    case 1 => layers(i) = new Tanh(layers(i-1));
    	    case 2 => layers(i) = new Sigmoid(layers(i-1));
    	    case 3 => layers(i) = new ReLU(layers(i-1));
    	    case 4 => layers(i) = new Softplus(layers(i-1));
    	  }
    	} else {
    	  layers(i) = new Norm(layers(i-1), opts.targetNorm, opts.nweight);
    	}
    }
    layers(depth-2) = new FC(layers(depth-3), ntargs, "", opts.constFeat, opts.aopts);
    layers(depth-1) = new GLM(layers(depth-2), opts.links);
    opts.layers = layers
    layers
  }
  
  /**
   * Build a stack of layer specs. layer(0) is an input layer, layer(n-1) is a GLM layer. 
   * Intermediate layers are FC, ReLU, Norm, Dropout, starting and ending with FC. 
   * First FC layer width is given as an argument, then it tapers off by taper.
   */
  
  def dlayers4(depth0:Int, width:Int, taper:Float, ntargs:Int, opts:Opts, nonlin:Int = 1):Array[LayerSpec] = {
    val depth = ((depth0+1)/4)*4 - 1;              // Round up to a multiple of 4 - 1
    val layers = new Array[LayerSpec](depth);
    var w = width;
    layers(0) = new Input;
    for (i <- 1 until depth - 2) {
      (i % 4) match {
        case 1 => {
        	layers(i) = new FC(layers(i-1), w, "", opts.constFeat, opts.aopts);
        	w = (taper*w).toInt;
        }
        case 2 => {
        	nonlin match {
        	case 1 => layers(i) = new Tanh(layers(i-1));
        	case 2 => layers(i) = new Sigmoid(layers(i-1));
        	case 3 => layers(i) = new ReLU(layers(i-1));
        	case 4 => layers(i) = new Softplus(layers(i-1));
        	}
        }
        case 3 => {
          layers(i) = new Norm(layers(i-1), opts.targetNorm, opts.nweight);
        }
        case _ => {
          layers(i) = new Dropout(layers(i-1), opts.dropout);          
        }
      }
    }
    layers(depth-2) = new FC(layers(depth-3), ntargs, "", opts.constFeat, opts.aopts);
    layers(depth-1) = new GLM(layers(depth-2), opts.links);
    opts.layers = layers
    layers
  }
  
  def mkDNNModel(fopts:Model.Opts) = {
    new DNN(fopts.asInstanceOf[DNN.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
    new ADAGrad(nopts.asInstanceOf[ADAGrad.Opts])
  } 
  
  def mkRegularizer(nopts:Mixin.Opts):Array[Mixin] = {
    Array(new L1Regularizer(nopts.asInstanceOf[L1Regularizer.Opts]))
  }
    
  class LearnOptions extends Learner.Options with DNN.Opts with MatDS.Opts with ADAGrad.Opts with L1Regularizer.Opts

  def learner(mat0:Mat, targ:Mat) = {
    val opts = new LearnOptions
    if (opts.links == null) opts.links = izeros(1,targ.nrows)
    opts.links.set(1)
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    dlayers(3, 0, 1f, targ.nrows, opts)                   // default to a 3-layer network
  	val nn = new Learner(
  	    new MatDS(Array(mat0, targ), opts), 
  	    new DNN(opts), 
  	    Array(new L1Regularizer(opts)),
  	    new ADAGrad(opts), 
  	    opts)
    (nn, opts)
  }
  
  def learnerX(mat0:Mat, targ:Mat) = {
    val opts = new LearnOptions
    if (opts.links == null) opts.links = izeros(1,targ.nrows)
    opts.links.set(1)
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    dlayers(3, 0, 1f, targ.nrows, opts)                   // default to a 3-layer network
  	val nn = new Learner(
  	    new MatDS(Array(mat0, targ), opts), 
  	    new DNN(opts), 
  	    null,
  	    null, 
  	    opts)
    (nn, opts)
  }
  
  class FDSopts extends Learner.Options with DNN.Opts with FilesDS.Opts with ADAGrad.Opts with L1Regularizer.Opts
  
  def learner(fn1:String, fn2:String):(Learner, FDSopts) = learner(List(FilesDS.simpleEnum(fn1,1,0),
  		                                                                  FilesDS.simpleEnum(fn2,1,0)));
  
  def learner(fn1:String):(Learner, FDSopts) = learner(List(FilesDS.simpleEnum(fn1,1,0)));

  def learner(fnames:List[(Int)=>String]):(Learner, FDSopts) = {   
    val opts = new FDSopts
    opts.fnames = fnames
    opts.batchSize = 100000;
    opts.eltsPerSample = 500;
    implicit val threads = threadPool(4);
    val ds = new FilesDS(opts)
//    dlayers(3, 0, 1f, targ.nrows, opts)                   // default to a 3-layer network
  	val nn = new Learner(
  			ds, 
  	    new DNN(opts), 
  	    Array(new L1Regularizer(opts)),
  	    new ADAGrad(opts), 
  	    opts)
    (nn, opts)
  } 
  
  def learnerX(fn1:String, fn2:String):(Learner, FDSopts) = learnerX(List(FilesDS.simpleEnum(fn1,1,0),
  		                                                                  FilesDS.simpleEnum(fn2,1,0)));
  
  def learnerX(fn1:String):(Learner, FDSopts) = learnerX(List(FilesDS.simpleEnum(fn1,1,0)));
  
  def learnerX(fnames:List[(Int)=>String]):(Learner, FDSopts) = {   
    val opts = new FDSopts
    opts.fnames = fnames
    opts.batchSize = 100000;
    opts.eltsPerSample = 500;
    implicit val threads = threadPool(4);
    val ds = new FilesDS(opts)
//    dlayers(3, 0, 1f, targ.nrows, opts)                   // default to a 3-layer network
  	val nn = new Learner(
  			ds, 
  	    new DNN(opts), 
  	    null,
  	    null, 
  	    opts)
    (nn, opts)
  }
    // This function constructs a learner and a predictor. 
  def learner(mat0:Mat, targ:Mat, mat1:Mat, preds:Mat):(Learner, LearnOptions, Learner, LearnOptions) = {
    val mopts = new LearnOptions;
    val nopts = new LearnOptions;
    mopts.lrate = 1f
    mopts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    mopts.autoReset = false
    if (mopts.links == null) mopts.links = izeros(targ.nrows,1)
    nopts.links = mopts.links
    nopts.batchSize = mopts.batchSize
    nopts.putBack = 1
    dlayers(3, 0, 1f, targ.nrows, mopts)                   // default to a 3-layer network
    val model = new DNN(mopts)
    val mm = new Learner(
        new MatDS(Array(mat0, targ), mopts), 
        model, 
        Array(new L1Regularizer(mopts)),
        new ADAGrad(mopts), mopts)
    val nn = new Learner(
        new MatDS(Array(mat1, preds), nopts), 
        model, 
        null,
        null, 
        nopts)
    (mm, mopts, nn, nopts)
  }
  
  def predictor(model0:Model, mat0:Mat, preds:Mat):(Learner, LearnOptions) = {
    val model = model0.asInstanceOf[DNN];
    val mopts = model.opts;
    val opts = new LearnOptions;
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    opts.links = mopts.links;
    opts.layers = mopts.layers;
    opts.addConstFeat = model.opts.asInstanceOf[DataSource.Opts].addConstFeat;
    opts.putBack = 1;
    opts.dropout = 1f;
    
    val newmod = new DNN(opts);
    newmod.refresh = false;
    newmod.copyFrom(model);
    val nn = new Learner(
        new MatDS(Array(mat0, preds), opts), 
        newmod, 
        null,
        null, 
        opts);
    (nn, opts)
  }
  
  class LearnParOptions extends ParLearner.Options with DNN.Opts with FilesDS.Opts with ADAGrad.Opts with L1Regularizer.Opts;
  
  def learnPar(fn1:String, fn2:String):(ParLearnerF, LearnParOptions) = {learnPar(List(FilesDS.simpleEnum(fn1,1,0), FilesDS.simpleEnum(fn2,1,0)))}
  
  def learnPar(fnames:List[(Int) => String]):(ParLearnerF, LearnParOptions) = {
    val opts = new LearnParOptions;
    opts.batchSize = 10000;
    opts.lrate = 1f;
    opts.fnames = fnames;
    implicit val threads = threadPool(4)
    val nn = new ParLearnerF(
        new FilesDS(opts), 
        opts, mkDNNModel _,
        opts, mkRegularizer _,
        opts, mkUpdater _, 
        opts)
    (nn, opts)
  }
}