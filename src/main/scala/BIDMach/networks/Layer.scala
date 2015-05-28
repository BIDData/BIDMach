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
 * Basic Net Layer class. There are currently 15 layer types:
 - InputLayer: just a placeholder for the first layer which is loaded with input output blocks. No learnable params. 
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


  
class Layer(val net:Net, val opts:Layer.Opts = new Layer.Options) {
  val inputs = new Array[Layer](1);
  val outputs = new Array[Mat](1);
  val derivs = new Array[Mat](1);
  def input = inputs(0);
  def output = outputs(0);
  def deriv = derivs(0);
  var target:Mat = null;
  def forward = {};
  def backward:Unit = {};
  def backward(ipass:Int, pos:Long):Unit = backward;
  def score:FMat = zeros(1,1);
  def modelmats = net.modelmats;
  def updatemats = net.updatemats;
  def convertMat(mat:Mat) = net.convertMat(mat);
  def useGPU = net.useGPU;
  def nopts = net.opts;

  def createoutput = {
  		if (output.asInstanceOf[AnyRef] == null) outputs(0) = input.output.zeros(input.output.nrows, input.output.ncols);
  }

  def createoutput(nrows:Int, ncols:Int) = {
  	if (output.asInstanceOf[AnyRef] == null) outputs(0) = input.output.zeros(nrows, ncols);
  }

  def clearDeriv = {
  	if (deriv.asInstanceOf[AnyRef] == null) derivs(0) = output.zeros(output.nrows, output.ncols);
  	deriv.clear;
  }
}


object Layer {  
  trait Opts{
  }
  class Options extends Opts {}
}

class ModelLayer(override val net:Net, override val opts:ModelLayer.Opts = new ModelLayer.Options) extends Layer(net, opts) {
	var modelName = "";
	var imodel = 0;
  
  def getModelMat(net:Net, modelName:String, imodel:Int):Int = {
		if (net.opts.nmodelmats > 0) {   // If explicit model numbers are given, use them. 
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
  }
}

object ModelLayer {  
  trait Opts extends Layer.Opts{
  	var modelName = "";
    var imodel = 0;
  }
  class Options extends Opts {}
}
/**
 * Linear layer. 
 * Includes a model matrix that contains the linear map. 
 */

class LinLayer(override val net:Net, override val opts:LinLayer.Opts = new LinLayer.Options) extends ModelLayer(net, opts) {
	var vexp:Mat = null;
  var texp:Mat = null;
  var lrate:Mat = null;
  var sumsq:Mat = null;
  var mask:Mat = null;
  var firststep = -1f;
  var waitsteps = 0;
  var epsilon = 0f;

  override def forward = {
  	if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
  		modelmats(imodel) = convertMat(normrnd(0, 1, opts.outdim, input.output.nrows + (if (opts.constFeat) 1 else 0)));
  		updatemats(imodel) = modelmats(imodel).zeros(modelmats(imodel).nrows, modelmats(imodel).ncols);
  		if (opts.aopts != null) initADAGrad;  
  	}
  	val mm = if (opts.constFeat) {
  		modelmats(imodel).colslice(1, modelmats(imodel).ncols);
  	} else {
  		modelmats(imodel);
  	}
  	createoutput(mm.nrows, input.output.ncols);
  	output ~ mm * input.output;
  	if (opts.constFeat) output ~ output + modelmats(imodel).colslice(0, 1);
  	clearDeriv;
  }

  override def backward(ipass:Int, pos:Long) = {
  	val mm = if (opts.constFeat && imodel > 0) {
  		modelmats(imodel).colslice(1, modelmats(imodel).ncols);
  	} else {
  		modelmats(imodel);
  	}
  	if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + (mm ^* deriv);
  	if (opts.aopts != null) {
  		if (firststep <= 0) firststep = pos.toFloat;
  		val istep = (pos + firststep)/firststep;
  		ADAGrad.multUpdate(deriv, input.output, modelmats(imodel), sumsq, mask, lrate, texp, vexp, epsilon, istep, waitsteps);
  	} else {
  		val dprod = deriv *^ input.output;
  		updatemats(imodel) ~ updatemats(imodel) + (if (opts.constFeat) (sum(deriv,2) \ dprod) else dprod);
  	}
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
  	sumsq = convertMat(zeros(d, m));
  	sumsq.set(aopts.initsumsq);
  	waitsteps = aopts.waitsteps;
  	epsilon = aopts.epsilon;
  	mask = aopts.mask;
  }
}

object LinLayer {  
  trait Opts extends ModelLayer.Opts {
    var constFeat:Boolean = false;
    var aopts:ADAGrad.Opts = null;
    var outdim = 0;
  }
  class Options extends Opts {}
  
  def apply(net:Net) = new LinLayer(net, new Options);
  
  def apply(net:Net, opts:Opts) = new LinLayer(net, opts);
  
  def apply(in:Layer, net:Net, opts:Opts) = {
    val x = new LinLayer(net, opts); 
    x.inputs(0) = in; 
    x.modelName = opts.modelName;
    x.imodel = x.getModelMat(net, opts.modelName, opts.imodel)
    x;}
  
  def apply(in:Layer, net:Net, modelName:String, imodel:Int, outdim:Int, constFeat:Boolean, aopts:ADAGrad.Opts) = {
    val x = new LinLayer(net, new Options);
    x.opts.modelName = modelName;
    x.opts.imodel = imodel;
    x.opts.constFeat = constFeat;
    x.opts.aopts = aopts;
    x.opts.outdim = outdim;
    x.modelName = modelName;
    x.imodel = x.getModelMat(net, modelName, imodel)
    x;
  }
}

/**
 * Rectifying Linear Unit layer.
 */

class ReLULayer(override val net:Net, override val opts:ReLULayer.Opts = new ReLULayer.Options) extends Layer(net, opts) {
	override def forward = {
			createoutput;
			output <-- max(input.output, 0f);
			clearDeriv;
	}

	override def backward = {
			if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + (deriv ∘ (input.output > 0f));
	}
}

object ReLULayer {  
  trait Opts extends Layer.Opts {
  }
  class Options extends Opts {}
  
  def apply(net:Net) = new ReLULayer(net, new Options);
  
  def apply(net:Net, opts:Opts) = new ReLULayer(net, opts);
  
  def apply(in:Layer, net:Net) = {val x = new ReLULayer(net, new Options); x.inputs(0) = in; x;}
  
  def apply(in:Layer, net:Net, opts:Opts) = {val x = new ReLULayer(net, opts); x.inputs(0) = in; x;}
}

/**
 * Input layer is currently just a placeholder.
 */

class InputLayer(override val net:Net, override val opts:InputLayer.Opts = new InputLayer.Options) extends Layer(net, opts) {
}

object InputLayer {  
  trait Opts extends Layer.Opts {
  }
  class Options extends Opts {}
  
  def apply(net:Net) = new InputLayer(net, new Options);
  
  def apply(net:Net, opts:Opts) = new InputLayer(net, opts);
  
  def apply(in:Layer, net:Net) = {val x = new InputLayer(net, new Options); x.inputs(0) = in; x;}
  
  def apply(in:Layer, net:Net, opts:Opts) = {val x = new InputLayer(net, opts); x.inputs(0) = in; x;}
}

/**
 * GLMLayer implements linear, logistic and hinge-loss SVM. 
 * Commonly used as an output layer so includes a score method.
 */

class GLMLayer(override val net:Net, override val opts:GLMLayer.Opts = new GLMLayer.Options) extends Layer(net, opts) {
	var ilinks:Mat = null;
	var totflops = 0L;

	override def forward = {
			createoutput;
			if (ilinks.asInstanceOf[AnyRef] == null) {
			  ilinks = convertMat(opts.links);
			  for (i <- 0 until opts.links.length) {
			  	totflops += GLM.linkArray(opts.links(i)).fnflops
			  }
			}
			output <-- GLM.preds(input.output, ilinks, totflops);
			clearDeriv;
	}

	override def backward = {
			if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + (deriv ∘ GLM.derivs(output, target, ilinks, totflops));
	}

	override def score:FMat = { 
			val v = GLM.llfun(output, target, ilinks, totflops);
			FMat(mean(v, 2));
	}
}

object GLMLayer {  
  trait Opts extends Layer.Opts {
    var links:IMat = null;
  }
  class Options extends Opts {}
  
  def apply(net:Net) = new GLMLayer(net, new Options);
  
  def apply(net:Net, opts:Opts) = new GLMLayer(net, opts);
  
  def apply(in:Layer, net:Net, opts:Opts) = {val x = new GLMLayer(net, opts); x.inputs(0) = in; x;}
  
  def apply(in:Layer, net:Net, links:IMat) = {
    val x = new GLMLayer(net, new Options); 
    x.inputs(0) = in; 
    x.opts.links = links;
    x;
    }
}

/**
 * Normalization layer adds a downward-propagating derivative term whenever its norm 
 * is different from the specified value (targetNorm).
 */

class NormLayer(override val net:Net, override val opts:NormLayer.Opts = new NormLayer.Options) extends Layer(net, opts) {
	var sconst:Mat = null;

  override def forward = {
		createoutput;
		output <-- input.output;
		clearDeriv;
  }

  override def backward = {
    if (input.deriv.asInstanceOf[AnyRef] != null) {
    	if (sconst.asInstanceOf[AnyRef] == null) sconst = output.zeros(1,1);
    	sconst.set(math.min(0.1f, math.max(-0.1f, (opts.targetNorm - norm(output)/output.length).toFloat * opts.weight)));
    	input.derivs(0) = output ∘ sconst;
    	input.deriv ~ input.deriv + deriv;
    }
  }
}

object NormLayer {  
  trait Opts extends Layer.Opts {
    var targetNorm = 1f;
    var weight = 1f;
  }
  class Options extends Opts {}
  
  def apply(net:Net) = new NormLayer(net, new Options);
  
  def apply(net:Net, opts:Opts) = new NormLayer(net, opts);
  
  def apply(in:Layer, net:Net) = {val x = new NormLayer(net, new Options); x.inputs(0) = in; x;}
  
  def apply(in:Layer, net:Net, opts:Opts) = {val x = new NormLayer(net, opts); x.inputs(0) = in; x;}
  
  def apply(in:Layer, net:Net, targetNorm:Float, weight:Float) = {
    val x = new NormLayer(net, new Options);
    x.opts.targetNorm = targetNorm;
    x.opts.weight = weight;
    x.inputs(0) = in; 
    x;
    }
}

/**
 * Dropout layer with fraction to keep "frac". Deletes the same neurons in forward and backward pass. 
 * Assumes that "randmat" is not changed between forward and backward passes. 
 */

class DropoutLayer(override val net:Net, override val opts:DropoutLayer.Opts = new DropoutLayer.Options) extends Layer(net, opts) {  
	var randmat:Mat = null;

  override def forward = {
		createoutput;
		randmat = input.output + 20f;   // Hack to make a cached container to hold the random output
		if (nopts.predict) {
			output ~ input.output * opts.frac;
		} else {
			if (useGPU) {
				grand(randmat.asInstanceOf[GMat]); 
			} else {
				rand(randmat.asInstanceOf[FMat]);
			}
			randmat ~ (randmat < opts.frac)
			output ~ input.output ∘ randmat;
		}
		clearDeriv;
  }

  override def backward = {
		if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + deriv ∘ randmat;
  }
}

object DropoutLayer {  
  trait Opts extends Layer.Opts {
    var frac = 1f;
  }
  class Options extends Opts {}
  
  def apply(net:Net) = new DropoutLayer(net, new Options);
  
  def apply(net:Net, opts:Opts) = new DropoutLayer(net, opts);
  
  def apply(in:Layer, net:Net) = {val x = new DropoutLayer(net, new Options); x.inputs(0) = in; x;}
  
  def apply(in:Layer, net:Net, opts:Opts) = {val x = new DropoutLayer(net, opts); x.inputs(0) = in; x;}
  
  def apply(in:Layer, net:Net, frac:Float) = {
    val x = new DropoutLayer(net, new Options); 
    x.opts.frac = frac;
    x.inputs(0) = in; 
    x;
  }
}

/**
 * Computes the sum of input layers. 
 */

class AddLayer(override val net:Net, override val opts:AddLayer.Opts = new AddLayer.Options) extends Layer(net, opts) { 
  
  override val inputs = new Array[Layer](opts.ninputs);

	override def forward = {
			createoutput(inputs(0).output.nrows, inputs(0).output.ncols);
			output <-- inputs(0).output;
			(1 until inputs.length).map((i:Int) => output ~ output + inputs(i).output);
			clearDeriv;
	}

	override def backward = {
			(0 until inputs.length).map((i:Int) => {
				if (inputs(i).deriv.asInstanceOf[AnyRef] != null) inputs(i).deriv ~ inputs(i).deriv + deriv
			});
	}
}

object AddLayer {  
  trait Opts extends Layer.Opts {
    var ninputs = 1;
  }
  class Options extends Opts {}
  
  def apply(net:Net) = new AddLayer(net, new Options);
  
  def apply(net:Net, opts:Opts) = new AddLayer(net, opts);
  
  def apply(in:Layer, net:Net) = {val x = new AddLayer(net, new Options); x.inputs(0) = in; x;}
  
  def apply(in:Layer, net:Net, opts:Opts) = {val x = new AddLayer(net, opts); x.inputs(0) = in; x;}
}

/**
 * Computes the product of its input layers. 
 */

class MulLayer(override val net:Net, override val opts:MulLayer.Opts = new MulLayer.Options) extends Layer(net, opts) {  
  
	override val inputs = new Array[Layer](opts.ninputs);

	override def forward = {
			createoutput(inputs(0).output.nrows, inputs(0).output.ncols);
			output <-- inputs(0).output;
			(1 until inputs.length).map((i:Int) => output ~ output ∘ inputs(i).output);
			clearDeriv;
	}

	override def backward = {
			val doutput = deriv ∘ output;
			(0 until inputs.length).map((i:Int) => {
				if (inputs(i).deriv.asInstanceOf[AnyRef] != null) inputs(i).deriv ~ inputs(i).deriv + doutput / inputs(i).output;
			});
	}
}

object MulLayer {  
  trait Opts extends Layer.Opts {
    var ninputs = 1;
  }
  class Options extends Opts {}
  
  def apply(net:Net) = new MulLayer(net, new Options);
  
  def apply(net:Net, opts:Opts) = new MulLayer(net, opts);
  
  def apply(in:Layer, net:Net) = {val x = new MulLayer(net, new Options); x.inputs(0) = in; x;}
  
  def apply(in:Layer, net:Net, opts:Opts) = {val x = new MulLayer(net, opts); x.inputs(0) = in; x;}
}

/**
 * Softmax layer. Output = exp(input) / sum(exp(input))
 */

class SoftmaxLayer(override val net:Net, override val opts:SoftmaxLayer.Opts = new SoftmaxLayer.Options) extends Layer(net, opts) {    

	override def forward = {
			createoutput;
			val exps = exp(input.output);
			output ~ exps / sum(exps);
			clearDeriv;
	}

	override def backward = {
			val exps = exp(input.output);
			val sumexps = sum(exps);
			val isum = 1f / (sumexps ∘ sumexps);
			if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~
			input.deriv + ((exps / sumexps) ∘ deriv) - (exps ∘ (isum ∘ (exps ∙ deriv))) ;
	}
}

object SoftmaxLayer {  
  trait Opts extends Layer.Opts {
  }
  class Options extends Opts {}
  
  def apply(net:Net) = new SoftmaxLayer(net, new Options);
  
  def apply(net:Net, opts:Opts) = new SoftmaxLayer(net, opts);
  
  def apply(in:Layer, net:Net, opts:Opts) = {val x = new SoftmaxLayer(net, opts); x.inputs(0) = in; x;}
}
/**
 * Tanh layer. 
 */

class TanhLayer(override val net:Net, override val opts:TanhLayer.Opts = new TanhLayer.Options) extends Layer(net, opts) {    

	override def forward = {
			createoutput;
			tanh(input.output, output);
			clearDeriv;
	}

	override def backward = {
			val tmp = tanh(input.output);
			if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + (1 - tmp ∘ tmp) ∘ deriv;
	}
}

object TanhLayer {  
  trait Opts extends Layer.Opts {
  }
  class Options extends Opts {}
  
  def apply(net:Net) = new TanhLayer(net, new Options);
  
  def apply(net:Net, opts:Opts) = new TanhLayer(net, opts);
  
  def apply(in:Layer, net:Net) = {val x = new TanhLayer(net, new Options); x.inputs(0) = in; x;}
  
  def apply(in:Layer, net:Net, opts:Opts) = {val x = new TanhLayer(net, opts); x.inputs(0) = in; x;}
}

/**
 * Sigmoid layer. Uses GLM implementations of logistic functions for performance. 
 */

class SigmoidLayer(override val net:Net, override val opts:SigmoidLayer.Opts = new SigmoidLayer.Options) extends Layer(net, opts) {
	var ilinks:Mat = null;
  var totflops = 0L;

  override def forward = {
		createoutput;
		if (ilinks.asInstanceOf[AnyRef] == null) {
			ilinks = izeros(input.output.nrows, 1);
			ilinks.set(GLM.logistic);
			ilinks = convertMat(ilinks);
		}
		if (totflops == 0L) totflops = input.output.nrows * GLM.linkArray(1).fnflops;
		output <-- GLM.preds(input.output, ilinks, totflops);
		clearDeriv;
}

  override def backward = {
		val tmp = output - (output ∘ output);
		if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + tmp ∘ deriv;
}
}

object SigmoidLayer {  
  trait Opts extends Layer.Opts {
  }
  class Options extends Opts {}
  
  def apply(net:Net) = new SigmoidLayer(net, new Options);
  
  def apply(net:Net, opts:Opts) = new SigmoidLayer(net, opts);
  
  def apply(in:Layer, net:Net) = {val x = new SigmoidLayer(net, new Options); x.inputs(0) = in; x;}
  
  def apply(in:Layer, net:Net, opts:Opts) = {val x = new SigmoidLayer(net, opts); x.inputs(0) = in; x;}
}
/**
 * Softplus layer.  
 */

class SoftplusLayer(override val net:Net, override val opts:SoftplusLayer.Opts = new SoftplusLayer.Options) extends Layer(net, opts) {
	var ilinks:Mat = null;
  var totflops = 0L;

  override def forward = {
		createoutput;
		val big = input.output > 60f;      
		output ~ ((1 - big) ∘ ln(1f + exp(min(60f, input.output)))) + big ∘ input.output;
		clearDeriv;
  }

  override def backward = {
		if (ilinks.asInstanceOf[AnyRef] == null) {
			ilinks = izeros(input.output.nrows, 1);
			ilinks.set(GLM.logistic);
		}
		if (totflops == 0L) totflops = input.output.nrows * GLM.linkArray(1).fnflops;
		ilinks = convertMat(ilinks);
		if (input.deriv.asInstanceOf[AnyRef] != null) {
			val tmp = GLM.preds(input.output, ilinks, totflops);
			input.deriv ~ input.deriv + tmp ∘ deriv;
		}
  }
}

object SoftplusLayer {  
  trait Opts extends Layer.Opts {
  }
  class Options extends Opts {}
  
  def apply(net:Net) = new SoftplusLayer(net, new Options);
  
  def apply(net:Net, opts:Opts) = new SoftplusLayer(net, opts);
  
  def apply(in:Layer, net:Net) = {val x = new SoftplusLayer(net, new Options); x.inputs(0) = in; x;}
  
  def apply(in:Layer, net:Net, opts:Opts) = {val x = new SoftplusLayer(net, opts); x.inputs(0) = in; x;}
}
/**
 * Natural Log layer. 
 */

class LnLayer(override val net:Net, override val opts:LnLayer.Opts = new LnLayer.Options) extends Layer(net, opts) {

	override def forward = {
			createoutput;
			ln(input.output, output);
			clearDeriv;
	}

	override def backward = {
			if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + deriv/input.output;    
	}
}

object LnLayer {  
  trait Opts extends Layer.Opts {
  }
  class Options extends Opts {}
  
  def apply(net:Net) = new LnLayer(net, new Options);
  
  def apply(net:Net, opts:Opts) = new LnLayer(net, opts);
  
  def apply(in:Layer, net:Net) = {val x = new LnLayer(net, new Options); x.inputs(0) = in; x;}
  
  def apply(in:Layer, net:Net, opts:Opts) = {val x = new LnLayer(net, opts); x.inputs(0) = in; x;}
}

/**
 * Exponential layer. 
 */

class ExpLayer(override val net:Net, override val opts:ExpLayer.Opts = new ExpLayer.Options) extends Layer(net, opts) {

	override def forward = {
			createoutput;
			exp(input.output, output);
			clearDeriv;
	}

	override def backward = {
			if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + deriv ∘ output;    
	}
}

object ExpLayer {  
  trait Opts extends Layer.Opts {
  }
  class Options extends Opts {}
  
  def apply(net:Net) = new ExpLayer(net, new Options);
  
  def apply(net:Net, opts:Opts) = new ExpLayer(net, opts);
  
  def apply(in:Layer, net:Net) = {val x = new ExpLayer(net, new Options); x.inputs(0) = in; x;}
  
  def apply(in:Layer, net:Net, opts:Opts) = {val x = new ExpLayer(net, opts); x.inputs(0) = in; x;}
}
/**
 * Sum layer. 
 */

class SumLayer(override val net:Net, override val opts:SumLayer.Opts = new SumLayer.Options) extends Layer(net, opts) {
	var vmap:Mat = null;

  override def forward = {
		createoutput(1, input.output.ncols);
		output <-- sum(input.output);
		clearDeriv;
  }

  override def backward = {
		if (vmap.asInstanceOf[AnyRef] == null) vmap = deriv.ones(output.nrows, 1);
		if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + vmap * deriv;    
  }
}

object SumLayer {  
  trait Opts extends Layer.Opts {
  }
  class Options extends Opts {}
  
  def apply(net:Net) = new SumLayer(net, new Options);
  
  def apply(net:Net, opts:Opts) = new SumLayer(net, opts);
  
  def apply(in:Layer, net:Net) = {val x = new SumLayer(net, new Options); x.inputs(0) = in; x;}
  
  def apply(in:Layer, net:Net, opts:Opts) = {val x = new SumLayer(net, opts); x.inputs(0) = in; x;}
}
 


