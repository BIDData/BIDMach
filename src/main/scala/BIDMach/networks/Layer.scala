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
 * Basic Net Layer class. There are currently 16 layer types:
 - InputLayer: just a placeholder for the first layer which is loaded with input output blocks. No learnable params. 
 - LinLayer: Linear layer. Has a matrix of learnable params which is the input-output map. 
 - RectLayer: Rectifying one-to-one layer. No params.
 - GLMLayer: a one-to-one layer with GLM mappings (linear, logistic, abs-logistic and SVM). No learnable params. 
 - NormLayer: normalizing layer that adds a derivative term based on the difference between current layer norm and a target norm. 
   No learnable params. The target norm and weight of this derivative term can be specified. 
 - DropoutLayer: A layer that implements random dropout. No learnable params, but dropout fraction can be specified. 
 - AddLayer: adds input layers element-wise.
 - MulLayer: multiplies input layers element-wise. Will also perform edge operations (one input can be a scalar). 
 - SoftmaxLayer: a softmax (normalized exponential) layer.
 - TanhLayer: Hyperbolic tangent non-linearity.
 - SigmoidLayer: Logistic function non-linearity.
 - SoftplusLayer: smooth ReLU unit. 
 - LnLayer: natural logarithm
 - ExpLayer: exponential
 - SumLayer: column-wise sum
 - CopyLayer: copies its input to its output. 
 *
 * Currently only four Layer types need params:
 - LinLayer: "outside" holds the output dimensions of the FClayer (input dimension set by previous layer). 
 - GLMLayer: "links" holds the links matrix (integer optss for loss types, see GLM), for the output of that layer. Its size should match the number of targets.
 - NormLayer: "targetNorm" holds a target per-element norm, and "weight" is the weight for this term in the derivative calculation.
 - DropoutLayer: "frac" holds the fraction of neurons to retain.
 *
 * The network topology is normally specified by opts.layers which is a sequence of "Layer.Options" objects. There is a nested Options
 * Class for each Layer class, which holds the params for defining that layer, and pointers to any input Layers via their Options classes.
 * In other words, the options classes allow construction of a mirror of the actual network topology. This allows patterns of
 * structure to be repeated using a single Options graph structure. 
 * 
 * Each LayerOptions instance has up to two inputs which are other LayerOptions instances (or null). This graph structure can be cyclic. 
 * When the model is created, the Layer structure mimics the LayerOptions structure. 
 * 
 * You can also create the Layer graph directly using the "setinput()" method in each layer. 
 */


  
class Layer(val net:Net, val opts:Layer.Options = new Layer.Options) {
  val _inputs = new Array[Layer](1);
  val _inputNums = Array(0);
  val _outputs = new Array[Mat](1);
  val _derivs = new Array[Mat](1);
  def inputlength = _inputs.length
  
  def inputs(i:Int) = _inputs(i);
  def outputs(i:Int) = _outputs(i);
  def derivs(i:Int) = _derivs(i);
  
  def setinput(i:Int, v:Layer) = {_inputs(i) = v;}
  def setoutput(i:Int, v:Mat) = {_outputs(i) = v;}
  def setderiv(i:Int, v:Mat) = {_derivs(i) = v;}
  def setinout(i:Int, v:Layer, j:Int) = {_inputs(i) = v; _inputNums(i) = j;}
  
  def input = _inputs(0);
  def output = _outputs(0);
  def deriv = _derivs(0);
  def input_=(v:Layer):Unit = {_inputs(0) = v;}
  def output_= (v:Mat):Unit = {_outputs(0) = v};
  def deriv_=(v:Mat):Unit = {_derivs(0) = v};
  
  def inputOut = input.outputs(_inputNums(0));
  
  var target:Mat = null;
  def forward = {};
  def backward:Unit = {};
  def backward(ipass:Int, pos:Long):Unit = backward;
  def score:FMat = zeros(1,1);
  var parent:Layer = null;
  lazy val modelmats = net.modelmats;
  lazy val updatemats = net.updatemats;
  lazy val useGPU = net.useGPU;
  lazy val nopts = net.opts;
  def convertMat(mat:Mat) = {net.convertMat(mat);}

  def createoutput = {
  	if (output.asInstanceOf[AnyRef] == null) output = inputOut.zeros(inputOut.nrows, inputOut.ncols);
  }

  def createoutput(nrows:Int, ncols:Int) = {
  	if (output.asInstanceOf[AnyRef] == null) output = inputOut.zeros(nrows, ncols);
  }

  def clearDeriv = {
  	if (deriv.asInstanceOf[AnyRef] == null) deriv = output.zeros(output.nrows, output.ncols);
  	deriv.clear;
  }
  
  def getModelMats(net:Net):Unit = {}
}


object Layer {  
  class Options{
    val inputs:Array[Options] = Array(null);
    var myLayer:Layer = null;
    var myGhost:Options = null;
    var parent:Options = null;
    
    def copyTo(opts:Options):Options = {
      opts.inputs(0) = inputs(0);
      myGhost = opts;
      opts;
    }
    
    override def clone:Options = {
      copyTo(new Options);
    }
    
    def create(net:Net):Layer = {null}
  }
}

class ModelLayer(override val net:Net, override val opts:ModelLayer.Options = new ModelLayer.Options) extends Layer(net, opts) {
	var imodel = 0;
  
  override def getModelMats(net:Net):Unit = {
		imodel = if (net.opts.nmodelmats > 0) {   // If explicit model numbers are given, use them. 
			opts.imodel;
		} else if (opts.modelName.length > 0) {               // If this is a named layer, look it up. 
			if (net.modelMap.contains(opts.modelName)) {
				net.modelMap(opts.modelName);
			} else {
				val len = net.modelMap.size;
				net.modelMap(opts.modelName) = len + net.opts.nmodelmats; 	
				len;
			}
		} else {                                         // Otherwise return the next available int
			net.imodel += 1;
			net.imodel - 1;
		};
  }
}

object ModelLayer {  
  class Options extends Layer.Options {
  	var modelName = "";
    var imodel = 0;
    
    def copyTo(opts:Options):Options = {
      super.copyTo(opts);
      opts.modelName = modelName;
      opts.imodel = imodel;
      opts;
    }
    
    override def clone:Options = {
      copyTo(new Options);
    }
  }
}
/**
 * Linear layer. 
 * Includes a model matrix that contains the linear map. 
 */

class LinLayer(override val net:Net, override val opts:LinLayer.Options = new LinLayer.Options) extends ModelLayer(net, opts) {
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
  	  val outdim = if (opts.outdim == 0) (inputOut.nrows + (if (opts.constFeat) 1 else 0)) else opts.outdim;
  		modelmats(imodel) = convertMat(normrnd(0, 1, outdim, inputOut.nrows + (if (opts.constFeat) 1 else 0)));
  		updatemats(imodel) = modelmats(imodel).zeros(modelmats(imodel).nrows, modelmats(imodel).ncols);
  		if (opts.aopts != null) initADAGrad;  
  	}
  	val mm = if (opts.constFeat) {
  		modelmats(imodel).colslice(1, modelmats(imodel).ncols);
  	} else {
  		modelmats(imodel);
  	}
  	createoutput(mm.nrows, inputOut.ncols);
  	output ~ mm * inputOut;
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
  		ADAGrad.multUpdate(deriv, inputOut, modelmats(imodel), sumsq, mask, lrate, texp, vexp, epsilon, istep, waitsteps);
  	} else {
  		val dprod = deriv *^ inputOut;
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
  class Options extends ModelLayer.Options {
    var constFeat:Boolean = false;
    var aopts:ADAGrad.Opts = null;
    var outdim = 0;
    
    def copyTo(opts:Options):Options = {
    	super.copyTo(opts);
    	opts.constFeat = constFeat;
    	opts.aopts = aopts;
    	opts.outdim = outdim;
    	opts;
    }
    
    override def clone:Options = {
    	copyTo(new Options);
    }
    
    override def create(net:Net):LinLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new LinLayer(net, new Options);
  
  def apply(net:Net, opts:Options):LinLayer = new LinLayer(net, opts);
  
}

/**
 * Rectifying Linear Unit layer.
 */

class RectLayer(override val net:Net, override val opts:RectLayer.Options = new RectLayer.Options) extends Layer(net, opts) {
	override def forward = {
			createoutput;
			output <-- max(inputOut, 0f);
			clearDeriv;
	}

	override def backward = {
			if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + (deriv ∘ (inputOut > 0f));
	}
}

object RectLayer {  
  class Options extends Layer.Options {
  	override def clone:Options = {
    	copyTo(new Options).asInstanceOf[Options];
    }
  	
  	override def create(net:Net):RectLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new RectLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new RectLayer(net, opts);
}

/**
 * Input layer is currently just a placeholder.
 */

class InputLayer(override val net:Net, override val opts:InputLayer.Options = new InputLayer.Options) extends Layer(net, opts) {
}

object InputLayer {  
  class Options extends Layer.Options {
    
  	override def clone:Options = {copyTo(new Options).asInstanceOf[Options];}
  	
  	override def create(net:Net):InputLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new InputLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new InputLayer(net, opts);
}

/**
 * GLMLayer implements linear, logistic and hinge-loss SVM. 
 * Commonly used as an output layer so includes a score method.
 */

class GLMLayer(override val net:Net, override val opts:GLMLayer.Options = new GLMLayer.Options) extends Layer(net, opts) {
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
			output <-- GLM.preds(inputOut, ilinks, totflops);
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
  class Options extends Layer.Options {
    var links:IMat = null;
    
    def copyTo(opts:Options):Options = {
  		super.copyTo(opts);
  		opts.links = links;
  		opts;
    }
    
    override def clone:Options = {copyTo(new Options);}   
    
    override def create(net:Net):GLMLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new GLMLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new GLMLayer(net, opts); 

}

/**
 * Normalization layer adds a downward-propagating derivative term whenever its norm 
 * is different from the optsified value (targetNorm).
 */

class NormLayer(override val net:Net, override val opts:NormLayer.Options = new NormLayer.Options) extends Layer(net, opts) {
	var sconst:Mat = null;

  override def forward = {
		createoutput;
		output <-- inputOut;
		clearDeriv;
  }

  override def backward = {
    if (input.deriv.asInstanceOf[AnyRef] != null) {
    	if (sconst.asInstanceOf[AnyRef] == null) sconst = output.zeros(1,1);
    	sconst.set(math.min(0.1f, math.max(-0.1f, (opts.targetNorm - norm(output)/output.length).toFloat * opts.weight)));
    	input.deriv = output ∘ sconst;
    	input.deriv ~ input.deriv + deriv;
    }
  }
}

object NormLayer {  
  class Options extends Layer.Options {
    var targetNorm = 1f;
    var weight = 1f;
    
    def copyTo(opts:Options):Options = {
  		super.copyTo(opts);
  		opts.targetNorm = targetNorm;
  		opts.weight = weight;
  		opts;
    }
    
    override def clone:Options = {copyTo(new Options);}
    
    override def create(net:Net):NormLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new NormLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new NormLayer(net, opts);  
}

/**
 * Dropout layer with fraction to keep "frac". Deletes the same neurons in forward and backward pass. 
 * Assumes that "randmat" is not changed between forward and backward passes. 
 */

class DropoutLayer(override val net:Net, override val opts:DropoutLayer.Options = new DropoutLayer.Options) extends Layer(net, opts) {  
	var randmat:Mat = null;

  override def forward = {
		createoutput;
		randmat = inputOut + 20f;   // Hack to make a cached container to hold the random output
		if (nopts.predict) {
			output ~ inputOut * opts.frac;
		} else {
			if (useGPU) {
				grand(randmat.asInstanceOf[GMat]); 
			} else {
				rand(randmat.asInstanceOf[FMat]);
			}
			randmat ~ (randmat < opts.frac)
			output ~ inputOut ∘ randmat;
		}
		clearDeriv;
  }

  override def backward = {
		if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + deriv ∘ randmat;
  }
}

object DropoutLayer {  
  class Options extends Layer.Options {
    var frac = 1f;
    
    def copyTo(opts:Options):Options = {
  		super.copyTo(opts);
  		opts.frac = frac;
  		opts;
    }
    
    override def clone:Options = {copyTo(new Options);}
    
    override def create(net:Net):DropoutLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new DropoutLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new DropoutLayer(net, opts);
}

/**
 * Computes the sum of input layers. 
 */

class AddLayer(override val net:Net, override val opts:AddLayer.Options = new AddLayer.Options) extends Layer(net, opts) { 
  
  override val _inputs = new Array[Layer](opts.ninputs);

	override def forward = {
			createoutput(inputOut.nrows, inputOut.ncols);
			output <-- inputOut;
			(1 until inputlength).map((i:Int) => output ~ output + inputs(i).output);
			clearDeriv;
	}

	override def backward = {
			(0 until inputlength).map((i:Int) => {
				if (inputs(i).deriv.asInstanceOf[AnyRef] != null) inputs(i).deriv ~ inputs(i).deriv + deriv
			});
	}
}

object AddLayer {  
  class Options extends Layer.Options {
    var ninputs = 2;
    override val inputs:Array[Layer.Options] = new Array[Layer.Options](ninputs);
    
    def copyTo(opts:Options):Options = {
  		super.copyTo(opts);
  		opts.ninputs = ninputs;
  		opts;
    }
    
    override def clone:Options = {copyTo(new Options);}
    
    override def create(net:Net):AddLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new AddLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new AddLayer(net, opts); 
}

/**
 * Computes the product of its input layers. 
 */

class MulLayer(override val net:Net, override val opts:MulLayer.Options = new MulLayer.Options) extends Layer(net, opts) {  
  
	override val _inputs = new Array[Layer](opts.ninputs);

	override def forward = {
			createoutput(inputOut.nrows, inputOut.ncols);
			output <-- inputOut;
			(1 until inputlength).map((i:Int) => output ~ output ∘ inputs(i).output);
			clearDeriv;
	}

	override def backward = {
			val doutput = deriv ∘ output;
			(0 until inputlength).map((i:Int) => {
				if (inputs(i).deriv.asInstanceOf[AnyRef] != null) inputs(i).deriv ~ inputs(i).deriv + doutput / inputs(i).output;
			});
	}
}

object MulLayer {  
  class Options extends Layer.Options {
    var ninputs = 2;
    override val inputs:Array[Layer.Options] = new Array[Layer.Options](ninputs);
    
    def copyTo(opts:Options):Options = {
  		super.copyTo(opts);
  		opts.ninputs = ninputs;
  		opts;
    }
    
    override def clone:Options = {copyTo(new Options);}
    
    override def create(net:Net):MulLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new MulLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new MulLayer(net, opts); 
}

/**
 * Softmax layer. Output = exp(input) / sum(exp(input))
 */

class SoftmaxLayer(override val net:Net, override val opts:SoftmaxLayer.Options = new SoftmaxLayer.Options) extends Layer(net, opts) {    

	override def forward = {
			createoutput;
			val exps = exp(inputOut - maxi(inputOut));  // ensures sum(exps) is between 1 and nfeats
			output ~ exps / sum(exps);
			clearDeriv;
	}

	override def backward = {
			val exps = exp(inputOut - maxi(inputOut));
			val sumexps = sum(exps);
			val isum = 1f / (sumexps ∘ sumexps);
			if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~
			input.deriv + ((exps / sumexps) ∘ deriv) - (exps ∘ (isum ∘ (exps ∙ deriv))) ;
	}
}

object SoftmaxLayer {  
  class Options extends Layer.Options {
   
    override def clone:Options = {copyTo(new Options).asInstanceOf[Options];}
    
    override def create(net:Net):SoftmaxLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new SoftmaxLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new SoftmaxLayer(net, opts);
}
/**
 * Tanh layer. 
 */

class TanhLayer(override val net:Net, override val opts:TanhLayer.Options = new TanhLayer.Options) extends Layer(net, opts) {    

	override def forward = {
			createoutput;
			tanh(inputOut, output);
			clearDeriv;
	}

	override def backward = {
			val tmp = tanh(inputOut);
			if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + (1 - tmp ∘ tmp) ∘ deriv;
	}
}

object TanhLayer {  
  class Options extends Layer.Options {
    
    override def clone:Options = {copyTo(new Options).asInstanceOf[Options];}
    
    override def create(net:Net):TanhLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new TanhLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new TanhLayer(net, opts);
}

/**
 * Sigmoid layer. Uses GLM implementations of logistic functions for performance. 
 */

class SigmoidLayer(override val net:Net, override val opts:SigmoidLayer.Options = new SigmoidLayer.Options) extends Layer(net, opts) {
	var ilinks:Mat = null;
  var totflops = 0L;

  override def forward = {
		createoutput;
		if (ilinks.asInstanceOf[AnyRef] == null) {
			ilinks = izeros(inputOut.nrows, 1);
			ilinks.set(GLM.logistic);
			ilinks = convertMat(ilinks);
		}
		if (totflops == 0L) totflops = inputOut.nrows * GLM.linkArray(1).fnflops;
		output <-- GLM.preds(inputOut, ilinks, totflops);
		clearDeriv;
}

  override def backward = {
		val tmp = output - (output ∘ output);
		if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + tmp ∘ deriv;
}
}

object SigmoidLayer {  
  class Options extends Layer.Options {
    
    override def clone:Options = {copyTo(new Options).asInstanceOf[Options];}
    
    override def create(net:Net):SigmoidLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new SigmoidLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new SigmoidLayer(net, opts); 
}
/**
 * Softplus layer.  
 */

class SoftplusLayer(override val net:Net, override val opts:SoftplusLayer.Options = new SoftplusLayer.Options) extends Layer(net, opts) {
	var ilinks:Mat = null;
  var totflops = 0L;

  override def forward = {
		createoutput;
		val big = inputOut > 60f;      
		output ~ ((1 - big) ∘ ln(1f + exp(min(60f, inputOut)))) + big ∘ inputOut;
		clearDeriv;
  }

  override def backward = {
		if (ilinks.asInstanceOf[AnyRef] == null) {
			ilinks = izeros(inputOut.nrows, 1);
			ilinks.set(GLM.logistic);
		}
		if (totflops == 0L) totflops = inputOut.nrows * GLM.linkArray(1).fnflops;
		ilinks = convertMat(ilinks);
		if (input.deriv.asInstanceOf[AnyRef] != null) {
			val tmp = GLM.preds(inputOut, ilinks, totflops);
			input.deriv ~ input.deriv + tmp ∘ deriv;
		}
  }
}

object SoftplusLayer {  
  class Options extends Layer.Options {
    
    override def clone:Options = {copyTo(new Options).asInstanceOf[Options];}
    
    override def create(net:Net):SoftplusLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new SoftplusLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new SoftplusLayer(net, opts); 
}
/**
 * Natural Log layer. 
 */

class LnLayer(override val net:Net, override val opts:LnLayer.Options = new LnLayer.Options) extends Layer(net, opts) {

	override def forward = {
			createoutput;
			ln(inputOut, output);
			clearDeriv;
	}

	override def backward = {
			if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + deriv/inputOut;    
	}
}

object LnLayer {  
  class Options extends Layer.Options {
    
  	override def clone:Options = {copyTo(new Options).asInstanceOf[Options];}
  	
  	override def create(net:Net):LnLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new LnLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new LnLayer(net, opts);
}

/**
 * Exponential layer. 
 */

class ExpLayer(override val net:Net, override val opts:ExpLayer.Options = new ExpLayer.Options) extends Layer(net, opts) {

	override def forward = {
			createoutput;
			exp(inputOut, output);
			clearDeriv;
	}

	override def backward = {
			if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + deriv ∘ output;    
	}
}

object ExpLayer {  
  class Options extends Layer.Options {
    
  	override def clone:Options = {copyTo(new Options).asInstanceOf[Options];}
  	
    override def create(net:Net):ExpLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new ExpLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new ExpLayer(net, opts);

}
/**
 * Sum layer. 
 */

class SumLayer(override val net:Net, override val opts:SumLayer.Options = new SumLayer.Options) extends Layer(net, opts) {
	var vmap:Mat = null;

  override def forward = {
		createoutput(1, inputOut.ncols);
		output <-- sum(inputOut);
		clearDeriv;
  }

  override def backward = {
		if (vmap.asInstanceOf[AnyRef] == null) vmap = deriv.ones(output.nrows, 1);
		if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + vmap * deriv;    
  }
}

object SumLayer {  
  class Options extends Layer.Options {
    
  	override def clone:Options = {copyTo(new Options).asInstanceOf[Options];}
  	
  	override def create(net:Net):SumLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new SumLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new SumLayer(net, opts);

}

class CopyLayer(override val net:Net, override val opts:CopyLayer.Options = new CopyLayer.Options) extends Layer(net, opts) {

  override def forward = {
  	if (output.asInstanceOf[AnyRef] == null) {
  	  val io = input.output(opts.inputnum);
  	  output = io.zeros(io.nrows, io.ncols);
  	}
		output <-- input.output(opts.inputnum);
		clearDeriv;
  }

  override def backward = {
    if (input.derivs(opts.inputnum).asInstanceOf[AnyRef] != null) input.derivs(opts.inputnum) ~ input.derivs(opts.inputnum) + deriv    
  }
}

object CopyLayer {  
  class Options extends Layer.Options {
    
    var inputnum = 0;
    
  	override def clone:Options = {copyTo(new Options).asInstanceOf[Options];}
  	
  	override def create(net:Net):CopyLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new CopyLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new CopyLayer(net, opts);

}

class CompoundLayer(override val net:Net, override val opts:CompoundLayer.Options = new CompoundLayer.Options) extends ModelLayer(net, opts) {
	
	override def setinput(i:Int, v:Layer) = {               // Assumes the inputs are the first k layers in internal_layers
	  _inputs(i) = v;
	  internal_layers(i).setinput(0, v);
	}
	
  override def setinout(i:Int, v:Layer, j:Int) = {               // Assumes the inputs are the first k layers in internal_layers
	  _inputs(i) = v;
	  internal_layers(i).setinout(0, v, j);
	}
	
	var internal_layers:Array[Layer] = null;
	
	override def forward = {
	  if (net.opts.debug == 0) {
	    internal_layers.map(_.forward);
	  } else {
	    for (i <- 0 until internal_layers.length) {
	      if (net.opts.debug > 0) println("  compound layer forward %d %s" format (i, internal_layers(i).getClass));
	      internal_layers(i).forward;
	    }
	  }
	  for (i <- 0 until opts.outputNumbers.length) {
	    _outputs(i) = internal_layers(opts.outputNumbers(i)).output
	    if (_derivs(i).asInstanceOf[AnyRef] == null){
	      _derivs(i) = internal_layers(opts.outputNumbers(i)).deriv;
	    }
	  }
	}
	
	override def backward(ipass:Int, pos:Long) = {
		if (net.opts.debug == 0) {
	  internal_layers.reverse.map(_.backward(ipass, pos));
		} else {
	    for (i <- internal_layers.length until 1 by -1) {
	      if (net.opts.debug > 0) println("  compound layer backward %d" format (i-1, internal_layers(i-1).getClass));
	      internal_layers(i-1).backward(ipass, pos);
	    }
	  }
	}
		
	override def getModelMats(net:Net) = {
	  internal_layers.map(_.getModelMats(net));
	}

	def construct = {
		internal_layers = new Array[Layer](opts.lopts.length);
	  for (i <- 0 until internal_layers.length) {
	  	internal_layers(i) = opts.lopts(i).create(net);
	  	opts.lopts(i).myLayer = internal_layers(i);
	  	internal_layers(i).parent = this;
	  }
	  for (i <- 0 until internal_layers.length) {
	  	for (j <- 0 until opts.lopts(i).inputs.length) {
    		if (opts.lopts(i).inputs(j) != null) internal_layers(i).setinput(j, opts.lopts(i).inputs(j).myLayer);
    	}
	  }
	}
}

object CompoundLayer {
  class Options extends ModelLayer.Options {  	  
	  var lopts:Array[Layer.Options] = null;
	  var outputNumbers:Array[Int] = null;
 	  var prefix = "";
  }
}

class LayerOptions(val nlayers:Int) {
  
  val layerOptionss = new Array[Layer.Options](nlayers);
  
  def apply(i:Int):Layer.Options = layerOptionss(i);
  
  def update(i:Int, lopts:Layer.Options) = {layerOptionss(i) = lopts; this}
  
  override def clone = copyTo(new LayerOptions(nlayers));
  
  def copyTo(lopts:LayerOptions):LayerOptions = {
    for (i <- 0 until nlayers) {
      lopts.layerOptionss(i) = layerOptionss(i).clone;
      layerOptionss(i).myGhost = lopts.layerOptionss(i);
    }
    for (i <- 0 until nlayers) {
      for (j <- 0 until layerOptionss(i).inputs.length) {
      	if (layerOptionss(i).inputs(j) != null) lopts.layerOptionss(i).inputs(j) = layerOptionss(i).inputs(j).myGhost;
      }
    }
    lopts;
  }
}
 


