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
 * Basic Net Layer class. There are currently 17 layer types:
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
 - OnehotLayer: Converts an integer array of feature values to a sparse matrix whose columns are the instances with one non-zero in the feature position. 
 *
 *
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

// Notes: 
// Layer Nodes can have multiple inputs and multiple outputs. 
// Each layer contains an array of inputs, an array of outputs, and an array of derivatives. 
// The output and derivatives are "owned" by the node and are simple arrays of Mat. 
//
// The inputs comprise a reference to another layer and an integer which is the number of output of that layer to use. 
// _inputs(i): refers to input layer i, and _inputNums(i): the number of the output of layer i we are using. 
//
// To simplify references to input matrices, convenience functions are provided:
//   inputData: refers to this layers first input matrix. 
//   inputDeriv: refers to the derivative matrix for the first input. 
//   inputDatas(i): refers to the i^th input matrix of this layer.
//   inputDerivs(i); refers to the derivative of the i^th input layer. 
//
// its also possible to assign to inputDeriv for backward processing. 
//
// To set layer A's i^th input to layer B's default (0th) output, do A.setinput(i, B)
// To set layer A's i^th input to layer B's j^th output, do A.setinout(i, B, j)

  
class Layer(val net:Net, val opts:Layer.Options = new Layer.Options) {
  // Internal data arrays
  val _inputs = new Array[Layer](1);
  val _inputTerminals = Array(0);
  val _outputs = new Array[Mat](1);
  val _derivs = new Array[Mat](1);
  def inputlength = _inputs.length
  
  // Setters and getters for general elements of those arrays
  def inputs(i:Int) = _inputs(i);
  def outputs(i:Int) = _outputs(i);
  def derivs(i:Int) = _derivs(i);  
  def setoutput(i:Int, v:Mat):Layer = {_outputs(i) = v; this}
  def setderiv(i:Int, v:Mat):Layer = {_derivs(i) = v; this}
  def setinput(i:Int, v:Layer):Layer = {_inputs(i) = v; _inputTerminals(i) = 0; this}
  def setinout(i:Int, v:Layer, j:Int):Layer = {_inputs(i) = v; _inputTerminals(i) = j; this}
  
  // Setters and getters for the first input or output
  def input = _inputs(0);
  def output = _outputs(0);
  def deriv = _derivs(0);
  def input_=(v:Layer):Unit = {_inputs(0) = v;}
  def output_= (v:Mat):Unit = {_outputs(0) = v};
  def deriv_=(v:Mat):Unit = {_derivs(0) = v};
  
  // Input getters (and one setter) which get the appropriate output from each input layer
  def inputData = _inputs(0)._outputs(_inputTerminals(0));
  def inputDeriv = _inputs(0)._derivs(_inputTerminals(0));
  def inputDeriv_=(v:Mat):Unit = {_inputs(0)._derivs(_inputTerminals(0)) = v}; 
  def inputDatas(i:Int) = _inputs(i)._outputs(_inputTerminals(i));
  def inputDerivs(i:Int) = _inputs(i)._derivs(_inputTerminals(i));
  
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
  	if (output.asInstanceOf[AnyRef] == null) output = inputData.zeros(inputData.nrows, inputData.ncols);
  }

  def createoutput(nrows:Int, ncols:Int) = {
  	if (output.asInstanceOf[AnyRef] == null) output = inputData.zeros(nrows, ncols);
  }

  def clearDeriv = {
  	if (deriv.asInstanceOf[AnyRef] == null) deriv = output.zeros(output.nrows, output.ncols);
  	deriv.clear;
  }
  
  def clearDerivs = {
    if (deriv.asInstanceOf[AnyRef] == null) {
      for (i <- 0 until _outputs.length) {
        _derivs(i) = output.zeros(_outputs(i).nrows, _outputs(i).ncols);
      }
    }
    for (i <- 0 until _derivs.length) {
      _derivs(i).clear
    }
  }
  
  def getModelMats(net:Net):Unit = {}
}


object Layer {  
  class Options{
    val inputs:Array[Options] = Array(null);
    val inputTerminals:Array[Int] = Array(0);
    var myLayer:Layer = null;
    var myGhost:Options = null;
    var parent:Options = null;
    var outputNumbers:Array[Int] = null;
    
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
//  var sumsq:Mat = null;
  var mask:Mat = null;
  var firststep = -1f;
  var waitsteps = 0;
  var epsilon = 0f;
  var ADAinitialized = false;

  override def forward = {
    if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
      val outdim = if (opts.outdim == 0) (inputData.nrows + (if (opts.constFeat) 1 else 0)) else opts.outdim;
      modelmats(imodel) = convertMat(normrnd(0, 1, outdim, inputData.nrows + (if (opts.constFeat) 1 else 0)));
      updatemats(imodel) = modelmats(imodel).zeros(modelmats(imodel).nrows, modelmats(imodel).ncols);  
    }
    if (opts.aopts != null && !ADAinitialized) initADAGrad;
    val mm = if (opts.constFeat) {
      modelmats(imodel).colslice(1, modelmats(imodel).ncols);
    } else {
      modelmats(imodel);
    }
    createoutput(mm.nrows, inputData.ncols);
    output ~ mm * inputData;
    if (opts.constFeat) output ~ output + modelmats(imodel).colslice(0, 1);
    clearDeriv;
  }

  override def backward(ipass:Int, pos:Long) = {
    val mm = if (opts.constFeat && imodel > 0) {
      modelmats(imodel).colslice(1, modelmats(imodel).ncols);
    } else {
      modelmats(imodel);
    }
    var dprod:Mat = null;
    if (inputDeriv.asInstanceOf[AnyRef] != null) {
      inputDeriv ~ inputDeriv + (mm ^* deriv);
    }
    if (opts.aopts != null) {
      if (firststep <= 0) firststep = pos.toFloat;
      val istep = (pos + firststep)/firststep;
      ADAGrad.multUpdate(deriv, inputData, modelmats(imodel), updatemats(imodel), mask, lrate, texp, vexp, epsilon, istep, waitsteps);
    } else {
      val dprod = deriv *^ inputData;
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
//    sumsq = convertMat(zeros(d, m));
    updatemats(imodel).set(aopts.initsumsq);
    waitsteps = aopts.waitsteps;
    epsilon = aopts.epsilon;
    mask = aopts.mask;
    ADAinitialized = true;
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
			output <-- max(inputData, 0f);
			clearDeriv;
	}

	override def backward = {
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv ∘ (inputData > 0f));
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
			output <-- GLM.preds(inputData, ilinks, totflops);
			clearDeriv;
	}

	override def backward = {
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv ∘ GLM.derivs(output, target, ilinks, totflops));
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
		output <-- inputData;
		clearDeriv;
  }

  override def backward = {
    if (inputDeriv.asInstanceOf[AnyRef] != null) {
    	if (sconst.asInstanceOf[AnyRef] == null) sconst = output.zeros(1,1);
    	sconst.set(math.min(0.1f, math.max(-0.1f, (opts.targetNorm - norm(output)/output.length).toFloat * opts.weight)));
    	inputDeriv = output ∘ sconst;
    	inputDeriv ~ inputDeriv + deriv;
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
		randmat = inputData + 20f;   // Hack to make a cached container to hold the random output
		if (nopts.predict) {
			output ~ inputData * opts.frac;
		} else {
			if (useGPU) {
				grand(randmat.asInstanceOf[GMat]); 
			} else {
				rand(randmat.asInstanceOf[FMat]);
			}
			randmat ~ randmat < opts.frac
			output ~ inputData ∘ randmat;
		}
		clearDeriv;
  }

  override def backward = {
		if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv ∘ randmat);
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
  override val _inputTerminals = new Array[Int](opts.ninputs);

	override def forward = {
			createoutput(inputData.nrows, inputData.ncols);
			output <-- inputData;
			(1 until inputlength).map((i:Int) => output ~ output + inputDatas(i));
			clearDeriv;
	}

	override def backward = {
			(0 until inputlength).map((i:Int) => {
				if (inputDerivs(i).asInstanceOf[AnyRef] != null) inputDerivs(i) ~ inputDerivs(i) + deriv
			});
	}
}

object AddLayer {  
  class Options extends Layer.Options {
    var ninputs = 2;
    override val inputs:Array[Layer.Options] = new Array[Layer.Options](ninputs);
    override val inputTerminals:Array[Int] = new Array[Int](ninputs);
    
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
	override val _inputTerminals = new Array[Int](opts.ninputs);
  val qeps = 1e-40f;
  
  def guardSmall(a:Mat, eps:Float):Mat = {
    a + (abs(a) < eps) * (2*eps);
  }

	override def forward = {
	  createoutput(inputData.nrows, inputData.ncols);
			output <-- inputData;
			(1 until inputlength).map((i:Int) => output ~ output ∘ inputDatas(i));
			clearDeriv;
	}

	override def backward = {
    if (_inputs.length == 2) {
      if (inputDerivs(0).asInstanceOf[AnyRef] != null) inputDerivs(0) ~ inputDerivs(0) + (deriv ∘ inputDatas(1));
      if (inputDerivs(1).asInstanceOf[AnyRef] != null) inputDerivs(1) ~ inputDerivs(1) + (deriv ∘ inputDatas(0));
    } else {
			val doutput = deriv ∘ output;
			(0 until inputlength).map((i:Int) => {
				if (inputDerivs(i).asInstanceOf[AnyRef] != null) inputDerivs(i) ~ inputDerivs(i) + (doutput / guardSmall(inputDatas(i), qeps));
			});
    }
	}
}

object MulLayer {  
  class Options extends Layer.Options {
    var ninputs = 2;
    override val inputs:Array[Layer.Options] = new Array[Layer.Options](ninputs);
    override val inputTerminals:Array[Int] = new Array[Int](ninputs);
    
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
  var coloffsets:Mat = null;

	override def forward = {
			createoutput;
			val exps = exp(inputData - maxi(inputData));  // ensures sum(exps) is between 1 and nfeats
			output ~ exps / sum(exps);
			clearDeriv;
	}

	override def backward = {
			val exps = exp(inputData - maxi(inputData));
			val sumexps = sum(exps);
			val isum = 1f / (sumexps ∘ sumexps);
			if (inputDeriv.asInstanceOf[AnyRef] != null) 
        inputDeriv ~ inputDeriv + (((exps / sumexps) ∘ deriv) - (exps ∘ (isum ∘ (exps ∙ deriv))));
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
 * Softmax layer. Output = exp(input) / sum(exp(input))
 */

class SoftmaxOutputLayer(override val net:Net, override val opts:SoftmaxOutputLayer.Options = new SoftmaxOutputLayer.Options) extends Layer(net, opts) { 
  var coloffsets:Mat = null;

  override def forward = {
      createoutput;
      val exps = exp(inputData - maxi(inputData));  // ensures sum(exps) is between 1 and nfeats
      output ~ exps / sum(exps);
      clearDeriv;
  }

  override def backward = {
		  if (coloffsets.asInstanceOf[AnyRef] == null) coloffsets = convertMat(irow(0->output.ncols)*output.nrows);
		  if (inputDeriv.asInstanceOf[AnyRef] != null) {
/*			  val exps = exp(inputData - maxi(inputData));
			  val sumexps = sum(exps);
        val preds = exps / sumexps; */
//			  val isum = 1f / (sumexps ∘ sumexps);
//        inputDeriv ~ inputDeriv + (((exps / sumexps) ∘ deriv) - (exps ∘ (isum ∘ (exps ∙ deriv)))); 
//        deriv ~ exps / (- sum(exps));
        deriv <-- (- output);
        val inds = target + coloffsets;
			  deriv(inds) = deriv(inds) + 1f;               // deriv = target - preds
        inputDeriv ~ inputDeriv + deriv; 
      }
  }
  
  override def score:FMat = {
    if (coloffsets.asInstanceOf[AnyRef] == null) coloffsets = convertMat(irow(0->output.ncols)*output.nrows);
    val inds = target + coloffsets;
    FMat(mean(ln(output(inds))));   
  }
}

object SoftmaxOutputLayer {  
  class Options extends Layer.Options {
   
    override def clone:Options = {copyTo(new Options).asInstanceOf[Options];}
    
    override def create(net:Net):SoftmaxOutputLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new SoftmaxOutputLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new SoftmaxOutputLayer(net, opts);
}
/**
 * Tanh layer. 
 */

class TanhLayer(override val net:Net, override val opts:TanhLayer.Options = new TanhLayer.Options) extends Layer(net, opts) {    

	override def forward = {
			createoutput;
			tanh(inputData, output);
			clearDeriv;
	}

	override def backward = {
			val tmp = tanh(inputData);
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + ((1 - tmp ∘ tmp) ∘ deriv);
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
			ilinks = izeros(inputData.nrows, 1);
			ilinks.set(GLM.logistic);
			ilinks = convertMat(ilinks);
		}
		if (totflops == 0L) totflops = inputData.nrows * GLM.linkArray(1).fnflops;
		output <-- GLM.preds(inputData, ilinks, totflops);
		clearDeriv;
}

  override def backward = {
		val tmp = output - (output ∘ output);
		if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (tmp ∘ deriv);
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
		val big = inputData > 60f;      
		output ~ ((1 - big) ∘ ln(1f + exp(min(60f, inputData)))) + (big ∘ inputData);
		clearDeriv;
  }

  override def backward = {
		if (ilinks.asInstanceOf[AnyRef] == null) {
			ilinks = izeros(inputData.nrows, 1);
			ilinks.set(GLM.logistic);
		}
		if (totflops == 0L) totflops = inputData.nrows * GLM.linkArray(1).fnflops;
		ilinks = convertMat(ilinks);
		if (inputDeriv.asInstanceOf[AnyRef] != null) {
			val tmp = GLM.preds(inputData, ilinks, totflops);
			inputDeriv ~ inputDeriv + (tmp ∘ deriv);
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
			ln(inputData, output);
			clearDeriv;
	}

	override def backward = {
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv/inputData);    
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
			exp(inputData, output);
			clearDeriv;
	}

	override def backward = {
			if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (deriv ∘ output);    
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
		createoutput(1, inputData.ncols);
		output <-- sum(inputData);
		clearDeriv;
  }

  override def backward = {
		if (vmap.asInstanceOf[AnyRef] == null) vmap = deriv.ones(output.nrows, 1);
		if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + (vmap * deriv);    
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
  	  val io = inputData;
  	  output = io.zeros(io.nrows, io.ncols);
  	}
		output <-- inputData;
		clearDeriv;
  }

  override def backward = {
    if (inputDeriv.asInstanceOf[AnyRef] != null) inputDeriv ~ inputDeriv + deriv    
  }
}

object CopyLayer {  
  class Options extends Layer.Options {
    
  	override def clone:Options = {copyTo(new Options).asInstanceOf[Options];}
  	
  	override def create(net:Net):CopyLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new CopyLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new CopyLayer(net, opts);

}

class SplitHorizLayer(override val net:Net, override val opts:SplitHorizLayer.Options = new SplitHorizLayer.Options) extends Layer(net, opts) {
  override val _outputs = new Array[Mat](opts.nparts);
  override val _derivs = new Array[Mat](opts.nparts);
  var nblock:Int = 0;
  var colranges = new Array[Mat](opts.nparts);
  
  override def forward = {
    if (output.asInstanceOf[AnyRef] == null) {
      nblock = inputData.ncols / opts.nparts;
      for (i <- 0 until opts.nparts) {
        colranges(i) = convertMat(irow((i*nblock)->((i+1)*nblock)));
      }
    }
    for (i <- 0 until opts.nparts) {
      setoutput(i, inputData.colslice(i*nblock, (i+1)* nblock));
    }
    clearDerivs;
  }

  override def backward = {
    if (inputDeriv.asInstanceOf[AnyRef] != null) {
      for (i <- 0 until opts.nparts) {
        inputDeriv(?, colranges(i)) = inputDeriv(?, colranges(i)) + derivs(i);
      }
    }   
  }
}

object SplitHorizLayer {  
  class Options extends Layer.Options {
    
    var nparts = 1;
    
    override def clone:Options = {copyTo(new Options).asInstanceOf[Options];}
    
    override def create(net:Net):SplitHorizLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new SplitHorizLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new SplitHorizLayer(net, opts);

}

class SplitVertLayer(override val net:Net, override val opts:SplitVertLayer.Options = new SplitVertLayer.Options) extends Layer(net, opts) {
  override val _outputs = new Array[Mat](opts.nparts);
  override val _derivs = new Array[Mat](opts.nparts);
  var nblock:Int = 0;
  var rowranges = new Array[Mat](opts.nparts);
  
  override def forward = {
    if (output.asInstanceOf[AnyRef] == null) {
      nblock = inputData.nrows / opts.nparts;
      for (i <- 0 until opts.nparts) {
        rowranges(i) = convertMat(icol((i*nblock)->((i+1)*nblock)));
      }
    }
    for (i <- 0 until opts.nparts) {
      setoutput(i, inputData(rowranges(i), ?));
    }
    clearDerivs;
  }

  override def backward = {
    if (inputDeriv.asInstanceOf[AnyRef] != null) {
      for (i <- 0 until opts.nparts) {
        inputDeriv(rowranges(i), ?) = inputDeriv(rowranges(i), ?) + derivs(i);
      }
    }   
  }
}

object SplitVertLayer {  
  class Options extends Layer.Options {
    
    var nparts = 1;
    
    override def clone:Options = {copyTo(new Options).asInstanceOf[Options];}
    
    override def create(net:Net):SplitVertLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new SplitVertLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new SplitVertLayer(net, opts);

}

class StackLayer(override val net:Net, override val opts:StackLayer.Options = new StackLayer.Options) extends Layer(net, opts) {
  override val _inputs = new Array[Layer](opts.ninputs);
  override val _inputTerminals = new Array[Int](opts.ninputs);

  var colranges = new Array[Mat](opts.ninputs);
  
  override def forward = {
    if (output.asInstanceOf[AnyRef] == null) {
      var orows = 0;
      for (i <- 0 until opts.ninputs) {
        val thisrow = inputDatas(i).nrows;
        colranges(i) = convertMat(irow(orows -> (orows + thisrow)));
        orows += thisrow;
      }
      output = convertMat(zeros(orows, inputData.ncols));
    }
    for (i <- 0 until opts.ninputs) {
      output(colranges(i), ?) = inputDatas(i);
    }
    clearDeriv;
  }

  override def backward = {
		for (i <- 0 until opts.ninputs) {
			if (inputDerivs(i).asInstanceOf[AnyRef] != null) {
        inputDerivs(i) <-- deriv(colranges(i), ?)
      }
    }   
  }
}

object StackLayer {  
  class Options extends Layer.Options {
    
    var ninputs = 2;
    override val inputs:Array[Layer.Options] = new Array[Layer.Options](ninputs);
    override val inputTerminals:Array[Int] = new Array[Int](ninputs);
    
    override def clone:Options = {copyTo(new Options).asInstanceOf[Options];}
    
    override def create(net:Net):StackLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new StackLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new StackLayer(net, opts);

}
/*
 * Designed to map linear integer feature arrays to sparse matrices. Doesnt deal with derivatives.
 */

class OnehotLayer(override val net:Net, override val opts:OnehotLayer.Options = new OnehotLayer.Options) extends Layer(net, opts) {

  override def forward = {
    output = oneHot(inputData);
  }
}

object OnehotLayer {  
  class Options extends Layer.Options {
    
    override def clone:Options = {copyTo(new Options).asInstanceOf[Options];}
    
    override def create(net:Net):OnehotLayer = {apply(net, this);}
  }
  
  def apply(net:Net) = new OnehotLayer(net, new Options);
  
  def apply(net:Net, opts:Options) = new OnehotLayer(net, opts);

}

class CompoundLayer(override val net:Net, override val opts:CompoundLayer.Options = new CompoundLayer.Options) extends ModelLayer(net, opts) {
	
	override def setinput(i:Int, v:Layer):CompoundLayer = {               // Assumes the inputs are the first k layers in internal_layers
	  _inputs(i) = v;
	  internal_layers(i).setinput(0, v);
    this
	}
	
  override def setinout(i:Int, v:Layer, j:Int):CompoundLayer = {               // Assumes the inputs are the first k layers in internal_layers
	  _inputs(i) = v;
    _inputTerminals(i) = j;
	  internal_layers(i).setinout(0, v, j);
    this
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
      internal_layers(i) match {
        case aa:LinLayer => aa.opts.aopts = opts.aopts;
        case _ =>
      }
	  }
	}
}

object CompoundLayer {
  class Options extends ModelLayer.Options {  	  
	  var lopts:Array[Layer.Options] = null;
    var aopts:ADAGrad.Opts = null;
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
 


