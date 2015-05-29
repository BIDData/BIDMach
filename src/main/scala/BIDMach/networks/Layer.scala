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


  
class Layer(val net:Net, val spec:Layer.Spec = new Layer.Spec) {
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
  lazy val modelmats = net.modelmats;
  lazy val updatemats = net.updatemats;
  lazy val useGPU = net.useGPU;
  lazy val nopts = net.opts;
  def convertMat(mat:Mat) = {net.convertMat(mat);}

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
  class Spec{
    val inputs:Array[Spec] = Array(null);
    var myLayer:Layer = null;
    var myGhost:Spec = null;
    
    def copyTo(spec:Spec):Spec = {
      spec.inputs(0) = inputs(0);
      myGhost = spec;
      spec;
    }
    
    override def clone:Spec = {
      copyTo(new Spec);
    }
    
    def create(net:Net):Layer = {null}
  }
}

class ModelLayer(override val net:Net, override val spec:ModelLayer.Spec = new ModelLayer.Spec) extends Layer(net, spec) {
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
  class Spec extends Layer.Spec {
  	var modelName = "";
    var imodel = 0;
    
    def copyTo(spec:Spec):Spec = {
      super.copyTo(spec);
      spec.modelName = modelName;
      spec.imodel = imodel;
      spec;
    }
    
    override def clone:Spec = {
      copyTo(new Spec);
    }
  }
}
/**
 * Linear layer. 
 * Includes a model matrix that contains the linear map. 
 */

class LinLayer(override val net:Net, override val spec:LinLayer.Spec = new LinLayer.Spec) extends ModelLayer(net, spec) {
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
  		modelmats(imodel) = convertMat(normrnd(0, 1, spec.outdim, input.output.nrows + (if (spec.constFeat) 1 else 0)));
  		updatemats(imodel) = modelmats(imodel).zeros(modelmats(imodel).nrows, modelmats(imodel).ncols);
  		if (spec.aopts != null) initADAGrad;  
  	}
  	val mm = if (spec.constFeat) {
  		modelmats(imodel).colslice(1, modelmats(imodel).ncols);
  	} else {
  		modelmats(imodel);
  	}
  	createoutput(mm.nrows, input.output.ncols);
  	output ~ mm * input.output;
  	if (spec.constFeat) output ~ output + modelmats(imodel).colslice(0, 1);
  	clearDeriv;
  }

  override def backward(ipass:Int, pos:Long) = {
  	val mm = if (spec.constFeat && imodel > 0) {
  		modelmats(imodel).colslice(1, modelmats(imodel).ncols);
  	} else {
  		modelmats(imodel);
  	}
  	if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + (mm ^* deriv);
  	if (spec.aopts != null) {
  		if (firststep <= 0) firststep = pos.toFloat;
  		val istep = (pos + firststep)/firststep;
  		ADAGrad.multUpdate(deriv, input.output, modelmats(imodel), sumsq, mask, lrate, texp, vexp, epsilon, istep, waitsteps);
  	} else {
  		val dprod = deriv *^ input.output;
  		updatemats(imodel) ~ updatemats(imodel) + (if (spec.constFeat) (sum(deriv,2) \ dprod) else dprod);
  	}
  }


  def initADAGrad {
  	val aopts = spec.aopts;
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
  class Spec extends ModelLayer.Spec {
    var constFeat:Boolean = false;
    var aopts:ADAGrad.Opts = null;
    var outdim = 0;
    
    def copyTo(spec:Spec):Spec = {
    	super.copyTo(spec);
    	spec.constFeat = constFeat;
    	spec.aopts = aopts;
    	spec.outdim = outdim;
    	spec;
    }
    
    override def clone:Spec = {
    	copyTo(new Spec);
    }
    
    override def create(net:Net):LinLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new LinLayer(net, new Spec);
  
  def apply(net:Net, spec:Spec):LinLayer = {
    val x = new LinLayer(net, spec); 
    x.modelName = spec.modelName;
    x.imodel = x.getModelMat(net, spec.modelName, spec.imodel)  
    x;}
  
}

/**
 * Rectifying Linear Unit layer.
 */

class ReLULayer(override val net:Net, override val spec:ReLULayer.Spec = new ReLULayer.Spec) extends Layer(net, spec) {
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
  class Spec extends Layer.Spec {
  	override def clone:Spec = {
    	copyTo(new Spec).asInstanceOf[Spec];
    }
  	
  	override def create(net:Net):ReLULayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new ReLULayer(net, new Spec);
  
  def apply(net:Net, spec:Spec) = new ReLULayer(net, spec);
}

/**
 * Input layer is currently just a placeholder.
 */

class InputLayer(override val net:Net, override val spec:InputLayer.Spec = new InputLayer.Spec) extends Layer(net, spec) {
}

object InputLayer {  
  class Spec extends Layer.Spec {
  	override def clone:Spec = {
    	copyTo(new Spec).asInstanceOf[Spec];
    }
  	
  	override def create(net:Net):InputLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new InputLayer(net, new Spec);
  
  def apply(net:Net, spec:Spec) = new InputLayer(net, spec);
}

/**
 * GLMLayer implements linear, logistic and hinge-loss SVM. 
 * Commonly used as an output layer so includes a score method.
 */

class GLMLayer(override val net:Net, override val spec:GLMLayer.Spec = new GLMLayer.Spec) extends Layer(net, spec) {
	var ilinks:Mat = null;
	var totflops = 0L;

	override def forward = {
			createoutput;
			if (ilinks.asInstanceOf[AnyRef] == null) {
			  ilinks = convertMat(spec.links);
			  for (i <- 0 until spec.links.length) {
			  	totflops += GLM.linkArray(spec.links(i)).fnflops
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
  class Spec extends Layer.Spec {
    var links:IMat = null;
    
    def copyTo(spec:Spec):Spec = {
  		super.copyTo(spec);
  		spec.links = links;
  		spec;
    }
    
    override def clone:Spec = {
  	  copyTo(new Spec);
    }
    
    override def create(net:Net):GLMLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new GLMLayer(net, new Spec);
  
  def apply(net:Net, spec:Spec) = new GLMLayer(net, spec); 

}

/**
 * Normalization layer adds a downward-propagating derivative term whenever its norm 
 * is different from the specified value (targetNorm).
 */

class NormLayer(override val net:Net, override val spec:NormLayer.Spec = new NormLayer.Spec) extends Layer(net, spec) {
	var sconst:Mat = null;

  override def forward = {
		createoutput;
		output <-- input.output;
		clearDeriv;
  }

  override def backward = {
    if (input.deriv.asInstanceOf[AnyRef] != null) {
    	if (sconst.asInstanceOf[AnyRef] == null) sconst = output.zeros(1,1);
    	sconst.set(math.min(0.1f, math.max(-0.1f, (spec.targetNorm - norm(output)/output.length).toFloat * spec.weight)));
    	input.derivs(0) = output ∘ sconst;
    	input.deriv ~ input.deriv + deriv;
    }
  }
}

object NormLayer {  
  class Spec extends Layer.Spec {
    var targetNorm = 1f;
    var weight = 1f;
    
    def copyTo(spec:Spec):Spec = {
  		super.copyTo(spec);
  		spec.targetNorm = targetNorm;
  		spec.weight = weight;
  		spec;
    }
    
    override def clone:Spec = {
  	  copyTo(new Spec);
    }
    
    override def create(net:Net):NormLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new NormLayer(net, new Spec);
  
  def apply(net:Net, spec:Spec) = new NormLayer(net, spec);  
}

/**
 * Dropout layer with fraction to keep "frac". Deletes the same neurons in forward and backward pass. 
 * Assumes that "randmat" is not changed between forward and backward passes. 
 */

class DropoutLayer(override val net:Net, override val spec:DropoutLayer.Spec = new DropoutLayer.Spec) extends Layer(net, spec) {  
	var randmat:Mat = null;

  override def forward = {
		createoutput;
		randmat = input.output + 20f;   // Hack to make a cached container to hold the random output
		if (nopts.predict) {
			output ~ input.output * spec.frac;
		} else {
			if (useGPU) {
				grand(randmat.asInstanceOf[GMat]); 
			} else {
				rand(randmat.asInstanceOf[FMat]);
			}
			randmat ~ (randmat < spec.frac)
			output ~ input.output ∘ randmat;
		}
		clearDeriv;
  }

  override def backward = {
		if (input.deriv.asInstanceOf[AnyRef] != null) input.deriv ~ input.deriv + deriv ∘ randmat;
  }
}

object DropoutLayer {  
  class Spec extends Layer.Spec {
    var frac = 1f;
    
    def copyTo(spec:Spec):Spec = {
  		super.copyTo(spec);
  		spec.frac = frac;
  		spec;
    }
    
    override def clone:Spec = {
  	  copyTo(new Spec);
    }
    
    override def create(net:Net):DropoutLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new DropoutLayer(net, new Spec);
  
  def apply(net:Net, spec:Spec) = new DropoutLayer(net, spec);
}

/**
 * Computes the sum of input layers. 
 */

class AddLayer(override val net:Net, override val spec:AddLayer.Spec = new AddLayer.Spec) extends Layer(net, spec) { 
  
  override val inputs = new Array[Layer](spec.ninputs);

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
  class Spec extends Layer.Spec {
    var ninputs = 1;
    
    def copyTo(spec:Spec):Spec = {
  		super.copyTo(spec);
  		spec.ninputs = ninputs;
  		spec;
    }
    
    override def clone:Spec = {
  	  copyTo(new Spec);
    }
    
    override def create(net:Net):AddLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new AddLayer(net, new Spec);
  
  def apply(net:Net, spec:Spec) = new AddLayer(net, spec); 
}

/**
 * Computes the product of its input layers. 
 */

class MulLayer(override val net:Net, override val spec:MulLayer.Spec = new MulLayer.Spec) extends Layer(net, spec) {  
  
	override val inputs = new Array[Layer](spec.ninputs);

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
  class Spec extends Layer.Spec {
    var ninputs = 1;
    
    def copyTo(spec:Spec):Spec = {
  		super.copyTo(spec);
  		spec.ninputs = ninputs;
  		spec;
    }
    
    override def clone:Spec = {
  	  copyTo(new Spec);
    }
    
    override def create(net:Net):MulLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new MulLayer(net, new Spec);
  
  def apply(net:Net, spec:Spec) = new MulLayer(net, spec); 
}

/**
 * Softmax layer. Output = exp(input) / sum(exp(input))
 */

class SoftmaxLayer(override val net:Net, override val spec:SoftmaxLayer.Spec = new SoftmaxLayer.Spec) extends Layer(net, spec) {    

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
  class Spec extends Layer.Spec {
   
    override def clone:Spec = {
  	  copyTo(new Spec).asInstanceOf[Spec];
    }
    
    override def create(net:Net):SoftmaxLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new SoftmaxLayer(net, new Spec);
  
  def apply(net:Net, spec:Spec) = new SoftmaxLayer(net, spec);
}
/**
 * Tanh layer. 
 */

class TanhLayer(override val net:Net, override val spec:TanhLayer.Spec = new TanhLayer.Spec) extends Layer(net, spec) {    

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
  class Spec extends Layer.Spec {
    
    override def clone:Spec = {
  	  copyTo(new Spec).asInstanceOf[Spec];
    }
    
    override def create(net:Net):TanhLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new TanhLayer(net, new Spec);
  
  def apply(net:Net, spec:Spec) = new TanhLayer(net, spec);
}

/**
 * Sigmoid layer. Uses GLM implementations of logistic functions for performance. 
 */

class SigmoidLayer(override val net:Net, override val spec:SigmoidLayer.Spec = new SigmoidLayer.Spec) extends Layer(net, spec) {
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
  class Spec extends Layer.Spec {
    
    override def clone:Spec = {
  	  copyTo(new Spec).asInstanceOf[Spec];
    }
    
    override def create(net:Net):SigmoidLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new SigmoidLayer(net, new Spec);
  
  def apply(net:Net, spec:Spec) = new SigmoidLayer(net, spec); 
}
/**
 * Softplus layer.  
 */

class SoftplusLayer(override val net:Net, override val spec:SoftplusLayer.Spec = new SoftplusLayer.Spec) extends Layer(net, spec) {
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
  class Spec extends Layer.Spec {
    
    override def clone:Spec = {
  	  copyTo(new Spec).asInstanceOf[Spec];
    }
    
    override def create(net:Net):SoftplusLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new SoftplusLayer(net, new Spec);
  
  def apply(net:Net, spec:Spec) = new SoftplusLayer(net, spec); 
}
/**
 * Natural Log layer. 
 */

class LnLayer(override val net:Net, override val spec:LnLayer.Spec = new LnLayer.Spec) extends Layer(net, spec) {

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
  class Spec extends Layer.Spec {
    
  	override def clone:Spec = {
  	  copyTo(new Spec).asInstanceOf[Spec];
    }
  	
  	override def create(net:Net):LnLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new LnLayer(net, new Spec);
  
  def apply(net:Net, spec:Spec) = new LnLayer(net, spec);
}

/**
 * Exponential layer. 
 */

class ExpLayer(override val net:Net, override val spec:ExpLayer.Spec = new ExpLayer.Spec) extends Layer(net, spec) {

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
  class Spec extends Layer.Spec {
    
  	override def clone:Spec = {
  	  copyTo(new Spec).asInstanceOf[Spec];
    }
  	
    override def create(net:Net):ExpLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new ExpLayer(net, new Spec);
  
  def apply(net:Net, spec:Spec) = new ExpLayer(net, spec);

}
/**
 * Sum layer. 
 */

class SumLayer(override val net:Net, override val spec:SumLayer.Spec = new SumLayer.Spec) extends Layer(net, spec) {
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
  class Spec extends Layer.Spec {
    
  	override def clone:Spec = {
  	  copyTo(new Spec).asInstanceOf[Spec];
    }
  	
  	override def create(net:Net):SumLayer = {
      apply(net, this);
    }
  }
  
  def apply(net:Net) = new SumLayer(net, new Spec);
  
  def apply(net:Net, spec:Spec) = new SumLayer(net, spec);

}

class LayerSpec(val nlayers:Int) {
  
  val layerSpecs = new Array[Layer.Spec](nlayers);
  
  def apply(i:Int):Layer.Spec = layerSpecs(i);
  
  def update(i:Int, lspec:Layer.Spec) = {layerSpecs(i) = lspec; this}
}
 


