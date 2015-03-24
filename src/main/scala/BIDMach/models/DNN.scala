package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach._

/**
 * Basic DNN class. Learns a supervised map from input blocks to output (target) data blocks. There are currently 12 layer types:
 - InputLayer: just a placeholder for the first layer which is loaded with input data blocks. No learnable params. 
 - FCLayer: Fully-Connected Linear layer. Has a matrix of learnable params which is the input-output map. 
 - RectLayer: Rectifying one-to-one layer. No params.
 - GLMLayer: a one-to-one layer with GLM mappings (linear, logistic, abs-logistic and SVM). No learnable params. 
 - NormLayer: normalizing layer that adds a derivative term based on the difference between current layer norm and a target norm. 
   No learnable params. The target norm and weight of this derivative term can be specified. 
 - DropoutLayer: A layer that implements random dropout. No learnable params, but dropout fraction can be specified. 
 - AddLayer: adds two input layers element-wise.
 - MulLayer: multiplies two input layers element-wise. Will also perform edge operations (one input can be a scalar). 
 - Softmax: a softmax (normalized exponential) layer.
 - Tanh: Hyperbolic tangent non-linearity.
 - Sigmoid: Logistic function non-linearity.
 - Cut: needed in each cycle of cyclic networks to allow caching to work. 
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
  var layers:Array[Layer] = null

  override def init() = {
	  mats = datasource.next;
	  var nfeats = mats(0).nrows;
	  datasource.reset;
	  layers = new Array[Layer](opts.layers.length);
	  var imodel = 0;
	  if (refresh) {
	  	val nmodelmats = opts.layers.count(_ match {case x:DNN.ModelLayerSpec => true; case _ => false});
	    setmodelmats(new Array[Mat](nmodelmats));
	  }
	  for (i <- 0 until opts.layers.length) {
	  	opts.layers(i) match {

	  	case fcs:DNN.FC => {
	  		layers(i) = new FCLayer(imodel);
	  		if (refresh) modelmats(imodel) = normrnd(0, 1, fcs.outsize, nfeats);
	  		nfeats = fcs.outsize;
	  		imodel += 1;
	  	}
	  	case rls:DNN.ReLU => {
	  		layers(i) = new ReLULayer;
	  	}
	  	case ils:DNN.Input => {
	  		layers(i) = new InputLayer;
	  	}
	  	case ols:DNN.GLM => {
	  		layers(i) = new GLMLayer(ols.links);
	  	}
	  	case nls:DNN.Norm => {
	  		layers(i) = new NormLayer(nls.targetNorm, nls.weight);
	  	}
	  	case dls:DNN.Dropout => {
	  		layers(i) = new DropoutLayer(dls.frac);
	  	}
	  	case als:DNN.Add => {
	  		layers(i) = new AddLayer;
	  	}
	  	case mls:DNN.Mul => {
	  		layers(i) = new MulLayer;
	  	}
	  	case mls:DNN.Softmax => {
	  		layers(i) = new SoftmaxLayer;
	  	}
	  	case tls:DNN.Tanh => {
	  		layers(i) = new TanhLayer;
	  	}
	  	case sls:DNN.Sigmoid => {
	  		layers(i) = new SigmoidLayer;
	  	}
	  	case cls:DNN.Cut => {
	  		layers(i) = new CutLayer;
	  	}
	  	}
	  	opts.layers(i).myLayer = layers(i);
	  }
	  updatemats = new Array[Mat](modelmats.length);
	  for (i <- 0 until modelmats.length) {
	    modelmats(i) = convertMat(modelmats(i));
	    updatemats(i) = modelmats(i).zeros(modelmats(i).nrows, modelmats(i).ncols);
	  } 		
	  for (i <- 0 until opts.layers.length) {
	  	if (opts.layers(i).input.asInstanceOf[AnyRef] != null) layers(i).input = opts.layers(i).input.myLayer.asInstanceOf[DNN.this.Layer];
	  	if (opts.layers(i).input2.asInstanceOf[AnyRef] != null) layers(i).input2 = opts.layers(i).input2.myLayer.asInstanceOf[DNN.this.Layer];
	  }
  }
  
  
  def doblock(gmats:Array[Mat], ipass:Int, i:Long):Unit = {
    layers(0).data = gmats(0);
    layers(layers.length-1).target = gmats(1);
    var i = 1
    while (i < layers.length) {
      layers(i).forward;
      i += 1;
    }
    while (i > 1) {
      i -= 1;
      layers(i).backward;
    }
  }
  
  def evalblock(mats:Array[Mat], ipass:Int, here:Long):FMat = {  
    layers(0).data = gmats(0);
    val targ = gmats(1);
    layers(layers.length-1).target = targ;
    var i = 1;
    while (i < layers.length) {
      layers(i).forward;
      i += 1;
    }
    if (putBack >= 0) {targ <-- layers(layers.length-1).data}
    layers(layers.length-1).score
  }
  
  class Layer {
    var data:Mat = null;
    var target:Mat = null;
    var deriv:Mat = null;
    var input:Layer = null;
    var input2:Layer = null;
    def forward = {};
    def backward = {};
    def score:FMat = zeros(1,1);
  }
  
  /**
   * Full connected layer. 
   * Includes a model matrix that contains the linear map. 
   */
  
  class FCLayer(val imodel:Int) extends Layer {
    override def forward = {
      data = modelmats(imodel) * input.data;
    }
    
    override def backward = {
      if (imodel > 0) input.deriv = modelmats(imodel) ^* deriv;
      updatemats(imodel) = deriv *^ input.data;
    }
  }
  
  /**
   * Rectifying Linear Unit layer.
   */
  
  class ReLULayer extends Layer {
    override def forward = {
      data = max(input.data, 0f)
    }
    
    override def backward = {
      input.deriv = deriv ∘ (input.data > 0f);
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
      data = GLM.preds(input.data, ilinks, totflops)
    }
    
    override def backward = {
      input.deriv = GLM.derivs(data, target, ilinks, totflops)
      if (deriv.asInstanceOf[AnyRef] != null) {
        input.deriv ~ input.deriv ∘ deriv;
      }
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
      data = input.data;  
    }
    
    override def backward = {
    	if (sconst.asInstanceOf[AnyRef] == null) sconst = data.zeros(1,1);
      sconst.set(math.min(0.1f, math.max(-0.1f, (targetNorm - norm(data)/data.length).toFloat * weight)));
      input.deriv = data ∘ sconst;
      input.deriv ~ input.deriv + deriv;
    }
  }
  
  /**
   * Dropout layer with fraction to keep "frac". Deletes the same neurons in forward and backward pass. 
   * Assumes that "randmat" is not changed between forward and backward passes. 
   */
  
  class DropoutLayer(val frac:Float) extends Layer {  
    var randmat:Mat = null;
    
    override def forward = {
      randmat = input.data + 20f;   // Hack to make a cached container to hold the random data
      if (useGPU) {
        grand(randmat.asInstanceOf[GMat]); 
      } else {
        rand(randmat.asInstanceOf[FMat]);
      }
      randmat ~ (randmat < frac)
      data = input.data ∘ randmat;
    }
    
    override def backward = {
      input.deriv = deriv ∘ randmat;
    }
  }
  
  /**
   * Computes the sum of two input layers. 
   */
  
  class AddLayer extends Layer {    
    
    override def forward = {
      data = input.data + input2.data;
    }
    
    override def backward = {
      input.deriv = deriv;
      input2.deriv = deriv;
    }
  }
  
  /**
   * Computes the product of two input layers. 
   */
  
  class MulLayer extends Layer {    
    
    override def forward = {
      data = input.data ∘ input2.data;
    }
    
    override def backward = {
      input.deriv = deriv ∘ input2.data;
      input2.deriv = deriv ∘ input.data;
    }
  }
  
  /**
   * Softmax layer. Output = exp(input) / sum(exp(input))
   */
  
  class SoftmaxLayer extends Layer {    
    
    override def forward = {
      val exps = exp(input.data);
      data = exps / sum(exps);
    }
    
    override def backward = {
      val exps = exp(input.data);
      val sumexps = sum(exps);
      val isum = 1f / (sumexps ∘ sumexps);
      input.deriv = ((exps / sumexps) ∘ deriv) - (exps ∘ (isum ∘ (exps ∙ deriv))) ;
    }
  }
  
  /**
   * Tanh layer. 
   */
  
  class TanhLayer extends Layer {    
    
    override def forward = {
      data = tanh(input.data);
    }
    
    override def backward = {
      val tmp = tanh(input.data);
      input.deriv = (1 - tmp ∘ tmp) ∘ deriv;
    }
  }
  
  /**
   * Sigmoid layer. Uses GLM implementations of logistic functions for performance. 
   */
  
  class SigmoidLayer extends Layer {
    var ilinks:Mat = null;
    var totflops = 0L;
    
    override def forward = {
      if (ilinks.asInstanceOf[AnyRef] == null) {
        ilinks = izeros(input.data.nrows, 1)
        ilinks.set(GLM.logistic);
      }
      if (totflops == 0L) totflops = input.data.nrows * GLM.linkArray(1).fnflops
      if (useGPU) ilinks = GIMat(ilinks);
      data = GLM.preds(input.data, ilinks, totflops)
    }
    
    override def backward = {
      input.deriv = GLM.derivs(data, target, ilinks, totflops)
      if (deriv.asInstanceOf[AnyRef] != null) {
        input.deriv ~ input.deriv ∘ deriv;
      }
    }
  }
  
  /**
   * Cut layer. Need to insert these in cyclic networks so that caching works. 
   */
  
  class CutLayer extends Layer {
    
    override def forward = {
      if (data.asInstanceOf[AnyRef] == null) {
        data = input.data.zeros(input.data.nrows, input.data.ncols);
        input.deriv = input.data.zeros(input.data.nrows, input.data.ncols);
      }
      data <-- input.data;
    }
    
    override def backward = {
      input.deriv <-- deriv;      
    }
  }
}

object DNN  {
  trait Opts extends Model.Opts {
	var layers:Seq[LayerSpec] = null;
    var links:IMat = null;
    var nweight:Float = 0.1f;
    var dropout:Float = 0.5f;
    var targetNorm:Float = 1f;
  }
  
  class Options extends Opts {}
  
  class LayerSpec(val input:LayerSpec, val input2:LayerSpec) {
    var myLayer:DNN#Layer = null;
  }
  
  class ModelLayerSpec(input:LayerSpec) extends LayerSpec(input, null){}
  
  class FC(input:LayerSpec, val outsize:Int) extends ModelLayerSpec(input) {}
  
  class ReLU(input:LayerSpec) extends LayerSpec(input, null) {}
  
  class Input extends LayerSpec(null, null) {}
  
  class GLM(input:LayerSpec, val links:IMat) extends LayerSpec(input, null) {}
  
  class Norm(input:LayerSpec, val targetNorm:Float, val weight:Float) extends LayerSpec(input, null) {}
  
  class Dropout(input:LayerSpec, val frac:Float) extends LayerSpec(input, null) {}
  
  class Add(input:LayerSpec, input2:LayerSpec) extends LayerSpec(input, input2) {}
  
  class Mul(input:LayerSpec, input2:LayerSpec) extends LayerSpec(input, input2) {}
  
  class Softmax(input:LayerSpec) extends LayerSpec(input, null) {}
  
  class Tanh(input:LayerSpec) extends LayerSpec(input, null) {}
  
  class Sigmoid(input:LayerSpec) extends LayerSpec(input, null) {}
  
  class Cut(input:LayerSpec) extends LayerSpec(input, null) {}
  
  /**
   * Build a stack of layer specs. layer(0) is an input layer, layer(n-1) is a GLM layer. 
   * Intermediate layers are FC alternating with ReLU, starting and ending with FC. 
   * First FC layer width is given as an argument, then it tapers off by taper.
   */
  
  def dlayers(depth0:Int, width:Int, taper:Float, ntargs:Int, opts:Opts):Array[LayerSpec] = {
    val depth = (depth0/2)*2 + 1;              // Round up to an odd number of layers 
    val layers = new Array[LayerSpec](depth);
    var w = width;
    layers(0) = new Input;
    for (i <- 1 until depth - 2) {
    	if (i % 2 == 1) {
    		layers(i) = new FC(layers(i-1), w);
    		w = (taper*w).toInt;
    	} else {
    		layers(i) = new ReLU(layers(i-1));
    	}
    }
    layers(depth-2) = new FC(layers(depth-3), ntargs);
    layers(depth-1) = new GLM(layers(depth-2), opts.links);
    opts.layers = layers
    layers
  }
  
  /**
   * Build a stack of layer specs. layer(0) is an input layer, layer(n-1) is a GLM layer. 
   * Intermediate layers are FC, ReLU, Norm, starting and ending with FC. 
   * First FC layer width is given as an argument, then it tapers off by taper.
   */
  
  def dlayers3(depth0:Int, width:Int, taper:Float, ntargs:Int, opts:Opts):Array[LayerSpec] = {
    val depth = (depth0/3)*3;              // Round up to a multiple of 3 
    val layers = new Array[LayerSpec](depth);
    var w = width;
    layers(0) = new Input;
    for (i <- 1 until depth - 2) {
    	if (i % 3 == 1) {
    		layers(i) = new FC(layers(i-1), w);
    		w = (taper*w).toInt;
    	} else if (i % 3 == 2) {
    		layers(i) = new ReLU(layers(i-1));
    	} else {
    	  layers(i) = new Norm(layers(i-1), opts.targetNorm, opts.nweight);
    	}
    }
    layers(depth-2) = new FC(layers(depth-3), ntargs);
    layers(depth-1) = new GLM(layers(depth-2), opts.links);
    opts.layers = layers
    layers
  }
  
  /**
   * Build a stack of layer specs. layer(0) is an input layer, layer(n-1) is a GLM layer. 
   * Intermediate layers are FC, ReLU, Norm, Dropout, starting and ending with FC. 
   * First FC layer width is given as an argument, then it tapers off by taper.
   */
  
  def dlayers4(depth0:Int, width:Int, taper:Float, ntargs:Int, opts:Opts):Array[LayerSpec] = {
    val depth = ((depth0+1)/4)*4 - 1;              // Round up to a multiple of 4 - 1
    val layers = new Array[LayerSpec](depth);
    var w = width;
    layers(0) = new Input;
    for (i <- 1 until depth - 2) {
      (i % 4) match {
        case 1 => {
        	layers(i) = new FC(layers(i-1), w);
        	w = (taper*w).toInt;
        }
        case 2 => {
        	layers(i) = new ReLU(layers(i-1));
        }
        case 3 => {
          layers(i) = new Norm(layers(i-1), opts.targetNorm, opts.nweight);
        }
        case _ => {
          layers(i) = new Dropout(layers(i-1), opts.dropout);          
        }
      }
    }
    layers(depth-2) = new FC(layers(depth-3), ntargs);
    layers(depth-1) = new GLM(layers(depth-2), opts.links);
    opts.layers = layers
    layers
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
  	    null,
  	    new ADAGrad(opts), 
  	    opts)
    (nn, opts)
  }

  def learner(fnames:List[(Int)=>String]) = {
    class xopts extends Learner.Options with DNN.Opts with SFilesDS.Opts with ADAGrad.Opts
    val opts = new xopts
    opts.fnames = fnames
    opts.batchSize = 100000;
    opts.eltsPerSample = 500;
    implicit val threads = threadPool(4);
    val ds = new SFilesDS(opts)
//    dlayers(3, 0, 1f, targ.nrows, opts)                   // default to a 3-layer network
  	val nn = new Learner(
  			ds, 
  	    new DNN(opts), 
  	    null,
  	    new ADAGrad(opts), 
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
    val opts = new LearnOptions;
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    opts.links = model.opts.links;
    opts.layers = model.opts.layers;
    opts.addConstFeat = model.opts.asInstanceOf[DataSource.Opts].addConstFeat;
    opts.putBack = 1;
    opts.dropout = 1f;
    
    val newmod = new DNN(opts);
    newmod.refresh = false;
    newmod.copyFrom(model)
    val nn = new Learner(
        new MatDS(Array(mat0, preds), opts), 
        newmod, 
        null,
        null, 
        opts);
    (nn, opts)
  }
}


