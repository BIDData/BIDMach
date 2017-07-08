package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,ND,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach._
import edu.berkeley.bid.CPUMACH
import edu.berkeley.bid.CUMACH
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import scala.util.hashing.MurmurHash3;
import java.util.HashMap;
import BIDMach.networks._
import akka.actor.{Actor,Props,ActorSystem,ActorRef};

/**
 * Net Layer class. There are currently 46 layer types:
 * 
 - AbsLayer: Absolute value of input. 
 - AddLayer: Sum two inputs element-wise.
 - BatchNormLayer: Batch normalization.
 - BatchNormScaleLayer: Batch normalization with scaling and bias.
 - ConstantLayer: Wraps a constant or externally-set matrix (can be updated between forward steps). 
 - ConvLayer: Convolution Layer.
 - CopyLayer: Copies inputs forward and derivatives backward.
 - CropLayer: Crops its input. 
 - DivLayer: Computes the quotient of its two inputs. 
 - DotLayer: Dot product (element-wise product followed by sum over columns).
 - DropoutLayer: A layer that implements random dropout. No learnable params, but dropout fraction can be specified.  
 - ExpLayer: exponential function of input.
 - FnLayer: Applies its fwdfn argument to one input. Can also use bwdfn1 and bwdfn2 for backward data and model updates. 
 - Fn2Layer: Applies its fwdfn argument to two inputs. Can also use bwdfn1 and bwdfn2 for backward data and model updates. 
 - FormatLayer: Corrects Tensor Format (NCHW or NHWC)
 - ForwardLayer: Goes forward only.
 - GLMLayer: a one-to-one layer with GLM mappings (linear, logistic, abs-logistic and SVM). No learnable params. 
 - InputLayer: just a placeholder for the first layer which is loaded with input output blocks. No learnable params. 
 - LinLayer: Linear layer. Has a matrix of learnable params which is the input-output map. 
 - LnLayer: natural logarithm.
 - LSTMLayer: an LSTM layer.
 - MaxLayer: computes the max of two inputs;
 - MaxiLayer: computes the max element in each column;
 - Maxi2Layer: computes the max element in each column as output(0), output(1) is a row of integer indices of the maxs.
 - MinLayer: computes the min of two inputs;
 - MiniLayer: computes the min element in each column;
 - Mini2Layer: computes the min element in each column as output(0), output(1) is a row of integer indices of the mins.
 - MulLayer: computes the product of its inputs. 
 - NegsampOutputLayer: a negative sampling Output Layer. 
 - NormLayer: normalizing layer that adds a derivative term based on the difference between current layer norm and a target norm. 
 - MulLayer: multiplies input layers element-wise. Will also perform edge operations (one input can be a scalar).
 - OnehotLayer: Converts an integer array of feature values to a sparse matrix whose columns are the instances with one non-zero in the feature position.
 - PoolLayer: Pooling layer.
 - RectLayer: Rectifying one-to-one layer. No params. 
 - ScaleLayer: Scale-Bias layer, usually used with BatchNorm. 
 - SelectLayer: Use second (integer) matrix input to select elements from columns of first input. 
 - SoftmaxLayer: a softmax (normalized exponential) layer.
 - SigmoidLayer: Logistic function non-linearity.
 - SoftplusLayer: smooth ReLU unit. 
 - SplitHorizLayer: Split the input into nparts horizontally.
 - SplitVertLayer: Split the input into nparts vertically. 
 - SqrtLayer: Square root layer. 
 - StackLayer: Vertically stack its inputs. 
 - SubLayer: Subtract inputs > 1 from first input. 
 - SumLayer: column-wise sum.
 - TanhLayer: Hyperbolic tangent non-linearity.
 * 
 * Shorthand operators and functions for Layer creation have been defined in Node.scala and Layer.scala. 
 * They are as follows (x and y are floating point input NodeTerms or LayerTerms, i is an integer input matrix):
 * 
 * 
 - AbsLayer:                     abs(y) 
 - AddLayer:                     x + y
 - BatchNormLayer:               batchnorm(x)
 - BatchNormScaleLayer:          bns(x)
 - ConstantLayer:                const(1e-3f)
 - ConvLayer:                    conv(x)(args...)
 - CopyLayer:                    copy(x)
 - CropLayer:                    crop(x)(args...)
 - DivLayer:                     x / y 
 - DotLayer:                     x dot y     or     x ∙ y
 - DropoutLayer:                 dropout(x) 
 - ExpLayer:                     exp(x)
 - FnLayer:                      fn(x)(fwdfn=fnname)
 - Fn2Layer:                     fn2(x)(fwdfn=fnname) 
 - FormatLayer:                  format(x)(args...)
 - ForwardLayer:                 forward(x)
 - GLMLayer:                     glm(x) 
 - InputLayer:                   input 
 - LinLayer:                     linear(x)(args...) 
 - LnLayer:                      ln(x)
 - LSTMLayer:                    lstm(x,y,z,name(string))
 - MaxLayer:                     max(x, y)
 - MaxiLayer:                    maxi(x)
 - Maxi2Layer:                   maxi2(x)
 - MinLayer:                     min(x,y)
 - MiniLayer:                    mini(x)
 - Mini2Layer:                   mini2(x)
 - MulLayer:                     x *@ y 
 - NegsampOutputLayer:           negsamp(x) 
 - NormLayer:                    norm(x) 
 - OnehotLayer:                  onehot(x)
 - PoolLayer:                    pool(x)(args...)
 - RectLayer:                    rect(x)    or    relu(x) 
 - ScaleLayer:                   scale(x)
 - SelectLayer:                  x(i) 
 - SoftmaxLayer:                 softmax(x)
 - SigmoidLayer:                 sigmoid(x)
 - SoftplusLayer:                softplus(x) 
 - SplitHorizLayer:              splithoriz(x)(nparts=...)
 - SplitVertLayer:               splitvert(x)(nparts=...)
 - SqrtLayer:                    sqrt(x)
 - StackLayer:                   x on y
 - SubLayer:                     x - y 
 - SumLayer:                     sum(x)
 - TanhLayer:                    tanh(x)
 * 
 * Many of these functions need a parent Net instance, but you can create that automatically using:
 * 
 * 	 import BIDMach.networks.layers.Layer._;
 *   Net.initDefault(opts);
 *
 * then define some layers using the functions above:
 * 
 *     val input1 = input;
 *     val input2 = input;
 *     val const1 = constant(5f);
 *     val s2 =     sum(input1, input2) *@ const1;
 *     ...          ...
 * 
 * Finally, to retrieve the Net instance containing these layers, do:
 * 
 *   val net = Net.getDefaultNet;
 * 
 * The network topology is normally specified by opts.layers which is a sequence of "Layer.Options" objects. There is a nested Options
 * Class for each Layer class, which holds the params for defining that layer, and pointers to any input Layers via their Options classes.
 * In other words, the options classes allow construction of a mirror of the actual network topology. This allows patterns of
 * structure to be repeated using a single Options graph structure. 
 * 
 * Each NodeSet instance has up to two inputs which are other NodeSet instances (or null). This graph structure can be cyclic. 
 * When the model is created, the Layer structure mimics the NodeSet structure. 
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

@SerialVersionUID(100L)
class Layer(val net:Net, val opts:NodeOpts = new Node) extends LayerTerm(null, 0) {
  // Internal data arrays
  val _inputs = new Array[LayerTerm](1);
  val _outputs = new Array[Mat](1);
  val _derivs = new Array[Mat](1);
  def inputlength = _inputs.length;
  def outputlength = _outputs.length;
  var forwardtime = 0.0
  var backwardtime = 0.0
  override def layer = this
  def inputs = _inputs;
  var doreturn = true;
  var myActor:ActorRef = null;
  
  private var _GUID = Mat.myrand.nextLong
  def setGUID(v:Long):Unit = {_GUID = v}
  def GUID:Long = _GUID
  
  val myGPU = getGPU;
  
  // Setters and getters for general elements of those arrays
  def outputs(i:Int) = _outputs(i);
  def derivs(i:Int) = _derivs(i);  
  def input(i:Int) = _inputs(i);
  def apply(i:Int) = new LayerTerm(this, i);
  
  def setOutput(i:Int, v:Mat):Layer = {_outputs(i) = v; this}
  def setDeriv(i:Int, v:Mat):Layer = {_derivs(i) = v; this}
  def setInput(i:Int, v:LayerTerm) = {_inputs(i) = v; this}
  def setInputs(v0:LayerTerm, v1:LayerTerm) = {setInput(0, v0); setInput(1, v1); this}
  def setInputs(v0:LayerTerm, v1:LayerTerm, v2:LayerTerm) = {setInput(0, v0); setInput(1, v1); setInput(2, v2); this}
  
  // Setters and getters for the first input or output
  def input = _inputs(0);
  def output = _outputs(0);
  def deriv = _derivs(0);
  
  def input_=(v:LayerTerm): Unit = {_inputs(0) = v}
  def output_= (v:Mat):Unit = {_outputs(0) = v};
  def deriv_=(v:Mat):Unit = {_derivs(0) = v};
  
  // Input getters (and one setter) which get the appropriate output from each input layer
  def inputData = {val i = _inputs(0); i.layer._outputs(i.term);}
  def inputDeriv = {val i = _inputs(0); i.layer._derivs(i.term);}
  def inputDeriv_=(v:Mat):Unit = {val i = _inputs(0); i.layer._derivs(i.term) = v;}  
  def inputDatas(i:Int) = {val lt = _inputs(i); lt.layer._outputs(lt.term);}
  def inputDerivs(i:Int) = {val lt = _inputs(i); lt.layer._derivs(lt.term);}
  def setInputDeriv(i:Int,m:Mat) = {val inn = _inputs(i); inn.layer._derivs(inn.term) = m;}
  
  var target:Mat = null;
  def forward = {};
  def backward:Unit = {};
  def backward(ipass:Int, pos:Long):Unit = backward;
  def score:FMat = zeros(1,1);
  var parent:Layer = null;
  lazy val modelmats = net.modelmats;
  lazy val updatemats = net.updatemats;
  lazy val lr_scales = net.lr_scales;
  lazy val useGPU = net.useGPU;
  lazy val nopts = net.opts;
  def convertMat(mat:Mat) = {net.convertMat(mat);}

  def createOutput = {
  	if (output.asInstanceOf[AnyRef] == null) output = inputData.zeros(inputData.dims);
  }

  def createOutput(dims:IMat) = {
  	if (output.asInstanceOf[AnyRef] == null) output = inputData.zeros(dims);
  }

  def clearDeriv = {
  	if (deriv.asInstanceOf[AnyRef] == null) deriv = output.zeros(output.dims);
  	deriv.clear;
  }
  
  def clearDerivLazy = {
    if (inputDeriv.asInstanceOf[AnyRef] != null) {
    	if (deriv.asInstanceOf[AnyRef] == null) deriv = output.zeros(output.dims);
    	deriv.clear;
    }
  }
  
  def clearDerivs = {
    if (deriv.asInstanceOf[AnyRef] == null) {
      for (i <- 0 until _outputs.length) {
        _derivs(i) = output.zeros(_outputs(i).dims);
      }
    }
    for (i <- 0 until _derivs.length) {
      _derivs(i).clear
    }
  }
  
  def clearMats = {
    for (i <- 0 until _outputs.length) {
      _outputs(i) = null;
      _derivs(i) = null;
      target = null;
    }
  }
  
  def clear = clearMats
  
  def getModelMats(net:Net):Unit = {}
  
  override def toString = {
    "layer@"+(hashCode % 0x10000).toString
  }
  
  /**
   * When a layer implements a broadcast-compatible op, e.g. +, *@, this reduces the output derivative to match the smaller input deriv dimension.
   */
 
  def squash(a:Mat, b:Mat):Mat = {
  	if (b.nrows == 1 && a.nrows > 1) {
  		if (b.ncols == 1 && a.ncols > 1) {
  			BIDMat.SciFunctions.sum(BIDMat.SciFunctions.sum(a));
  		} else {
  			BIDMat.SciFunctions.sum(a);
  		}
  	} else {
  		a;
  	}
  }
  
  /**
   * For backward caching, check whether any input derivative is non-zero, i.e. if there is any need to back-prop for this layer. 
   */
  
  def anyNonNullDeriv = {
  	var onegood = false;
  	var i = 0;
  	while (! onegood && i < inputlength) {
  		onegood = (inputDerivs(i).asInstanceOf[AnyRef] != null);
  		i += 1;
  	}
  	onegood;
  }
  
  /**
   * Setup output and output derivative for Connected (inplace) layers. 
   */ 
  
  def inplaceConnectGetOutput(forceOut:Boolean = false) = {
  	val inplace = Net.getPlacing(opts.inplace, net.opts.inplace);
  	if (inplace == Net.NoInPlace) {
  		createOutput;  
  	} else {
  		output = inputData; 
  	}
  	inplaceConnectSetupDerivs(forceOut);
  }
  
  /**
   * Setup output and output derivative for Non-Connected (not inplace) layers. 
   */
  
  def inplaceNoConnectGetOutput(forceOut:Boolean = false) = {
  	val inplace = Net.getPlacing(opts.inplace, net.opts.inplace);
  	createOutput;
  	inplaceNoConnectSetupDerivs(forceOut);
  }
  
  def inplaceConnectSetupDerivs(forceOut:Boolean = false) = {
  	val inplace = Net.getPlacing(opts.inplace, net.opts.inplace);
  	if (inplace == Net.NoInPlace || !doreturn){
  		clearDeriv;
  	} else if (inplace == Net.InPlace) {
  	  deriv = inputDeriv;
  	}	else if (anyNonNullDeriv || forceOut) {
  		for (i <- 0 until outputlength) {
  			setDeriv(i, ?);
  		}
  	}
  }
  
  def inplaceNoConnectSetupDerivs(forceOut:Boolean = false) = {
  	val inplace = Net.getPlacing(opts.inplace, net.opts.inplace);
  	if (inplace == Net.NoInPlace || inplace == Net.InPlace || !doreturn){
  		clearDeriv;
  	}	else if (anyNonNullDeriv || forceOut) {
  		for (i <- 0 until outputlength) {
  			setDeriv(i, ?);
  		}
  	}
  }
  

  
  /**
   * Set up input derivative matrices for connected layers. Assumes getMat zeros the matrix. 
   * The GUID for each input derivative should be unique to this layer and input derivative number. 
   */

  def inplaceConnectGetInputDerivs() = {
  	val inplace = Net.getPlacing(opts.inplace, net.opts.inplace);
  	if (inplace == Net.BackwardCaching) {
  	  inputDeriv = deriv;
  		for (i <- 1 until inputlength) {
  			if (inputDerivs(i).asInstanceOf[AnyRef] == ?) {
  			  val newmat = net.getMat(inputDatas(i).dims, inputData);
  			  newmat.setGUID(ND.hash2(GUID, i));
  				setInputDeriv(i, newmat);
  			}
  		}
  	}
  }
  
  /**
   * Set up input derivative matrices for non-connected layers. Assumes getMat zeros the matrix. 
   * The GUID for each input derivative should be unique to this layer and input derivative number. 
   */
  
  def inplaceNoConnectGetInputDerivs() = {
  	val inplace = Net.getPlacing(opts.inplace, net.opts.inplace);
  	if (inplace == Net.BackwardCaching) {
  		for (i <- 0 until inputlength) {
  			if (inputDerivs(i).asInstanceOf[AnyRef] == ?) {
  			  val newmat = net.getMat(inputDatas(i).dims, inputData);
  			  newmat.setGUID(ND.hash2(GUID, i));
  				setInputDeriv(i, newmat);
  			}
  		}
  	}
  }
  
  /**
   * If appropriate, return the current derivative matrix to the backward cache for this net. 
   * Since this is a connected layer, the derivative is aliased with the input derivative and so is not 
   * actually returned to cache. 
   */
  
  def inplaceConnectReleaseDeriv() = {
  	val inplace = Net.getPlacing(opts.inplace, net.opts.inplace);
  	if (inplace == Net.BackwardCaching && doreturn) {
  		for (i <- 0 until outputlength) {
  	  	setDeriv(i, null);
  	  }
  	}
  }
  
  /**
   * If appropriate, return the current derivative matrix to the backward cache for this net. 
   */
  
  def inplaceNoConnectReleaseDeriv() = {
  	val inplace = Net.getPlacing(opts.inplace, net.opts.inplace);
  	if (inplace == Net.BackwardCaching && doreturn) {
  	  for (i <- 0 until outputlength) {
  	  	net.returnMat(derivs(i));
  	  	setDeriv(i, null);
  	  }
  	}
  }

  def getTensorFormat:Int = {
    if (opts.tensorFormat != Net.UseNetFormat) {
      opts.tensorFormat;
    } else {
      net.opts.tensorFormat;
    }
  }
}

class LayerTerm(val _layer:Layer, val term:Int) extends Serializable {
  def layer = _layer;
  
  def +    (a:LayerTerm) = {val n=this; new AddLayer(null){inputs(0)=n; inputs(1)=a}};
  
  def -    (a:LayerTerm) = {val n=this; new SubLayer(null){inputs(0)=n; inputs(1)=a}};

  def *@   (a:LayerTerm) = {val n=this; new MulLayer(null){inputs(0)=n; inputs(1)=a;}};
    
  def ∘    (a:LayerTerm) = {val n=this; new MulLayer(null){inputs(0)=n; inputs(1)=a;}};
    
  def /    (a:LayerTerm) = {val n=this; new DivLayer(null){inputs(0)=n; inputs(1)=a;}};
  
  def dot  (a:LayerTerm) = {val n=this; new DotLayer(null){inputs(0)=n; inputs(1)=a;}};
  
  def ∙    (a:LayerTerm) = {val n=this; new DotLayer(null){inputs(0)=n; inputs(1)=a;}};
  
  def ^    (a:LayerTerm) = {val n=this; new PowerLayer(null){inputs(0)=n; inputs(1)=a;}};
  
  def pow  (a:LayerTerm) = {val n=this; new PowerLayer(null){inputs(0)=n; inputs(1)=a;}};
        
  def over (a:LayerTerm) = {val n=this; new StackLayer(null){inputs(0)=n; inputs(1)=a;}};
  
  def apply(a:LayerTerm) = {val n=this; new SelectLayer(null){inputs(0)=n; inputs(1)=a;}};
}

object Layer {
  
  def abs(a:LayerTerm) = new AbsLayer(null){inputs(0) = a;};
  
  def batchNorm(a:LayerTerm)(avgFactor:Float=0.1f, normMode:Int=BatchNormLayer.SPATIAL) = {
    new BatchNormLayer(null, new BatchNormNode{expAvgFactor=avgFactor; batchNormMode=normMode}){inputs(0)=a;}
  }
  
  def batchNormScale(a:LayerTerm)(name:String="", avgFactor:Float=0.1f, normMode:Int=BatchNormLayer.SPATIAL, hasBias:Boolean = true, 
      lr_scale:Float=1f, bias_scale:Float=1f, inplace:Int = Net.UseNetPlacing, net:Net=null) = {
    val hb = hasBias;
  	val mname = name;
  	val lrs = lr_scale;
  	val bs = bias_scale;
  	val inp = inplace;
    new BatchNormScaleLayer(net, new BatchNormScaleNode{modelName = mname; expAvgFactor=avgFactor; batchNormMode=normMode; 
    hasBias=hb; lr_scale=lrs; bias_scale=bs; inplace=inp}){inputs(0)=a;}
  }
  
  def constant(v:Mat)(net:Net=null):ConstantLayer = {
    new ConstantLayer(net, new ConstantNode{value = v;})
  }
  
  def const(v:Mat):ConstantLayer = {
    new ConstantLayer(null, new ConstantNode{value = v;})
  }
  
  def conv(a:LayerTerm)(name:String="", w:Int, h:Int, nch:Int, stride:IMat = irow(1), pad:IMat = irow(1), hasBias:Boolean = true, 
      initfn:(Mat,Float)=>Mat = Net.xavier, initv:Float = 1f,
      initbiasfn:(Mat,Float)=>Mat = Net.constant, initbiasv:Float = 0f,
      lr_scale:Float=1f, bias_scale:Float=1f,
      convType:Int=cudnnConvolutionMode.CUDNN_CROSS_CORRELATION, net:Net=null) = {
    val str = stride;
    val pd = pad;
    val hb = hasBias;
    val mname = name;
    val initf = initfn;
    val initv0 = initv;
    val initbiasf = initbiasfn;
    val initbiasv0 = initbiasv;
    val lrs = lr_scale;
  	val bs = bias_scale;
    val ct = convType;
    new ConvLayer(net, new ConvNode{modelName = mname; kernel=irow(w,h); noutputs=nch; stride=str; pad=pd; hasBias=hb; 
    initfn = initf; initv = initv0; initbiasfn = initbiasf; initbiasv = initbiasv0; lr_scale=lrs; bias_scale=bs; convType=ct}){inputs(0)=a;};
  }
  
  def copy(a:LayerTerm) = new CopyLayer(null){inputs(0) = a;}

  def copy0 = new CopyLayer(null);
  
  def crop(a:LayerTerm)(sizes:IMat=irow(3,224,224,0), offsets:IMat=irow(0,-1,-1,-1), randoffsets:IMat=null, net:Net=null) = {
    val csizes = sizes;
    val coffsets = offsets;
    val roffsets = randoffsets;
    new CropLayer(net, new CropNode{sizes = csizes; offsets = coffsets; randoffsets = roffsets}){inputs(0) = a;}
  }
  
  def cropMirror(a:LayerTerm)(sizes:IMat=irow(3,224,224,0), offsets:IMat=irow(0,-1,-1,-1), randoffsets:IMat=null, net:Net=null) = {
    val csizes = sizes;
    val coffsets = offsets;
    val roffsets = randoffsets;
    new CropMirrorLayer(net, new CropMirrorNode{sizes = csizes; offsets = coffsets; randoffsets = roffsets}){inputs(0) = a;}
  }
  
  def dropout(a:LayerTerm)(frac:Float=0.5f, net:Net=null) = {
    val dfrac = frac;
    new DropoutLayer(net, new DropoutNode{frac = dfrac}){inputs(0) = a}
  }
  
  def efn(a:LayerTerm)(fwdfn:(Float)=>Float=null, bwdfn:(Float,Float,Float)=>Float=null) = {
    val fwd = fwdfn;
    val bwd = bwdfn;
    new EfnLayer(null, new EfnNode{fwdfn=fwd; bwdfn=bwd}){inputs(0) = a;};
  }
  
  def exp(a:LayerTerm) = new ExpLayer(null){inputs(0) = a;};
  
  def fn(a:LayerTerm)(fwdfn:(Mat)=>Mat=null, bwdfn:(Mat,Mat,Mat)=>Mat=null) = {
    val fwd = fwdfn;
    val bwd = bwdfn;
    new FnLayer(null, new FnNode{fwdfn=fwd; bwdfn=bwd}){inputs(0) = a;};
  }
  
  def fn2(a:LayerTerm, b:LayerTerm)(fwdfn:(Mat,Mat)=>Mat=null, bwdfn1:(Mat,Mat,Mat,Mat)=>Mat=null, bwdfn2:(Mat,Mat,Mat,Mat)=>Mat=null) = {
    val fwd = fwdfn;
    val bwd1 = bwdfn1;
    val bwd2 = bwdfn2;
    new Fn2Layer(null, new Fn2Node{fwdfn=fwd; bwdfn1=bwd1; bwdfn2=bwd2}){inputs(0) = a; inputs(1) = b;};
  }
   
  def format(a:LayerTerm)(conversion:Int = TensorFormatLayer.AUTO, inputFormat:Int = Net.TensorNHWC, net:Net = null) = {
    val con = conversion;
    val fmt = inputFormat;
    new TensorFormatLayer(net, new TensorFormatNode{conversion = con; inputFormat = fmt;}){inputs(0) = a;}
  }
  
  def forward(a:LayerTerm) = new ForwardLayer(null){inputs(0) = a;}
  
  def GLM(a:LayerTerm)(implicit opts:GLMNodeOpts) = new GLMLayer(null, opts){inputs(0) = a};
  
  def input(a:LayerTerm) = new InputLayer(null){inputs(0) = a;};
  
  def input = new InputLayer(null);
  
  def linear(a:LayerTerm)(name:String="", outdim:Int=0, hasBias:Boolean=true, aopts:ADAGrad.Opts=null,
      initfn:(Mat,Float)=>Mat = Net.xavier, initv:Float = 1f,
      initbiasfn:(Mat,Float)=>Mat = Net.constant, initbiasv:Float = 0f,
      lr_scale:Float=1f, bias_scale:Float=1f,
      withInteractions:Boolean=false, tmatShape:(Int,Int)=>(Array[Int], Array[Int], Array[Int], Array[Int]) = null, net:Net = null) = {
    val odim = outdim;
    val hBias = hasBias;
    val aaopts = aopts;
    val mname = name;
    val tms = tmatShape;
    val wi = withInteractions;
    val initf = initfn;
    val initv0 = initv;
    val initbiasf = initbiasfn;
    val initbiasv0 = initbiasv;
    val lrs = lr_scale;
  	val bs = bias_scale;
    new LinLayer(net, new LinNode{modelName = mname; outdim=odim; hasBias=hBias; 
    initfn = initf; initv = initv0; initbiasfn = initbiasf; initbiasv = initbiasv0; 
    lr_scale=lrs; bias_scale=bs;
    aopts=aaopts; withInteractions=wi; tmatShape = tms}){inputs(0)=a;};
  }
  
  def ln(a:LayerTerm) = new LnLayer(null){inputs(0) = a};
  
  def lstm(h:LayerTerm, c:LayerTerm, i:LayerTerm, m:String)(opts:LSTMNodeOpts, net:Net=null) = {
    val node = new LSTMNode;
    opts.copyOpts(node);
    node.modelName = m;
    node.constructGraph;
    val n = LSTMLayer(net, node);
    n.setInput(0, h);
    n.setInput(1, c);
    n.setInput(2, i);
    n
  }
  
  def LRNwithin(h:LayerTerm)(dim:Int=5, alpha:Float=1f, beta:Float=0.5f, net:Net=null) = {
    val node = new LRNwithinNode;
    node.dim = dim;
    node.alpha = alpha;
    node.beta = beta;
    node.constructGraph;
    val layer = LRNwithinLayer(net, node);
    layer.setInput(0, h);
    layer
  }

  def LRNacross(a:LayerTerm)(dim:Int=5, alpha:Float=1f, beta:Float=0.5f, k:Float=2f, net:Net=null) = {
    val node = new LRNacrossNode;
    node.dim = dim;
    node.alpha = alpha;
    node.beta = beta;
    node.k = k;
    new LRNacrossLayer(net, node){inputs(0)=a;}
  }
  
  def max(a:LayerTerm, b:LayerTerm)= {
    new MaxLayer(null){inputs(0) = a; inputs(1) = b;};
  }
  
  def min(a:LayerTerm, b:LayerTerm)= {
    new MinLayer(null){inputs(0) = a; inputs(1) = b;};
  }
  
  def maxi(a:LayerTerm)= {
    new MaxiLayer(null){inputs(0) = a};
  }
  
  def mini(a:LayerTerm)= {
    new MiniLayer(null){inputs(0) = a};
  }
  
  def maxi2(a:LayerTerm)= {
    new Maxi2Layer(null){inputs(0) = a};
  }
  
  def mini2(a:LayerTerm)= {
    new Mini2Layer(null){inputs(0) = a};
  }

  
  def negsamp(a:LayerTerm)(name:String="", outdim:Int=0, hasBias:Boolean=true, aopts:ADAGrad.Opts=null, 
      nsamps:Int=100, expt:Float=0.5f, scoreType:Int=0, doCorrect:Boolean=true, lr_scale:Float=1f, bias_scale:Float=1f, net:Net=null) = {
    val odim = outdim;
    val hBias = hasBias;
    val aaopts = aopts;
    val nnsamps = nsamps;
    val eexpt = expt;
    val dcr = doCorrect;
    val sct = scoreType;
    val mname = name;
    val lrs = lr_scale;
  	val bs = bias_scale;
    new NegsampOutputLayer(net, new NegsampOutputNode{modelName=mname; outdim=odim; hasBias=hBias; aopts=aaopts; 
    lr_scale=lrs; bias_scale=bs; nsamps=nnsamps; expt=eexpt; scoreType=sct; docorrect=dcr}){inputs(0)=a;};
  }
  
  def norm(a:LayerTerm)(opts:NormNodeOpts) = new NormLayer(null){inputs(0) = a;}
  
  def oneHot(a:LayerTerm) = new OnehotLayer(null){inputs(0) = a};
  
  def pool(a:LayerTerm)(h:Int=1, w:Int=1, stride:Int=1, pad:Int=0, 
      poolingMode:Int=cudnnPoolingMode.CUDNN_POOLING_MAX, 
      poolingNaN:Int=cudnnNanPropagation.CUDNN_PROPAGATE_NAN,
      tensorFormat:Int = Net.UseNetFormat, net:Net=null) = {
  	val hh = h;
  	val ww = w;
  	val str = stride;
  	val ppad = pad;
  	val pm = poolingMode;
  	val pn = poolingNaN;
  	val tf = tensorFormat;
    new PoolingLayer(net, new PoolingNode{h=hh; w=ww; stride=str; pad=ppad; poolingMode=pm; poolingNaN=pn; tensorFormat=tf;}){inputs(0)=a;}  
  }
  
  def scale(a:LayerTerm)(name:String="", normMode:Int=BatchNormLayer.SPATIAL, hasBias:Boolean = true, lr_scale:Float=1f, bias_scale:Float=1f, net:Net=null) = {
  	val hb = hasBias;
  	val mname = name;
  	val lrs = lr_scale;
  	val bs = bias_scale;
    new ScaleLayer(net, new ScaleNode{modelName = mname; batchNormMode=normMode; hasBias=hb; lr_scale=lrs; bias_scale=bs;}){inputs(0)=a;}   
  }
    
  def randmirror(a:LayerTerm)(prob:Float=0.5f, net:Net=null) = {
    val p = prob;
    new RandomMirrorLayer(net, new RandomMirrorNode{prob=p;}){inputs(0) = a};
  } 
  
  def rect(a:LayerTerm)(inplace:Int=Net.UseNetPlacing) = {
    val inplac = inplace;
    new RectLayer(null, new RectNode{inplace=inplac;}){inputs(0) = a};
  } 
  
  def relu(a:LayerTerm)(inplace:Int=Net.UseNetPlacing) = {
    val inplac = inplace;
    new RectLayer(null, new RectNode{inplace=inplac;}){inputs(0) = a};
  }
  
  def sigmoid(a:LayerTerm) = new SigmoidLayer(null){inputs(0) = a};
  
  def sign(a:LayerTerm) = new SignLayer(null){inputs(0) = a;};
  
  def σ(a:LayerTerm) = new SigmoidLayer(null){inputs(0) = a};

  def softmax(a:LayerTerm) = new SoftmaxLayer(null){inputs(0) = a};
  
  def softmaxout(a:LayerTerm)(scoreType:Int=0, lossType:Int = 0, net:Net=null) =  {
    val scoreTyp = scoreType;
    val lossTyp = lossType;
    new SoftmaxOutputLayer(net, new SoftmaxOutputNode{scoreType=scoreTyp; lossType=lossTyp}){inputs(0) = a}
  }
  
  def softplus(a:LayerTerm) = new SoftplusLayer(null){inputs(0) = a};
  
  def splithoriz(a:LayerTerm)(np:Int) = new SplitHorizLayer(null, new SplitHorizNode{nparts = np}){inputs(0) = a};
  
  def splitvert(a:LayerTerm)(np:Int, net:Net=null) = {
    new SplitVertLayer(net, new SplitVertNode{nparts = np}){inputs(0) = a};
  }
  
  def sqrt(a:LayerTerm) = new SqrtLayer(null){inputs(0) = a;};
  
  def sum(a:LayerTerm) = new SumLayer(null){inputs(0) = a};
  
  def tanh(a:LayerTerm) = new TanhLayer(null){inputs(0) = a};
  
}


trait OutputLayer {}

object LayerFn {
  final val SIGMOIDFN = 0;
  final val TANHFN = 1;
  final val SOFTPLUSFN = 2;
  
  val fwdflops = irow(20, 20, 40);
  val bwdflops = irow(3, 3, 20);
  
  // Loosely check dimensions. Skip dimensions of 1 in either tensor.
  def checkdims(dims0:IMat, dims1:IMat) = {
    if (dims1.asInstanceOf[AnyRef] != null) {
      var i0 = 0;
      var i1 = 0;
      while (i0 < dims0.length && i1 < dims1.length) {
        while (i0 < dims0.length && dims0(i0) == 1) i0 += 1;
        while (i1 < dims1.length && dims1(i1) == 1) i1 += 1; 
        if ((i0 >= dims0.length) != (i1 >= dims1.length)) {
          throw new RuntimeException("dimensions mismatch in Layer Function " + dims0.toString + " and " + dims1.toString);
        } else if (i0 < dims0.length && i1 < dims1.length && dims0(i0) != dims1(i1)) {
        	throw new RuntimeException("dimensions mismatch in Layer Function " + dims0.toString + " and " + dims1.toString);           
        }
        i0 += 1;
        i1 += 1;
      }
    }
  }
  
  def applyfwd(a:Mat, ifn:Int):Mat = applyfwd(a, null, ifn);
  
  def applyfwd(a:Mat, out:Mat, ifn:Int):Mat = {
    Mat.nflops += 1L * a.length * fwdflops(ifn);
    checkdims(a.dims, out.dims);
    a match {
      case ag:GMat => {
        val oMat = GMat.newOrCheckGMat(a.dims, out, a.GUID, ifn, "LayerFn".##);
        CUMACH.applyfwd(ag.pdata, oMat.pdata, ifn, a.length);
        oMat
      }
      case af:FMat => {
        val oMat = FMat.newOrCheckFMat(a.dims, out, a.GUID, ifn, "LayerFn".##);
        CPUMACH.applyfwd(af.data, oMat.data, ifn, a.length, Mat.numThreads);
        oMat
      }
    }
  }

  def applyderiv(a:Mat, b:Mat, ifn:Int):Mat = applyderiv(a, b, null, ifn)
      
  def applyderiv(a:Mat, b:Mat, out:Mat, ifn:Int):Mat = {
	  Mat.nflops += 1L * a.length * bwdflops(ifn);
	  checkdims(a.dims, b.dims);
    (a, b) match {
      case (ag:GMat, bg:GMat) => {
        val oMat = GMat.newOrCheckGMat(a.dims, out, a.GUID, ifn, "LayerFn".##);
        CUMACH.applyderiv(ag.pdata, bg.pdata, oMat.pdata, ifn, a.length);
        oMat
      }
      case (af:FMat, bf:FMat) => {
        val oMat = FMat.newOrCheckFMat(a.dims, out, a.GUID, ifn, "LayerFn".##);
        CPUMACH.applyderiv(af.data, bf.data, oMat.data, ifn, a.length, Mat.numThreads);
        oMat
      }
    }
  }
}
 


