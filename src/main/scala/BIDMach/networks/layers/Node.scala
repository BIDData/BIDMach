package BIDMach.networks.layers

/**
 * Node class. Nodes are elements of the computation graph of a Neural Network. 
 * Nodes contain no storage. They are designed to represent the networks topology and options.
 * Nodes differ from Layers which are fully-functional network elements. Layers contain Node's options and also storage for activations.  
 * The computation graph is normally used to initialize a Net instance, which instantiates the Layers corresponding to each node. 
 * 
 * Here are the current Node types:
 * 
 - AbsNode: Absolute value of input. 
 - AddNode: Sum two inputs element-wise.
 - BatchNormNode: Batch normalization.
 - BatchNormScaleNode: Batch normalization with scaling and bias.
 - CompoundNode: A Node wrapping a sub-network, such as an LSTM.
 - ConstantNode: Wraps a constant or externally-set matrix (can be updated between forward steps). 
 - ConvNode: Convolution Node.
 - CopyNode: Copies inputs forward and derivatives backward.
 - CropNode: Crops its input. 
 - CropMirror: Crop and (randomly) mirror the input. 
 - DivNode: Computes the quotient of its two inputs. 
 - DotNode: Dot product (element-wise product followed by sum over columns).
 - DropoutNode: A Node that implements random dropout. No learnable params, but dropout fraction can be specified.  
 - EfnNode: Node that applies a float-valued function to each pixel of its input. 
 - ElasticNode: Node that implements an elastic update. 
 - ExpNode: exponential function of input.
 - FnNode: Applies its fwdfn argument to one input. Can also use bwdfn1 and bwdfn2 for backward data and model updates. 
 - Fn2Node: Applies its fwdfn argument to two inputs. Can also use bwdfn1 and bwdfn2 for backward data and model updates. 
 - ForwardNode: Goes forward only.
 - GLMNode: a one-to-one Node with GLM mappings (linear, logistic, abs-logistic and SVM). No learnable params. 
 - InputNode: just a placeholder for the first Node which is loaded with input output blocks. No learnable params. 
 - LinNode: Linear Node. Has a matrix of learnable params which is the input-output map. 
 - LnNode: natural logarithm.
 - LSTMNode: an LSTM Node.
 - MaxNode: computes the element-wise max of two inputs;
 - MaxiNode: computes the max element in each column;
 - Maxi2Node: computes the max element in each column as output(0), output(1) is a row of integer indices of the maxs.
 - MinNode: computes the element-wise min of two inputs;
 - MiniNode: computes the min element in each column;
 - Mini2Node: computes the min element in each column as output(0), output(1) is a row of integer indices of the mins.
 - MulNode: computes the product of its inputs. 
 - NegsampOutputNode: a negative sampling Output Node. 
 - NormNode: normalizing Node that adds a derivative term based on the difference between current Node norm and a target norm. 
 - MulNode: multiplies input Nodes element-wise. Will also perform edge operations (one input can be a scalar).
 - OnehotNode: Converts an integer array of feature values to a sparse matrix whose columns are the instances with one non-zero in the feature position.
 - PoolingNode: Pooling Node.
 - PowerNode: Raise its input to a power.
 - RandomMirrorNode: Randomly mirror the input horizontally. 
 - RectNode: Rectifying one-to-one Node. No params. 
 - ScaleNode: Scale-Bias Node, usually used with BatchNorm. 
 - SelectNode: Use second (integer) matrix input to select elements from columns of first input. 
 - SigmoidNode: Logistic function non-linearity.
 - SignNode: Compute the sign of the input.
 - SoftmaxNode: a softmax (normalized exponential) Node.
 - SoftmaxOutputNode: a softmax (normalized exponential) output Node.
 - SoftplusNode: smooth ReLU unit. 
 - SplitHorizNode: Split the input into nparts horizontally.
 - SplitVertNode: Split the input into nparts vertically. 
 - SqrtNode: Square root Node. 
 - StackNode: Vertically stack its inputs. 
 - SubNode: Subtract inputs > 1 from first input. 
 - SumNode: column-wise sum.
 - TanhNode: Hyperbolic tangent non-linearity.
 - TensorFormatNode: Corrects Tensor Format (NCHW or NHWC)
 * 
 * Shorthand operators and functions for Node creation have been defined in Node.scala and Layer.scala. 
 * 
 * The syntax for most layers is nodename(inputs)(optional args)
 *  
 * The shortcuts are as follows (x and y are input NodeTerms or LayerTerms with float data, i is an input node with integer data):
 * 
 - AbsNode:                     abs(y) 
 - AddNode:                     x + y
 - BatchNormNode:               batchnorm(x)(avgFactor:Float=0.1f, normMode:Int=BatchNormNode.SPATIAL)
 - BatchNormScaleNode:          bns(x)(name:String="", avgFactor:Float=0.1f, normMode:Int=BatchNormNode.SPATIAL, hasBias:Boolean=true,
                                        lr_scale:Float=1f, bias_scale:Float=1f, inplace:Int = Net.UseNetPlacing)
 - ConstantNode:                const(1e-3f)
 - ConvNode:                    conv(x)(name:String="", w:Int, h:Int, nch:Int, stride:IMat = irow(1), pad:IMat = irow(1), hasBias:Boolean = true, 
                                         initfn:(Mat,Float)=>Mat = Net.xavier, initv:Float = 1f,
                                         initbiasfn:(Mat,Float)=>Mat = Net.constant, initbiasv:Float = 0f,
                                         lr_scale:Float=1f, bias_scale:Float=1f,
                                         convType:Int=cudnnConvolutionMode.CUDNN_CROSS_CORRELATION)
 - CopyNode:                    copy(x)
 - CropNode:                    crop(x)(sizes:IMat=irow(3,224,224,0), offsets:IMat=irow(0,-1,-1,-1), randoffsets:IMat=null)
 - CropMirror:                   cropmirror(x)(sizes:IMat=irow(3,224,224,0), offsets:IMat=irow(0,-1,-1,-1), randoffsets:IMat=null)
 - DivNode:                     x / y 
 - DotNode:                     x dot y     or     x ∙ y
 - DropoutNode:                 dropout(x)(frac:Float=0.5f)
 - EfnNode:                     efn(x)(fwdfn:(Float)=>Float=null, bwdfn:(Float,Float,Float)=>Float=null)
 - ExpNode:                     exp(x)
 - FnNode:                      fn(x)(fwdfn:(Mat)=>Mat=null, bwdfn:(Mat,Mat,Mat)=>Mat=null)
 - Fn2Node:                     fn2(x)(fwdfn:(Mat,Mat)=>Mat=null, bwdfn1:(Mat,Mat,Mat,Mat)=>Mat=null, bwdfn2:(Mat,Mat,Mat,Mat)=>Mat=null) 
 - ForwardNode:                 forward(x)
 - GLMNode:                     glm(x)(links:IMat)
 - InputNode:                   input()
 - LinNode:                     linear(x)(name:String="", outdim:Int=0, hasBias:Boolean=true, aopts:ADAGrad.Opts=null, withInteractions:Boolean=false, 
                                           initfn:(Mat,Float)=>Mat = Net.xavier, initv:Float = 1f,
                                           initbiasfn:(Mat,Float)=>Mat = Net.constant, initbiasv:Float = 0f,
                                           lr_scale:Float=1f, bias_scale:Float=1f,
                                           tmatShape:(Int,Int)=>(Array[Int], Array[Int], Array[Int], Array[Int]) = null)
 - LnNode:                      ln(x)
 - LRNacrossNode:               LRNacross(x)(dim:Int=5, alpha:Float=1f, beta:Float=0.5f, k:Float=2f)
 - LRNwithinNode:               LRNwithin(x)(dim:Int=5, alpha:Float=1f, beta:Float=0.5f)
 - LSTMNode:                    lstm(x,y,z,name:String)(opts:LSTMNodeOpts)
 - MaxNode:                     max(x,y)
 - MaxiNode:                    maxi(x)
 - Maxi2Node:                   maxi2(x)
 - MinNode:                     min(x,y)
 - MiniNode:                    mini(x)
 - Mini2Node:                   mini2(x)
 - MulNode:                     x *@ y 
 - NegsampOutputNode:           negsamp(x)(name:String="", outdim:Int=0, hasBias:Boolean=true, aopts:ADAGrad.Opts=null, nsamps:Int=100, 
                                            expt:Float=0.5f, lr_scale:Float=1f, bias_scale:Float=1f, scoreType:Int=0, doCorrect:Boolean=true)
 - NormNode:                    norm(x)(targetNorm:Float = 1f, weight:Float = 1f)
 - OnehotNode:                  onehot(x)
 - PoolNode:                    pool(x)(h:Int=1, w:Int=1, stride:Int=1, pad:Int=0, 
                                         poolingMode:Int=cudnnPoolingMode.CUDNN_POOLING_MAX, 
                                         poolingNaN:Int = cudnnNanPropagation.CUDNN_PROPAGATE_NAN,
                                         tensorFormat:Int = Net.UseNetFormat)
 - RandMirror:                   randmirror(x)(prob:Float=0.5f)
 - RectNode:                    rect(x)(inplace:Int=Net.UseNetPlacing)    or    relu(x)(inplace:Int=Net.UseNetPlacing) 
 - ScaleNode:                   scale(x)(name:String="", normMode:Int=BatchNormNode.SPATIAL, hasBias:Boolean = true,
                                          lr_scale:Float=1f, bias_scale:Float=1f)
 - SelectNode:                  x(i) 
 - SignNode:                    sign(x)
 - SigmoidNode:                 sigmoid(x)
 - SoftmaxNode:                 softmax(x)
 - SoftmaxOutputNode:           softmaxout(x)(scoreType:Int=0, lossType:Int=0)
 - SoftplusNode:                softplus(x) 
 - SplitHorizNode:              splithoriz(x)(nparts:Int)
 - SplitVertNode:               splitvert(x)(nparts:Int)
 - SqrtNode:                    sqrt(x)
 - StackNode:                   x on y
 - SubNode:                     x - y 
 - SumNode:                     sum(x)
 - TanhNode:                    tanh(x)
 - TensorFormatNode:            format(x)(conversion:Int = TensorFormatLayer.AUTO, inputFormat:Int = Net.TensorNHWC)
 *
 * General arguments are as follows:
 * 
 - aopts:ADAGrad.Opts=null       ADAGRAD options instance. Used for just-in-time model updates without an updater. 
 - avgFactor:Float               Specifies the moving average factor for Batch Norm.
 - bias_scale:Float              Scale the global learning rate by this factor when updating the bias for this Node. 
 - hasBias:Boolean               Specifies whether this (model) Node has a bias term or not. 
 - initfn:(Mat,Float)=>Mat       Initializer for the model matrix for this Node. Takes the model matrix as input, and an init value.
                                 Values: Net.xavier, Net.constant
 - initbiasfn:(Mat,Float)=>Mat   Initializer for the bias matrix for this Node. Takes the bias matrix as input, and an init value.
                                 Values: Net.xavier, Net.constant
 - initv:Float                   The initialization value (a scale factor) for the model matrix, which is passed to the initfn.
 - initbiasv:Float               The initialization value (a scale factor) for the bias matrix, which is passed to the initbiasfn. 
 - inplace:Int                   Specifies the memory-sharing model for this node.
 - lr_scale:Float                Scale the global learning rate by this factor when updating the model for this Node. 
 - name:String                   A name for this node. Model nodes with the same name share parameters. 
 - normMode:Int                  The kind of Batch Normalization: BatchNormNode.SPATIAL or BatchNormNode.PER_ACTIVATION
 - nparts:Int                    Number of parts to split this Node into, for splitvert and splithoriz. 
 - tensorFormat:Int              The tensor format. Net.tensorNCHW or Net.tensorNHWC or Net.useNetFormat
 - 
 * 
 * Arguments for particular Nodes:
 * ConvNode
 - w:Int                         Width of the kernel
 - h:Int                         Height of the kernel
 - nch:Int                       Number of filters (output dimension)
 - stride:IMat = irow(1)         The stride, can be a 1x1 matrix.
 - pad:IMat = irow(1)            The padding
 - convType:Int                  Convolution type: Net.CrossCorrelation or Net.Convolution or Net.useNetConvType
 *
 * PoolNode
 - w:Int                         Width of the kernel
 - h:Int                         Height of the kernel
 - stride:Int                    The stride
 - pad:Int                       The padding
 - poolingMode:Int               The pooling mode: cudnnPoolingMode.CUDNN_POOLING_MAX or CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
 - poolingNaN:Int                Whether to propagate NaNs, cudnnNanPropagation.CUDNN_PROPAGATE_NAN
 *
 * Crop and CropMirror Nodes:
 - sizes:IMat=irow(3,224,224,0)  Size of the cropped image
 - offsets:IMat=irow(0,-1,-1,-1) Offsets from the input image boundaries. -1 means center the cropped window in this dimension.
 - randoffsets:IMat=null         Magnitude of random offsets in each dimension. 
 * 
 * LinNode
 - withInteractions:Boolean=false model interactions between features.
 - tmatShape:(Int,Int)=>(Array[Int], Array[Int], Array[Int], Array[Int]) = null) Shape for a TMat (Tiled matrix) for this Node's model. 
 - 
 * LRNNodes:
 - dim:Int=5
 - alpha:Float=1f 
 - beta:Float=0.5f
 - k:Float=2f
 *
 * SoftmaxOutput Node
 - lossType:Int                  What kind of loss to optimize:
                                   SoftmaxOutputLayer.CrossEntropyLoss
                                   SoftmaxOutputLayer.CaffeMultinomialLogisticLoss
                                   SoftmaxOutputLayer.MultinomialLoss
                                   
 - scoreType:Int                 What score to report:
                                   SoftmaxOutputLayer.CrossEntropyScore
                                   SoftmaxOutputLayer.AccuracyScore
 *
 * Many of these functions need a parent Net instance, but you can create that automatically using:
 * 
 *   import BIDMach.networks.layers.Layer._;
 *   Net.initDefault(opts);
 *
 * then define some nodes using the functions above:
 * 
 *     val input1 = input();
 *     val input2 = input();
 *     val const1 = constant(5f);
 *     val s2 =     (input1 + input2) *@ const1;
 *     val s3 =     exp(s2)
 *     ...          ...
 * 
 * Finally, to retrieve the computation graph containing these nodes, do:
 * 
 *   val nodes = Net.getDefaultNet;
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

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach.networks.layers._
import BIDMach._
import edu.berkeley.bid.CPUMACH
import edu.berkeley.bid.CUMACH
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import scala.util.hashing.MurmurHash3;
import scala.language.implicitConversions
import java.util.HashMap;
import BIDMach.networks._


@SerialVersionUID(100L)
trait NodeOpts extends BIDMat.Opts {

  var name = "";  
  var inplace:Int = Net.UseNetPlacing;
  var tensorFormat:Int = Net.UseNetFormat;
  
  def copyOpts(opts:NodeOpts):NodeOpts = {
    opts.name = name;
    opts.tensorFormat = tensorFormat
    opts.inplace = inplace;
		opts;
  }
}

class Node extends NodeTerm(null, 0) with NodeOpts {
	val inputs:Array[NodeTerm] = Array(null);
  var myLayer:Layer = null;
  var myGhost:Node = null;
  var parent:Node = null;
  var outputNumbers:Array[Int] = null;
  
  override def node = this;
  
  Net.addNode(this);
  
  def copyTo(opts:Node):Node = {
    copyOpts(opts);
    opts.inputs(0) = inputs(0);
    myGhost = opts;
    opts;
  }
  
  override def toString = {
    "node@"+(hashCode % 0x10000).toString
  }

  override def clone:Node = {
		copyTo(new Node).asInstanceOf[Node];
  } 
  
  def apply(i:Int) = new NodeTerm(this, i);
  
  def create(net:Net):Layer = {null}
}

trait OutputNode {}


class NodeTerm(val _node:Node, val term:Int) extends Serializable {  
  
  def node = _node;
  
  def +    (a:NodeTerm) = {val n=this; new AddNode{inputs(0)=n; inputs(1)=a}};
  
  def -    (a:NodeTerm) = {val n=this; new SubNode{inputs(0)=n; inputs(1)=a}};

  def *@   (a:NodeTerm) = {val n=this; new MulNode{inputs(0)=n; inputs(1)=a;}};
    
  def ∘    (a:NodeTerm) = {val n=this; new MulNode{inputs(0)=n; inputs(1)=a;}};
  
  def /    (a:NodeTerm) = {val n=this; new DivNode{inputs(0)=n; inputs(1)=a;}};
  
  def dot  (a:NodeTerm) = {val n=this; new DotNode{inputs(0)=n; inputs(1)=a;}};
  
  def ∙    (a:NodeTerm) = {val n=this; new DotNode{inputs(0)=n; inputs(1)=a;}};
  
  def ^    (a:NodeTerm) = {val n=this; new PowerNode{inputs(0)=n; inputs(1)=a;}};
  
  def pow  (a:NodeTerm) = {val n=this; new PowerNode{inputs(0)=n; inputs(1)=a;}};
        
  def over (a:NodeTerm) = {val n=this; new StackNode{inputs(0)=n; inputs(1)=a;}};
  
  def apply(a:NodeTerm) = {val n=this; new SelectNode{inputs(0)=n; inputs(1)=a;}};
}

object Node {
  
	def abs(a:NodeTerm) = new AbsNode{inputs(0) = a;};
	
	def add(a:Array[NodeTerm]) = new AddNode{ninputs = a.length; Array.copy(a, 0, inputs, 0, a.length);};
  
  def batchNorm(a:NodeTerm)(avgFactor:Float=0.1f, normMode:Int=BatchNormLayer.SPATIAL) = {
    new BatchNormNode{inputs(0)=a; expAvgFactor=avgFactor; batchNormMode=normMode}    
  }
  
  def batchNormScale(a:NodeTerm)(name:String="", avgFactor:Float=0.1f, normMode:Int=BatchNormLayer.SPATIAL, hasBias:Boolean = true,
      lr_scale:Float=1f, bias_scale:Float=1f, inplace:Int = Net.UseNetPlacing) = {
  	val hb = hasBias;
  	val mname = name;
  	val lrs = lr_scale;
  	val bs = bias_scale;
  	val inp = inplace;
    new BatchNormScaleNode{inputs(0)=a; modelName=mname; expAvgFactor=avgFactor; batchNormMode=normMode; hasBias=hb; lr_scale=lrs; bias_scale=bs; inplace=inp}    
  }
    
  def constant(v:Mat) = {
    new ConstantNode{value = v;}
  }
  
  def const(v:Mat) = {
    new ConstantNode{value = v;}
  }
    
  def conv(a:NodeTerm)(name:String="", w:Int, h:Int, nch:Int, stride:IMat = irow(1), pad:IMat = irow(1), hasBias:Boolean = true, 
      initfn:(Mat,Float)=>Mat = Net.xavier, initv:Float = 1f,
      initbiasfn:(Mat,Float)=>Mat = Net.constant, initbiasv:Float = 0f,
      lr_scale:Float=1f, bias_scale:Float=1f,
      convType:Int=cudnnConvolutionMode.CUDNN_CROSS_CORRELATION) = {
    val str = stride;
    val pd = pad;
    val hb = hasBias;
    val initf = initfn;
    val initv0 = initv;
    val initbiasf = initbiasfn;
    val initbiasv0 = initbiasv;
    val lrs = lr_scale;
  	val bs = bias_scale;
    val mname = name;
    val ct = convType;
    new ConvNode{inputs(0)=a; modelName=mname; kernel=irow(w,h); noutputs=nch; stride=str; pad=pd; hasBias=hb; 
    initfn = initf; initv = initv0; initbiasfn = initbiasf; initbiasv = initbiasv0; convType=ct; lr_scale=lrs; bias_scale=bs;}
  }
  
  def copy(a:NodeTerm) = new CopyNode{inputs(0) = a;}

  def copy() = new CopyNode;
  
  def crop(a:NodeTerm)(sizes:IMat=irow(3,224,224,0), offsets:IMat=irow(0,-1,-1,-1), randoffsets:IMat=null) = {
    val csizes = sizes;
    val coffsets = offsets;
    val roffsets = randoffsets;
    new CropNode{inputs(0) = a; sizes = csizes; offsets = coffsets; randoffsets = roffsets};
  }
  
  def cropMirror(a:NodeTerm)(sizes:IMat=irow(3,224,224,0), offsets:IMat=irow(0,-1,-1,-1), randoffsets:IMat=null) = {
    val csizes = sizes;
    val coffsets = offsets;
    val roffsets = randoffsets;
    new CropMirrorNode{inputs(0) = a; sizes = csizes; offsets = coffsets; randoffsets = roffsets};
  }
  
  def dropout(a:NodeTerm)(frac:Float=0.5f) = {
    val dfrac = frac;
    new DropoutNode{inputs(0) = a; frac = dfrac}
  }
  
  def efn(a:NodeTerm)(fwdfn:(Float)=>Float=null, bwdfn:(Float,Float,Float)=>Float=null) = {
    val fwd = fwdfn;
    val bwd = bwdfn;
    new EfnNode{inputs(0) = a; fwdfn=fwd; bwdfn=bwd};
  }
  
  def exp(a:NodeTerm) = new ExpNode{inputs(0) = a;};
  
  def format(a:NodeTerm)(conversion:Int = TensorFormatLayer.AUTO, inputFormat:Int = Net.TensorNHWC) = {
    val con = conversion;
    val fmt = inputFormat;
    new TensorFormatNode{inputs(0) = a; conversion = con; inputFormat = fmt;}
  }
  
  def forward(a:NodeTerm) = new ForwardNode{inputs(0) = a;}
  
  def fn(a:NodeTerm)(fwdfn:(Mat)=>Mat=null, bwdfn:(Mat,Mat,Mat)=>Mat=null) = {
    val fwd = fwdfn;
    val bwd = bwdfn;
    new FnNode{inputs(0) = a; fwdfn=fwd; bwdfn=bwd};
  }
  
  def fn2(a:NodeTerm,b:NodeTerm)(fwdfn:(Mat,Mat)=>Mat=null, bwdfn1:(Mat,Mat,Mat,Mat)=>Mat=null, bwdfn2:(Mat,Mat,Mat,Mat)=>Mat=null) = {
    val fwd = fwdfn;
    val bwd1 = bwdfn1;
    val bwd2 = bwdfn2;
    new Fn2Node{inputs(0) = a; inputs(1) = b; fwdfn=fwd; bwdfn1=bwd1; bwdfn2=bwd2};
  }
  
  def glm_(a:NodeTerm)(implicit opts:GLMNodeOpts) = new GLMNode{inputs(0) = a; links = opts.links};
  
  def glm(a:NodeTerm)(links:IMat) = {
    val ilinks = links; 
    new GLMNode{inputs(0) = a; links = ilinks}
    };
  
  def input(a:NodeTerm) = new InputNode{inputs(0) = a;};
  
  def input() = new InputNode;
  
  def linear(a:NodeTerm)(name:String="", outdim:Int=0, hasBias:Boolean=true, aopts:ADAGrad.Opts=null, withInteractions:Boolean=false, 
      initfn:(Mat,Float)=>Mat = Net.xavier, initv:Float = 1f,
      initbiasfn:(Mat,Float)=>Mat = Net.constant, initbiasv:Float = 0f,
      lr_scale:Float=1f, bias_scale:Float=1f,
      tmatShape:(Int,Int)=>(Array[Int], Array[Int], Array[Int], Array[Int]) = null) = {
    val odim = outdim;
    val hBias = hasBias;
    val aaopts = aopts;
    val mname = name;
    val wi = withInteractions;
    val tms = tmatShape; 
    val initf = initfn;
    val initv0 = initv;
    val initbiasf = initbiasfn;
    val initbiasv0 = initbiasv;
    val lrs = lr_scale;
  	val bs = bias_scale;
    new LinNode{inputs(0)=a; modelName = mname; outdim=odim; hasBias=hBias; 
    initfn = initf; initv = initv0; initbiasfn = initbiasf; initbiasv = initbiasv0; lr_scale=lrs; bias_scale=bs;
    aopts=aaopts; withInteractions = wi; tmatShape = tms};
  }
  
  def linear_(a:NodeTerm)(implicit opts:LinNodeOpts) = {
    val n = new LinNode{inputs(0) = a;}
    opts.copyOpts(n);
    n
  }
    
  def ln(a:NodeTerm) = new LnNode{inputs(0) = a};
  
  def LRNwithin(h:NodeTerm)(dim:Int=5, alpha:Float=1f, beta:Float=0.5f) = {
    val n = new LRNwithinNode;
    n.dim = dim;
    n.alpha = alpha;
    n.beta = beta;
    n.constructGraph;
    n.inputs(0) = h;
    n
  }
  
  def LRNacross(a:NodeTerm)(dim:Int=5, alpha:Float=1f, beta:Float=0.5f, k:Float=2f) = {
    val n = new LRNacrossNode{inputs(0) = a};
    n.dim = dim;
    n.alpha = alpha;
    n.beta = beta;
    n.k = k;
    n
  }
  
  def lstm(h:NodeTerm, c:NodeTerm, i:NodeTerm, m:String)(opts:LSTMNodeOpts) = {
    val n = new LSTMNode;
    opts.copyOpts(n);
    n.modelName = m;
    n.constructGraph;
    n.inputs(0) = h;
    n.inputs(1) = c;
    n.inputs(2) = i;
    n
  }
  
  def lstm_fused(inc:NodeTerm, lin1:NodeTerm, lin2:NodeTerm, lin3:NodeTerm, lin4:NodeTerm) = {
    new LSTMfusedNode{
      inputs(0) = inc;
      inputs(1) = lin1;
      inputs(2) = lin2;
      inputs(3) = lin3;
      inputs(4) = lin4;
    }
  }
  
  def max(a:NodeTerm,b:NodeTerm) = {
    new MaxNode{inputs(0) = a; inputs(1) = b;};
  }
  
  def min(a:NodeTerm,b:NodeTerm) = {
    new MinNode{inputs(0) = a; inputs(1) = b;};
  }
  
  def maxi(a:NodeTerm) = {
    new MaxiNode{inputs(0) = a};
  }
  
  def mini(a:NodeTerm) = {
    new MiniNode{inputs(0) = a};
  }
  
  def negsamp(a:NodeTerm)(name:String="", outdim:Int=0, hasBias:Boolean=true, aopts:ADAGrad.Opts=null, nsamps:Int=100, expt:Float=0.5f, 
      lr_scale:Float=1f, bias_scale:Float=1f, scoreType:Int=0, doCorrect:Boolean=true) = {
    val odim = outdim;
    val hBias = hasBias;
    val aaopts = aopts;
    val nnsamps = nsamps;
    val eexpt = expt;
    val dcr = doCorrect;
    val sct = scoreType;
    val lrs = lr_scale;
  	val bs = bias_scale;
    val mname = name;
    new NegsampOutputNode{inputs(0)=a; modelName=mname; outdim=odim; hasBias=hBias; aopts=aaopts; nsamps=nnsamps; expt=eexpt; 
    scoreType=sct; docorrect=dcr; lr_scale=lrs; bias_scale=bs;};
  }
    
  def negsamp_(a:NodeTerm)(implicit opts:NegsampOutputNodeOpts) =     {
    val n = new NegsampOutputNode{inputs(0) = a}
    opts.copyOpts(n);
    n
  }
  
  def norm(a:NodeTerm)(targetNorm:Float = 1f, weight:Float = 1f) =  {
    val tnorm = targetNorm;
    val nweight = weight;
    new NormNode{inputs(0) = a; targetNorm = tnorm; weight = nweight}
  }
  
  def norm_(a:NodeTerm)(implicit opts:NormNodeOpts) =  {
    val n = new NormNode{inputs(0) = a;}
    opts.copyOpts(n);
    n
  }
  
  def oneHot(a:NodeTerm) = new OnehotNode{inputs(0) = a};
  
  def pool(a:NodeTerm)(h:Int=1, w:Int=1, stride:Int=1, pad:Int=0, 
      poolingMode:Int=cudnnPoolingMode.CUDNN_POOLING_MAX, 
      poolingNaN:Int = cudnnNanPropagation.CUDNN_PROPAGATE_NAN,
      tensorFormat:Int = Net.UseNetFormat) = {
  	val hh = h;
  	val ww = w;
  	val str = stride;
  	val ppad = pad;
  	val pm = poolingMode;
  	val pn = poolingNaN;
  	val tf = tensorFormat;
    new PoolingNode{inputs(0)=a; h=hh; w=ww; stride=str; pad=ppad; poolingMode=pm; poolingNaN=pn; tensorFormat=tf;}; 
  }
  
  def randmirror(a:NodeTerm)(prob:Float=0.5f) = {
    val p = prob;
    new RandomMirrorNode{inputs(0) = a; prob = p};
  }
   
  def rect(a:NodeTerm)(inplace:Int=Net.UseNetPlacing) = {
    val inplac = inplace;
    new RectNode{inputs(0) = a; inplace = inplac};
  }
  
  def relu(a:NodeTerm)(inplace:Int=Net.UseNetPlacing) = {
  	val inplac = inplace;
    new RectNode{inputs(0) = a; inplace = inplac};
  }
  
  def scale(a:NodeTerm)(name:String="", normMode:Int=BatchNormLayer.SPATIAL, hasBias:Boolean = true,
      lr_scale:Float=1f, bias_scale:Float=1f) = {
  	val hb = hasBias;
  	val mname = name;
  	val lrs = lr_scale;
  	val bs = bias_scale;
    new ScaleNode{inputs(0)=a; modelName=mname; batchNormMode=normMode; hasBias=hb; lr_scale=lrs; bias_scale=bs;}    
  }
  
  def sigmoid(a:NodeTerm) = new SigmoidNode{inputs(0) = a};
  
  def sign(a:NodeTerm) = new SignNode{inputs(0) = a;};
  
  def σ(a:NodeTerm) = new SigmoidNode{inputs(0) = a};

  def softmax(a:NodeTerm) = new SoftmaxNode{inputs(0) = a};
  
  def softmaxout(a:NodeTerm)(scoreType:Int=0, lossType:Int=0) =  {
    val scoreTyp = scoreType;
    val lossTyp = lossType;
    new SoftmaxOutputNode{inputs(0) = a; scoreType=scoreTyp; lossType = lossTyp}
  }
  
  def softplus(a:NodeTerm) = new SoftplusNode{inputs(0) = a};
  
  def splithoriz(a:NodeTerm)(np:Int) = new SplitHorizNode{inputs(0) = a; nparts = np};
  
  def splitvert(a:NodeTerm)(np:Int) = new SplitVertNode{inputs(0) = a; nparts = np};
  
  def sqrt(a:NodeTerm) = new SqrtNode{inputs(0) = a;};
  
  def stack(a:Array[NodeTerm]) = new StackNode{ninputs = a.length; Array.copy(a, 0, inputs, 0, a.length);};
  
  def sum(a:NodeTerm) = new SumNode{inputs(0) = a;}
  
  def tanh(a:NodeTerm) = new TanhNode{inputs(0) = a};
  
  implicit def NodeToNodeMat(n:Node):NodeMat = NodeMat.elem(n);

}
