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

@SerialVersionUID(100L)
class Layer(val net:Net, val opts:NodeOpts = new Node) extends LayerTerm(null, 0) {
  // Internal data arrays
  val _inputs = new Array[LayerTerm](1);
  val _outputs = new Array[Mat](1);
  val _derivs = new Array[Mat](1);
  def inputlength = _inputs.length
  var forwardtime = 0.0
  var backwardtime = 0.0
  override def layer = this
  def inputs = _inputs;
  
  private var _GUID = Mat.myrand.nextLong
  def setGUID(v:Long):Unit = {_GUID = v}
  def GUID:Long = _GUID
  
  // Setters and getters for general elements of those arrays
  def outputs(i:Int) = _outputs(i);
  def derivs(i:Int) = _derivs(i);  
  def input(i:Int) = _inputs(i);
  def apply(i:Int) = new LayerTerm(this, i);
  
  def setOutput(i:Int, v:Mat):Layer = {_outputs(i) = v; this}
  def setDeriv(i:Int, v:Mat):Layer = {_derivs(i) = v; this}
  def setInput(i:Int, v:LayerTerm) = {_inputs(i) = v; this}
  
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

  def createOutput = {
  	if (output.asInstanceOf[AnyRef] == null) output = inputData.zeros(inputData.nrows, inputData.ncols);
  }

  def createOutput(nrows:Int, ncols:Int) = {
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
  
  override def toString = {
    "layer@"+(hashCode % 0x10000).toString
  }
}


object Layer {  
}

class LayerTerm(val _layer:Layer, val term:Int) extends Serializable {
  def layer = _layer;
}

trait OutputLayer {}

class LayerOptions(val nlayers:Int) extends Serializable {
  
  val layerOptionss = new Array[Node](nlayers);
  
  def apply(i:Int):Node = layerOptionss(i);
  
  def update(i:Int, lopts:Node) = {layerOptionss(i) = lopts; this}
  
  override def clone = copyTo(new LayerOptions(nlayers));
  
  def copyTo(lopts:LayerOptions):LayerOptions = {
    for (i <- 0 until nlayers) {
      lopts.layerOptionss(i) = layerOptionss(i).clone;
      layerOptionss(i).myGhost = lopts.layerOptionss(i);
    }
    for (i <- 0 until nlayers) {
      for (j <- 0 until layerOptionss(i).inputs.length) {
      	if (layerOptionss(i).inputs(j) != null) lopts.layerOptionss(i).inputs(j) = layerOptionss(i).inputs(j).node.myGhost;
      }
    }
    lopts;
  }
}

object LayerFn {
  final val SIGMOIDFN = 0;
  final val TANHFN = 1;
  final val SOFTPLUSFN = 2;
  
  val fwdflops = irow(20, 20, 40);
  val bwdflops = irow(3, 3, 20);
  
  def applyfwd(a:Mat, ifn:Int):Mat = applyfwd(a, null, ifn);
  
  def applyfwd(a:Mat, out:Mat, ifn:Int):Mat = {
    Mat.nflops += 1L * a.length * fwdflops(ifn);
    a match {
      case af:FMat => {
        val omat = FMat.newOrCheckFMat(a.nrows, a.ncols, out, a.GUID, ifn, "LayerFn".##);
        CPUMACH.applyfwd(af.data, omat.data, ifn, a.length, Mat.numThreads);
        omat
      }
      case ag:GMat => {
        val omat = GMat.newOrCheckGMat(a.nrows, a.ncols, out, a.GUID, ifn, "LayerFn".##);
        CUMACH.applyfwd(ag.data, omat.data, ifn, a.length);
        omat
      }
    }
  }

  def applyderiv(a:Mat, b:Mat, ifn:Int):Mat = applyderiv(a, b, null, ifn)
      
  def applyderiv(a:Mat, b:Mat, out:Mat, ifn:Int):Mat = {
	  Mat.nflops += 1L * a.length * bwdflops(ifn);
    if (a.nrows != b.nrows || a.ncols != b.ncols) {
      throw new RuntimeException("applyderiv rows or columns dont match (%d %d) (%d %d" format (a.nrows, a.ncols, b.nrows, b.ncols));
    }
    (a, b) match {
      case (af:FMat, bf:FMat) => {
        val omat = FMat.newOrCheckFMat(a.nrows, a.ncols, out, a.GUID, ifn, "LayerFn".##);
        CPUMACH.applyderiv(af.data, bf.data, omat.data, ifn, a.length, Mat.numThreads);
        omat
      }
      case (ag:GMat, bg:GMat) => {
        val omat = GMat.newOrCheckGMat(a.nrows, a.ncols, out, a.GUID, ifn, "LayerFn".##);
        CUMACH.applyderiv(ag.data, bg.data, omat.data, ifn, a.length);
        omat
      }
    }
  }
}
 


