package BIDMach.io

import onnx.Onnx.{ModelProto,GraphProto,NodeProto,ValueInfoProto,TensorProto}
import java.io._
import java.nio.ByteBuffer
import scala.collection.JavaConversions._
import scala.collection.mutable.{HashMap,ArrayBuffer}
import BIDMat.{BMat,Mat,SBMat,CMat,CSMat,DMat,FMat,FFilter,GMat,GFilter,GDMat,GIMat,GLMat,GSMat,GSDMat,HMat,IMat,JSON,LMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.networks.Net
import BIDMach.networks.layers._
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._

class Onnx { 

  var nmodels = 0;

  var mmap:HashMap[String,Int] = null;
  var tmap:HashMap[String,Boolean] = null;
  var builder:ModelProto.Builder = null
  var outputMap:HashMap[String,Node] = null
  var inputMap:HashMap[String,ValueInfoProto] = null
  var graph:GraphProto = null
  var nodeProtos:java.util.List[NodeProto] = null;
  var inputs:java.util.List[ValueInfoProto] = null;
  var weights:java.util.List[TensorProto] = null;
  var weightsMap:HashMap[String,Mat] = null;
  var nodes:NodeSet = null;
  var modelmats:Array[Mat] = null;

  def readModel(modelFile:String, frontEnd:NodeSet=null, dataLinks:HashMap[String,Int]=null) = { 
    nmodels = 0;
    mmap = new HashMap[String,Int];
    tmap = new HashMap[String,Boolean];
    outputMap = new HashMap[String,Node];
    inputMap = new HashMap[String,ValueInfoProto];
    weightsMap = new HashMap[String,Mat];

    // Parse the graph and extract nodes and tensors
    builder = ModelProto.newBuilder()
    builder.mergeFrom(new FileInputStream(modelFile));
    graph = builder.getGraph
    nodeProtos = graph.getNodeList
    inputs = graph.getInputList
    weights = graph.getInitializerList

    // Create input map
    for (b <- inputs) { inputMap(b.getName) = b; }
    
    getInitializerData;

    createNodes(frontEnd, dataLinks);

    linkNodes;

    createModelmats;

    (nodes, modelmats)
  }

  def addModel2(name:String, name2:String, transposed:Boolean = false) = { 
    if (!mmap.contains(name)) { 
      mmap(name) = nmodels;
      tmap(name) = transposed;
      if (name2 != null) {mmap(name2) = nmodels+1; tmap(name2) = false}
      nmodels += 2;
    }
  }

  def addModel4(name:String, name2:String, name3:String, name4:String) = { 
    if (!mmap.contains(name)) { 
      mmap(name) = nmodels;
      tmap(name) = false;
      if (name2 != null) {mmap(name2) = nmodels+1; tmap(name2) = false}
      if (name3 != null) {mmap(name3) = nmodels+2; tmap(name3) = false}
      if (name4 != null) {mmap(name4) = nmodels+3; tmap(name4) = false}
      nmodels += 4;
    }
  }

  def createConvNode(node:NodeProto) = { 
    val pads = izeros(1, 4)
    val kernel_shape = iones(1, 2)
    val strides = iones(1, 2)
    for (attrib <- node.getAttributeList) { 
      attrib.getName match { 
	case "pads" => { 
	  pads(0) = attrib.getInts(0).toInt
	  pads(1) = attrib.getInts(1).toInt
	  pads(2) = attrib.getInts(2).toInt
	  pads(3) = attrib.getInts(3).toInt
	}
	case "kernel_shape" => { 
	  kernel_shape(0) = attrib.getInts(0).toInt
	  kernel_shape(1) = attrib.getInts(1).toInt
	}
	case "strides" => { 
	  strides(0) = attrib.getInts(0).toInt
	  strides(1) = attrib.getInts(1).toInt
	}
      }
    }
    val mname = node.getInput(1);
    val hb = (node.getInputCount > 2);
    val initf:(Mat,Float)=>Mat = Net.xavier
    val initv0:Float = 1f
    val initbiasf:(Mat,Float)=>Mat = Net.constant
    val initbiasv0:Float = 0f
    val lrs:Float=1f
    val bs:Float=1f
    val ct=Net.UseNetConvType
    
    val weightSpec=inputMap(node.getInput(1));
    val wdims = weightSpec.getType.getTensorType.getShape
    val odim = wdims.getDim(0).getDimValue.toInt

    val newnode = new ConvNode{inputs(0)=null; modelName=mname; kernel=kernel_shape; noutputs=odim; stride=strides; pad=pads; hasBias=hb; 
			       initfn = initf; initv = initv0; initbiasfn = initbiasf; initbiasv = initbiasv0;}
    outputMap(node.getOutput(0)) = newnode
    addModel2(node.getInput(1), if (node.getInputCount > 2) node.getInput(2) else null);
    newnode
  }

  def createGemmNode(node:NodeProto) = { 
    val hBias = (node.getInputCount > 2);
    val mname = node.getInput(1);
    val initf = Net.xavier;
    val initv0 = 1f;
    val initbiasf = Net.constant;
    val initbiasv0 = 0f;
    val lrs = 1f;
    val bs = 1f;

    val weightSpec=inputMap(node.getInput(1));
    val wdims = weightSpec.getType.getTensorType.getShape
    val odim = wdims.getDim(0).getDimValue.toInt

    val newnode = new LinNode{inputs(0)=null; modelName = mname; outdim=odim; hasBias=hBias; initfn = initf; initv = initv0; 
			      initbiasfn = initbiasf; initbiasv = initbiasv0; lr_scale=lrs; bias_scale=bs;};
    outputMap(node.getOutput(0)) = newnode
    addModel2(node.getInput(1), if (node.getInputCount > 2) node.getInput(2) else null, true);
    newnode
  }

  def createMaxPoolNode(node:NodeProto) = { 
    val pads = izeros(1, 4)
    val kernel_shape = iones(1, 2)
    val strides = iones(1, 2)
    for (attrib <- node.getAttributeList) { 
      attrib.getName match { 
	case "pads" => { 
	  pads(0) = attrib.getInts(0).toInt
	  pads(1) = attrib.getInts(1).toInt
	  pads(2) = attrib.getInts(2).toInt
	  pads(3) = attrib.getInts(3).toInt
	}
	case "kernel_shape" => { 
	  kernel_shape(0) = attrib.getInts(0).toInt
	  kernel_shape(1) = attrib.getInts(1).toInt
	}
	case "strides" => { 
	  strides(0) = attrib.getInts(0).toInt
	  strides(1) = attrib.getInts(1).toInt
	}
      }
    }
    val newnode = new PoolingNode{inputs(0)=null; h=kernel_shape(0); w=kernel_shape(1); stride=strides(0); pad=pads(0);}
    outputMap(node.getOutput(0)) = newnode
    newnode
  }

  def createLRNNode(node:NodeProto) = { 
    var alpha0 = 0.00001f
    var beta0 = 0.75f
    var bias = 1.0f
    var size = 1
    for (attrib <- node.getAttributeList) { 
      attrib.getName match { 
	case "alpha" => alpha0 = attrib.getF
	case "beta" => beta0 = attrib.getF
	case "bias" => bias = attrib.getF
	case "size" => size = attrib.getI.toInt
      }
    }
    val newnode = new LRNacrossNode{inputs(0)=null; dim=size; alpha=alpha0; beta=beta0; k=bias}
    outputMap(node.getOutput(0)) = newnode
    newnode
  }

  def createAveragePoolNode(node:NodeProto) = { 
    val pads = izeros(1, 4)
    val kernel_shape = iones(1, 2)
    val strides = iones(1, 2)
    for (attrib <- node.getAttributeList) { 
      attrib.getName match { 
	case "pads" => { 
	  pads(0) = attrib.getInts(0).toInt
	  pads(1) = attrib.getInts(1).toInt
	  pads(2) = attrib.getInts(2).toInt
	  pads(3) = attrib.getInts(3).toInt
	}
	case "kernel_shape" => { 
	  kernel_shape(0) = attrib.getInts(0).toInt
	  kernel_shape(1) = attrib.getInts(1).toInt
	}
	case "strides" => { 
	  strides(0) = attrib.getInts(0).toInt
	  strides(1) = attrib.getInts(1).toInt
	}
      }
    }
    val newnode = new PoolingNode{inputs(0)=null; h=kernel_shape(0); w=kernel_shape(1); stride=strides(0); pad=pads(0);
				  poolingMode=cudnnPoolingMode.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING}
    outputMap(node.getOutput(0)) = newnode
    newnode
  }

  def createGlobalAveragePoolNode(node:NodeProto) = { 
    val newnode = new PoolingNode{inputs(0)=null; h= -1; w= -1; stride=1; pad=0;
				  poolingMode=cudnnPoolingMode.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING}
    outputMap(node.getOutput(0)) = newnode
    newnode
  }

  def createGlobalMaxPoolNode(node:NodeProto) = { 
    val newnode = new PoolingNode{inputs(0)=null; h= -1; w= -1; stride=1; pad=0;}
    outputMap(node.getOutput(0)) = newnode
    newnode
  }

  def createBatchNormScaleNode(node:NodeProto) = { 
    var epsilon = 1e-5f
    var momentum = 0.9f;
    var doSpatial = 1;
    for (attrib <- node.getAttributeList) { 
      attrib.getName match { 
	case "epsilon" => epsilon = attrib.getF
	case "momentum" => momentum = attrib.getF
	case "spatial" => doSpatial = attrib.getI.toInt
      }
    }
    val avgFactor = 1f - momentum;

    val mname = node.getInput(1);
    val normMode = if (doSpatial == 1) BatchNormLayer.SPATIAL else BatchNormLayer.PER_ACTIVATION
    val hb = true;
    val lrs = 1f
    val bs = 1f
    val inp = Net.UseNetPlacing
    val newnode = new BatchNormScaleNode{inputs(0)=null; modelName=mname; expAvgFactor=avgFactor; batchNormMode=normMode; 
					 hasBias=hb; lr_scale=lrs; bias_scale=bs; inplace=inp}    
    outputMap(node.getOutput(0)) = newnode
    addModel4(node.getInput(1), node.getInput(2), node.getInput(3), node.getInput(4))
    newnode
  }


  def createReshapeNode(node:NodeProto) = { 
    val shapeSpec=reverse(IMat(weightsMap(node.getInput(1))).t)
    //  val dims = shapeSpec.getType.getTensorType.getShape
    val newnode = new ReshapeNode{inputs(0)=null; dims=shapeSpec}
    outputMap(node.getOutput(0)) = newnode
    newnode
  }

  def createReluNode(node:NodeProto) = { 
    val inplac = Net.UseNetPlacing
    val newnode = new RectNode{inputs(0) = null; inplace = inplac};
    outputMap(node.getOutput(0)) = newnode
    newnode
  }

  def createSumNode(node:NodeProto) = { 
    val newnode = new AddNode{inputs(0) = null;};
    outputMap(node.getOutput(0)) = newnode
    newnode
  }

  def createConcatNode(node:NodeProto) = { 
    var axis = 1
    for (attrib <- node.getAttributeList) { 
      attrib.getName match { 
	case "axis" => axis = attrib.getI.toInt
      }
    }
    val newnode = new StackNode{inputs(0) = null;};
    outputMap(node.getOutput(0)) = newnode
    newnode
  }

  def createDropoutNode(node:NodeProto) = { 
    var ratio = 0.5f
    for (attrib <- node.getAttributeList) { 
      attrib.getName match { 
	case "ratio" => ratio = attrib.getF
      }
    }
    val newnode = new DropoutNode{inputs(0) = null; frac=ratio};
    outputMap(node.getOutput(0)) = newnode
    newnode
  }

  def createSoftmaxNode(node:NodeProto, outputName:String) = { 
    val newnode = if (outputName.matches(node.getOutput(0))) { 
      val scoreTyp = 1
      val lossTyp = 1
      new SoftmaxOutputNode{inputs(0) = null; scoreType=scoreTyp; lossType = lossTyp}
    } else { 
      new SoftmaxNode{inputs(0) = null;};
    }
    outputMap(node.getOutput(0)) = newnode
    newnode
  }

  def getInitializerData { 
    for (w <- weights) { 
      val dims0 = w.getDimsList.map(_.toInt).toArray
      val dims = dims0.size match { 
	case 4 => Array(dims0(1), dims0(3), dims0(2), dims0(0))
	case 2 => Array(dims0(1), dims0(0))
	case 1 => Array(dims0(0), 1)
      }
      val ttype = w.getDataType;
      val bb = w.getRawData.toByteArray;
      val bbw = ByteBuffer.wrap(bb);
      bbw.order(java.nio.ByteOrder.LITTLE_ENDIAN);
      val mat = ttype match { 
	case 1 => { 
	  val m = FMat.make(dims);
	  val fb = bbw.asFloatBuffer
	  for (i <- 0 until m.length) m(i) = fb.get(i);
	  m
	}
	case 2 => { 
	  val m = BMat.make(dims);
	  for (i <- 0 until m.length) m(i) = bb(i);
	  m
	}
	case 3 => { 
	  val m = BMat.make(dims);
	  for (i <- 0 until m.length) m(i) = bb(i);
	  m
	}
	case 6 => { 
	  val m = IMat.make(dims);
	  val fb = bbw.asIntBuffer
	  for (i <- 0 until m.length) m(i) = fb.get(i);
	  m
	}
	case 7 => { 
	  val m = LMat.make(dims);
	  val fb = bbw.asLongBuffer
	  for (i <- 0 until m.length) m(i) = fb.get(i);
	  m
	}
	case 11 => { 
	  val m = DMat.make(dims);
	  val fb = bbw.asDoubleBuffer
	  for (i <- 0 until m.length) m(i) = fb.get(i);
	  m
	}
      }
      weightsMap(w.getName) = mat;
    }
  }
  
  def createNodes(frontEnd:NodeSet, dataLinks:HashMap[String,Int]) { 
    val nfront = if (frontEnd.asInstanceOf[AnyRef] == null) 0 else frontEnd.size
    nodes = new NodeSet(nodeProtos.size + nfront);
    if (nfront > 0) { 
      for (i <- 0 until nfront) nodes(i) = frontEnd(i);
      for ((k,v) <- dataLinks) outputMap(k) = nodes(v);
    }
    var i = nfront
    for (node <- nodeProtos) { 
      val newnode = node.getOpType match { 
	case "Conv" => createConvNode(node);
	case "BatchNormalization" => createBatchNormScaleNode(node);
	case "Relu" => createReluNode(node);
	case "MaxPool" => createMaxPoolNode(node);
	case "AveragePool" => createAveragePoolNode(node);
	case "GlobalAveragePool" => createGlobalAveragePoolNode(node);
	case "Sum" => createSumNode(node);
	case "Concat" => createConcatNode(node);
	case "Dropout" => createDropoutNode(node);
	case "LRN" => createLRNNode(node);
	case "Reshape" => createReshapeNode(node);
	case "Gemm" => createGemmNode(node);
	case "Softmax" => createSoftmaxNode(node, graph.getOutput(0).getName);
      }
      nodes(i) = newnode
      i += 1;
    }
  }

  def linkToInputs(nodep:NodeProto, ninputs:Int) { 
    val node = outputMap(nodep.getOutput(0));
    for (i <- 0 until ninputs) { 
      val inname = nodep.getInput(i)
      val innode = outputMap(inname);
      node.inputs(i) = innode;
    }
  }

  def linkNodes { 
    for (node <- nodeProtos) { 
      node.getOpType match { 
	case "Conv" => linkToInputs(node, 1);
	case "BatchNormalization" => linkToInputs(node, 1);
	case "Relu" => linkToInputs(node, 1);
	case "MaxPool" => linkToInputs(node, 1);
	case "AveragePool" => linkToInputs(node, 1);
	case "GlobalAveragePool" => linkToInputs(node, 1);
	case "Sum" => linkToInputs(node, node.getInputCount);
	case "Concat" => linkToInputs(node, node.getInputCount);
	case "Dropout" => linkToInputs(node, 1);
	case "LRN" => linkToInputs(node, 1);	
	case "Reshape" => linkToInputs(node, 1);
	case "Gemm" => linkToInputs(node, 1);
	case "Softmax" => linkToInputs(node, 1);
      }
    }
  }

  def createModelmats { 
    modelmats = new Array[Mat](nmodels)
    for (w <- weights) { 
      val wname = w.getName
      if (weightsMap.contains(wname) && mmap.contains(wname)) { 
	val data = weightsMap(wname)
	modelmats(mmap(wname)) = if (tmap(wname)) data.t else data
      }
    }
  }

  val perm = irow(2,1,0);
  def fromRGBtoBGR(a:Mat):Mat = {
    val dd = a.dims
    val aa = a.reshapeView(dd(1),dd(2),dd(0),dd(3));
    val bb = aa(?,?,perm,?)
    bb.reshapeView(dd);
  }
}


