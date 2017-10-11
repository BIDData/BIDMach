package BIDMach.extern

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,JSON,SMat,SDMat,TMat,FFilter,Filter,GFilter}
import BIDMat.MatFunctions._
import BIDMach._
import BIDMach.datasources.DataSource
import BIDMach.networks.Net
import BIDMach.networks.layers._
import scala.collection.JavaConversions._
import scala.collection.mutable
import java.io.InputStream
import _root_.caffe.Caffe
import _root_.caffe.Caffe.LRNParameter.NormRegion
import _root_.caffe.Caffe.PoolingParameter.PoolMethod
import com.google.protobuf._
import jcuda.jcudnn.cudnnPoolingMode

object CaffeIO {
  def loadUntrainedModel(fin:Readable, net:Net):Unit = {
    val caffeBuilder = Caffe.NetParameter.newBuilder()
    TextFormat.merge(fin, caffeBuilder)
    parseProtobuf(caffeBuilder, net)
  }
  
  def loadTrainedModel(fin:InputStream, net:Net):Unit = {
    val cis = CodedInputStream.newInstance(fin)
    cis.setSizeLimit(1 << 30)
    val netParam = Caffe.NetParameter.parseFrom(cis)
    parseProtobuf(netParam, net)
  }

  private def parseProtobuf(netParam:Caffe.NetParameterOrBuilder, net:Net) = {
    // TODO: enforce NCHW if necessary
    // Translate every layer and build a mapping of blobs to layers feeding into them
    val nodes = new mutable.ArrayBuffer[Node]
    // TODO: make sure that either everything is trained or nothing is
    val modelMats = new mutable.ArrayBuffer[Mat]
    val nodesWithTop = new mutable.HashMap[String,mutable.Buffer[Node]]
    for (layer <- netParam.getLayerList()) {
      layer.getType() match {
        case "Convolution" => translateConvolution(layer, nodes, modelMats, net)
        case "Pooling" => translatePooling(layer, nodes)
        case "LRN" => translateLRN(layer, nodes)
        case "BatchNorm" => {
          val bnParam = layer.getBatchNormParam()
          nodes += new BatchNormNode {
            epsilon = bnParam.getEps()
            expAvgFactor = bnParam.getMovingAverageFraction()
          }
        }

        case "SoftmaxWithLoss" => nodes += new SoftmaxOutputNode
        case "HingeLoss" => nodes += new GLMNode { links = irow(3) }
        case "Accuracy" => nodes += new AccuracyNode

        case "ReLU" => nodes += new RectNode
        case "Sigmoid" => nodes += new SigmoidNode
        case "TanH" => nodes += new TanhNode
        case "BNLL" => nodes += new SoftplusNode

        case "Data" => {
          val dataParam = layer.getDataParam()
          
          if (net.opts.isInstanceOf[DataSource.Opts]) {
            net.opts.asInstanceOf[DataSource.Opts].batchSize = dataParam.getBatchSize()
          }
          
          nodes += new InputNode
        }
        case "MemoryData" => nodes += new InputNode
        case "HDF5Data" => nodes += new InputNode

        case "InnerProduct" => translateInnerProduct(layer, nodes, modelMats)
        case "Split" => nodes += new CopyNode
        case "Softmax" => nodes += new SoftmaxNode
        case "Dropout" => {
          val dropoutParam = layer.getDropoutParam()
          nodes += new DropoutNode { frac = dropoutParam.getDropoutRatio() }
        }
        // TODO: implement base, shift, scale for the following two
        case "Exp" => nodes += new ExpNode
        case "Log" => nodes += new LnNode
        case "Scale" => {
          val scaleParam = layer.getScaleParam()
          nodes += new ScaleNode { hasBias = scaleParam.getBiasTerm() }
        }
        // TODO: change once we implement all layer types
        case _ => throw new NotImplementedError("\"%s\" is not implemented yet" format layer.getType())
      }
      
      if (nodes.last.isInstanceOf[ModelNode]) {
        val modelNode = nodes.last.asInstanceOf[ModelNode]
        
        if (layer.getParamCount() >= 1) {
          modelNode.lr_scale = layer.getParam(0).getLrMult()
          
          if (layer.getParamCount() >= 2) {
            modelNode.bias_scale = layer.getParam(1).getLrMult()
          }
        }
      }
      
      for (t <- layer.getTopList()) {
        nodesWithTop.getOrElseUpdate(t, new mutable.ArrayBuffer) += nodes.last
      }
    }
    
    // Assign layer inputs based on which Caffe blobs each layer links to.
    // When several layers reuse the same blob, order the layers in the order they were specified
    // in the prototxt.
    val blobIterIndices = new mutable.HashMap[String,Int]
    for ((layer, curNode) <- netParam.getLayerList() zip nodes) {
      // XXX this should account for multiple bottom blobs
      if (layer.getBottomCount() >= 1) {
        val bottom = layer.getBottom(0)
        if (layer.getTopList().contains(bottom)) {
          val j = blobIterIndices.getOrElse(bottom, 0)
          curNode.inputs(0) = nodesWithTop(bottom)(j)
          blobIterIndices(bottom) = j + 1
        } else {
          curNode.inputs(0) = nodesWithTop(bottom)(nodesWithTop(bottom).length - 1)
        }
      }
    }
    
    // Assign batch norm modes
    for (node <- nodes.filter(_.isInstanceOf[BatchNormNode]).map(_.asInstanceOf[BatchNormNode])) {
      if (node.inputs(0).isInstanceOf[ConvNode]) {
        node.batchNormMode = BatchNormLayer.Spatial
      } else {
        node.batchNormMode = BatchNormLayer.PerActivation
      }
    }
    
    // Create a table of forward links
    val downstreamNodes = new mutable.HashMap[Node,mutable.Buffer[Node]]
    for (node <- nodes) {
      for (prevNode <- node.inputs.filter(_ != null)) {
        downstreamNodes.getOrElseUpdate(prevNode.node, new mutable.ArrayBuffer) += node
      }
    }
    
    // Combine batch norm layers immediately followed by scale layers
    val newNodes = new mutable.ArrayBuffer[Node]
    val skip = new mutable.HashSet[Node]
    for (node <- nodes) {
      if (node.isInstanceOf[BatchNormNode] && downstreamNodes(node).length == 1
          && downstreamNodes(node)(0).isInstanceOf[ScaleNode]) {
        val bnNode = node.asInstanceOf[BatchNormNode]
        val scaleNode = downstreamNodes(node)(0).asInstanceOf[ScaleNode]
        val combinedNode = new BatchNormScaleNode {
          epsilon = bnNode.epsilon
          expAvgFactor = bnNode.expAvgFactor
          tensorFormat = bnNode.tensorFormat
          batchNormMode = bnNode.batchNormMode
          
          hasBias = scaleNode.hasBias
          imodel = scaleNode.imodel
        }
        
        Array.copy(bnNode.inputs, 0, combinedNode.inputs, 0, bnNode.inputs.length)
        for (downstream <- downstreamNodes(scaleNode)) {
          for (i <- 0 until downstream.inputs.length) {
            if (downstream.inputs(i) == scaleNode) {
              downstream.inputs(i) = combinedNode
            }
          }
        }
        
        newNodes += combinedNode
        skip += scaleNode
      } else if (!skip.contains(node)) {
        newNodes += node
      }
    }

    net.opts.nodeset = new NodeSet(newNodes.toArray)
    if (modelMats.length > 0) {
      net.setmodelmats(modelMats.toArray)
      net.opts.nmodelmats = modelMats.length
    }
  }
  
  private def translateConvolution(layer:Caffe.LayerParameter, nodes:mutable.Buffer[Node], modelMats:mutable.Buffer[Mat], net:Net) = {
    val convParam = layer.getConvolutionParam()
    
    val convNode = new ConvNode {
      noutputs = convParam.getNumOutput()
      hasBias = convParam.getBiasTerm()
      if (convParam.hasPadW()) {
        pad = convParam.getPadW() \ convParam.getPadH()
      } else if (convParam.getPadCount() == 0) {
        pad = irow(0)
      } else {
        pad = irow(convParam.getPadList().map(_.intValue()).toList)
      }
      if (convParam.hasKernelW()) {
        kernel = convParam.getKernelW() \ convParam.getKernelH()
      } else {
        kernel = irow(convParam.getKernelSizeList().map(_.intValue()).toList)
      }
      if (convParam.hasStrideW()) {
        stride = convParam.getStrideW() \ convParam.getStrideH()
      } else if (convParam.getStrideCount() == 0) {
        pad = irow(1)
      } else {
        stride = irow(convParam.getStrideList().map(_.intValue()).toList)
      }
      dilation = irow(convParam.getDilationList().map(_.intValue()).toList)
      
      // BIDMach (currently) only supports xavier initialization
      if (convParam.getWeightFiller().getType() != "xavier") {
        throw new NotImplementedError("Only xavier initialization is currently implemented for convolution layers")
      }
    }
    
    if (layer.getBlobsCount() > 0) {
      if (!convNode.hasBias && layer.getBlobsCount() != 1) {
        throw new IllegalArgumentException("Convolution layer without bias needs 1 matrix")
      }
      if (convNode.hasBias && layer.getBlobsCount() != 2) {
        throw new IllegalArgumentException("Convolution layer with bias needs 2 matrices")
      }
      
      convNode.imodel = modelMats.length

      // TODO: avoid duplicating code with ConvLayer here
      val shape = layer.getBlobs(0).getShape().getDimList().map(_.intValue()).toArray
      val filter = FFilter2Ddn(shape(3), shape(2), shape(1), shape(0), convNode.stride(0), convNode.pad(0))
      // TODO: is this an abstraction barrier violation
      layer.getBlobs(0).getDataList().map(_.floatValue()).copyToArray(filter.data)
      modelMats += (if (net.opts.useGPU && Mat.hasCUDA > 0 && Mat.hasCUDNN) {
        val x = GFilter(filter)
        x.convType = convNode.convType
        x.setTensorFormat(Net.getCUDNNformat(convNode.tensorFormat, net.opts.tensorFormat));
        x
      } else {
        filter
      })
      
      if (convNode.hasBias) {
        val n = layer.getBlobs(1).getShape().getDim(0).toInt
        modelMats += blob2Mat(layer.getBlobs(1)).reshapeView(n, 1, 1, 1)
      } else {
        modelMats += null
      }
    }
    
    nodes += convNode
  }
  
  private def translatePooling(layer:Caffe.LayerParameter, nodes:mutable.Buffer[Node]) = {
    val poolingParam = layer.getPoolingParam()
    nodes += new PoolingNode {
      if (poolingParam.hasPadH()) {
        pady = poolingParam.getPadH()
        padx = poolingParam.getPadW()
      } else {
        pady = poolingParam.getPad()
        padx = pady
      }
      
      if (poolingParam.hasKernelH()) {
        h = poolingParam.getKernelH()
        w = poolingParam.getKernelW()
      } else {
        h = poolingParam.getKernelSize()
        w = h
      }
      
      if (poolingParam.hasStrideH()) {
        stridey = poolingParam.getStrideH()
        stridex = poolingParam.getStrideW()
      } else {
        stridey = poolingParam.getStride()
        stridex = stridey
      }

      poolingMode = poolingParam.getPool() match {
        case PoolMethod.MAX => cudnnPoolingMode.CUDNN_POOLING_MAX
        case PoolMethod.AVE => cudnnPoolingMode.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
        case PoolMethod.STOCHASTIC => throw new NotImplementedError("Stochastic pooling is not supported yet")
      }
    }
  }
  
  private def translateLRN(layer:Caffe.LayerParameter, nodes:mutable.Buffer[Node]) = {
    val lrnParam = layer.getLrnParam()
    if (lrnParam.getNormRegion() == NormRegion.WITHIN_CHANNEL) {
      nodes += new LRNwithinNode {
        dim = lrnParam.getLocalSize()
        alpha = lrnParam.getAlpha()
        beta = lrnParam.getBeta()
      }
    } else {
      nodes += new LRNacrossNode {
        dim = lrnParam.getLocalSize()
        alpha = lrnParam.getAlpha()
        beta = lrnParam.getBeta()
        k = lrnParam.getK()
      }
    }
  }
  
  private def translateInnerProduct(layer:Caffe.LayerParameter, nodes:mutable.Buffer[Node], modelMats:mutable.Buffer[Mat]) = {
    val ipp = layer.getInnerProductParam()
    val linNode = new LinNode {
      outdim = ipp.getNumOutput()
      hasBias = ipp.getBiasTerm()
    }
    if (layer.getBlobsCount() > 0) {
      linNode.imodel = modelMats.length
      if (!linNode.hasBias) {
        if (layer.getBlobsCount() != 1) {
          throw new IllegalArgumentException("Linear layer without bias needs 1 matrix")
        }
        addMatsFromBlobs(modelMats, linNode, layer)
        modelMats += null
      } else {
        if (layer.getBlobsCount() != 2) {
          throw new IllegalArgumentException("Linear layer without bias needs 2 matrices")
        }
        if (layer.getBlobs(0).getShape().getDim(0) != layer.getBlobs(1).getShape().getDim(0)) {
          throw new IllegalArgumentException("Weight and bias dimensions for linear layer don't agree")
        }
        if ((layer.getBlobs(0).getDoubleDataCount() > 0) != (layer.getBlobs(1).getDoubleDataCount() > 0)) {
          throw new IllegalArgumentException("Weight and bias matrices must both be double data or both be single data")
        }
        
        // We unfortunately can't use addMatsFromBlobs here because Caffe's bias blob is 1D, while BIDMach wants
        // a 2D bias where the second dimension is 1.
        val outDim = layer.getBlobs(0).getShape().getDim(0).intValue()
        val weightMat = blob2Mat(layer.getBlobs(0))
        val biasMat = blob2Mat(layer.getBlobs(1)).reshape(outDim, 1)
        modelMats += weightMat
        modelMats += biasMat
      }
    }
    nodes += linNode
  }
  
  /**
   * Converts each blob in the given layer to a Mat, and appends these Mats to {@code modelMats}.
   * Additionally sets the {@code imodel} value of {@code node} to the correct value.
   * Automatically converts data from row-major order to column-major order.
   */
  private def addMatsFromBlobs(modelMats:mutable.Buffer[Mat], node:ModelNode, layer:Caffe.LayerParameter) = {
    node.imodel = modelMats.length
    for (blob <- layer.getBlobsList()) {
      modelMats += blob2Mat(blob)
    }
  }
  
  /**
   * Converts the given blob into a Mat.
   * Automatically converts data from row-major order to column-major order.
   */
  private def blob2Mat(blob:Caffe.BlobProto):Mat = {
    // We convert from row-major to column-major by creating a Mat with reversed dimensions,
    // loading it up with the row-major data, and then performing a deep transpose
    val dimList = blob.getShape().getDimList()
    val reverseDims = dimList.map(_.intValue()).reverse.toArray
    if (blob.getDoubleDataCount() > 0) {
      val data = blob.getDoubleDataList().map(_.doubleValue()).toArray
      // TODO: should I bother with GDMat
      new DMat(reverseDims, data).transpose((reverseDims.length - 1) to 0 by -1)
    } else {
      val data = blob.getDataList().map(_.floatValue()).toArray
      // TODO: should I bother with GFMat
      new FMat(reverseDims, data).transpose((reverseDims.length - 1) to 0 by -1)
    }
  }
}
