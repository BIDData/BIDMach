package BIDMach.extern

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,JSON,SMat,SDMat,TMat,FFilter,Filter,GFilter}
import BIDMat.MatFunctions._
import BIDMach._
import BIDMach.datasources.DataSource
import BIDMach.networks.Net
import BIDMach.networks.layers._
import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.Option
import scala.util.control.Breaks._
import java.io.InputStream
import _root_.caffe.Caffe
import _root_.caffe.Caffe.LRNParameter.NormRegion
import _root_.caffe.Caffe.PoolingParameter.PoolMethod
import com.google.protobuf._
import jcuda.jcudnn.cudnnPoolingMode

object CaffeIO {
  private class CaffeLayer(val param:Caffe.LayerParameter) {
    val inputs = new mutable.ArrayBuffer[CaffeLayer]
    var inodeFirst = -1
    var inodeLast = -1
  }

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
    // Caffe only supports CrossCorrelation convolution
    net.opts.convType = Net.CrossCorrelation
    
    val layers = toposort(resolveLayerLinks(netParam.getLayerList()))

    // TODO: enforce NCHW if necessary
    // Translate every layer and build a mapping of blobs to layers feeding into them
    val nodes = new mutable.ArrayBuffer[Node]
    // TODO: make sure that either everything is trained or nothing is
    val modelMats = new mutable.ArrayBuffer[Mat]
    implicit def nodeOnly(node:Node):(Array[Node],Seq[Mat]) = (Array(node), null)
    var i = 0
    while (i < layers.length) {
      val layer = layers(i)
      var incr = 1

      var (newNodes, newMats):(Array[Node],Seq[Mat]) = layer.param.getType() match {
        case "Convolution" => translateConvolution(layer, net)
        case "Pooling" => translatePooling(layer)
        case "LRN" => translateLRN(layer)
        case "BatchNorm" => {
          val bnParam = layer.param.getBatchNormParam()
          if (i + 1 < layers.length && layers(i + 1).inputs.contains(layer) && layers(i + 1).param.getType() == "Scale") {
            // Combine this layer and the next into a single BatchNormScale node
            incr = 2
            val scaleParam = layers(i + 1).param.getScaleParam()
            val node = translateBatchNorm(layer, scaleParam)
            layers(i + 1).inodeFirst = nodes.length
            layers(i + 1).inodeLast = nodes.length
            node
          } else {
            translateBatchNorm(layer, null)
          }
        }

        case "SoftmaxWithLoss" => new SoftmaxOutputNode
        case "HingeLoss" => new GLMNode { links = irow(3) }
        case "Accuracy" => new AccuracyNode

        case "ReLU" => new RectNode
        case "Sigmoid" => new SigmoidNode
        case "TanH" => new TanhNode
        case "BNLL" => new SoftplusNode

        case "Data" => {
          val dataParam = layer.param.getDataParam()
          
          if (net.opts.isInstanceOf[DataSource.Opts]) {
            net.opts.asInstanceOf[DataSource.Opts].batchSize = dataParam.getBatchSize()
          }
          
          (addTransformNodes(layer.param.getTransformParam(), new InputNode), null)
        }
        case "MemoryData" => new InputNode
        case "HDF5Data" => new InputNode

        case "InnerProduct" => translateInnerProduct(layer)
        case "Split" => new CopyNode
        case "Softmax" => new SoftmaxNode
        case "Dropout" => {
          val dropoutParam = layer.param.getDropoutParam()
          new DropoutNode { frac = dropoutParam.getDropoutRatio() }
        }
        // TODO: implement base, shift, scale for the following two
        case "Exp" => new ExpNode
        case "Log" => new LnNode
        case "Scale" => {
          val scaleParam = layer.param.getScaleParam()
          new ScaleNode { hasBias = scaleParam.getBiasTerm() }
        }
        // TODO: change once we implement all layer types
        case unknownType => throw new NotImplementedError("\"%s\" is not implemented yet" format unknownType)
      }
      
      newNodes match {
        case Array(modelNode:ModelNode) => {
          if (layer.param.getParamCount() >= 1) {
            modelNode.lr_scale = layer.param.getParam(0).getLrMult()
            
            if (layer.param.getParamCount() >= 2) {
              modelNode.bias_scale = layer.param.getParam(1).getLrMult()
            }
          }
          
          if (newMats ne null) {
            modelNode.imodel = modelMats.length
            modelMats ++= newMats
          }
        }
        case _ =>
      }
      
      layer.inodeFirst = nodes.length
      layer.inodeLast = nodes.length + newNodes.length - 1

      for ((input, i) <- layer.inputs.zipWithIndex) {
        newNodes(0).inputs(i) = nodes(input.inodeLast)
      }
      
      nodes ++= newNodes
      i += incr
    }

    net.opts.nodeset = new NodeSet(nodes.toArray)
    if (modelMats.length > 0) {
      net.setmodelmats(modelMats.toArray)
      net.opts.nmodelmats = modelMats.length
    }
  }
  
  /** Creates a list of {@code Layer} objects, setting their {@code lowers} and {@code uppers}
   *  attributes based on their links to blobs in the protobuf.
   */
  private def resolveLayerLinks(layerList:Seq[Caffe.LayerParameter]):Seq[CaffeLayer] = {
    val layerBuf = new mutable.ArrayBuffer[CaffeLayer]
    for (layerParam <- layerList) {
      layerBuf += new CaffeLayer(layerParam)
    }
    
    // Make a table of top -> layers whose params have that top
    val layersWithTop = new mutable.HashMap[String,mutable.Buffer[CaffeLayer]]
    for (layer <- layerBuf) {
      for (t <- layer.param.getTopList()) {
        layersWithTop.getOrElseUpdate(t, new mutable.ArrayBuffer) += layer
      }
    }
    
    // Assign layer inputs based on which Caffe blobs each layer links to.
    // When several layers reuse the same blob in-place, order the layers in the order they were
    // specified in the prototxt.
    val blobIterIndices = new mutable.HashMap[String,Int]
    for (layer <- layerBuf) {
      // XXX this should account for multiple bottom blobs
      if (layer.param.getBottomCount() >= 1) {
        val bottom = layer.param.getBottom(0)
        if (layer.param.getTopList().contains(bottom)) {
          val j = blobIterIndices.getOrElse(bottom, 0)
          layer.inputs += layersWithTop(bottom)(j)
          blobIterIndices(bottom) = j + 1
        } else {
          layer.inputs += layersWithTop(bottom).last
        }
      }
    }
    
    layerBuf
  }
  
  /** Toposort the list of layers if necessary.
   *  It's highly unlikely this is out of order, but might as well sort if necessary.
   */
  private def toposort(layers:Seq[CaffeLayer]):Seq[CaffeLayer] = {
    // Check to see if it's already toposorted. If so, don't recreate the list.
    val seen = new mutable.HashSet[CaffeLayer]
    var ok = true
    breakable {
      for (layer <- layers) {
        for (input <- layer.inputs) {
          if (!seen.contains(input)) {
            ok = false
            break
          }
        }
        seen += layer
      }
    }

    if (ok) {
      layers
    } else {
      // Slow path: do the sort
      // Remember that input pointers point in the opposite direction of data flow. Hence, inputs are sinks.
      val sorted = new mutable.ArrayBuffer[CaffeLayer]
      val previsited = new mutable.HashSet[CaffeLayer]
      val postvisited = new mutable.HashSet[CaffeLayer]
      def visit(layer:CaffeLayer):Unit = {
        if (!postvisited.contains(layer)) {
          if (!previsited.contains(layer)) throw new IllegalArgumentException("Cycle detected")
          previsited += layer
          for (input <- layer.inputs) {
            visit(input)
          }
          postvisited += layer
          sorted += layer
        }
      }
      for (layer <- layers if (!postvisited.contains(layer))) {
        visit(layer)
      }
      sorted
    }
  }
  
  private def translateConvolution(layer:CaffeLayer, net:Net) = {
    val convParam = layer.param.getConvolutionParam()
    
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
        stride = irow(1)
      } else {
        stride = irow(convParam.getStrideList().map(_.intValue()).toList)
      }
      dilation = irow(convParam.getDilationList().map(_.intValue()).toList)
      
      // BIDMach (currently) only supports xavier initialization
      if (convParam.getWeightFiller().getType() != "xavier") {
        throw new NotImplementedError("Only xavier initialization is currently implemented for convolution layers")
      }
    }
    
    val modelMats = new mutable.ArrayBuffer[Mat]
    if (layer.param.getBlobsCount() > 0) {
      if (!convNode.hasBias && layer.param.getBlobsCount() != 1) {
        throw new IllegalArgumentException("Convolution layer without bias needs 1 matrix")
      }
      if (convNode.hasBias && layer.param.getBlobsCount() != 2) {
        throw new IllegalArgumentException("Convolution layer with bias needs 2 matrices")
      }

      // TODO: avoid duplicating code with ConvLayer here
      val shape = layer.param.getBlobs(0).getShape().getDimList().map(_.intValue()).toArray
      val filter = FFilter2Ddn(shape(3), shape(2), shape(1), shape(0), convNode.stride(0), convNode.pad(0))
      // TODO: is this an abstraction barrier violation
      layer.param.getBlobs(0).getDataList().map(_.floatValue()).copyToArray(filter.data)
      modelMats += (if (net.opts.useGPU && Mat.hasCUDA > 0 && Mat.hasCUDNN) {
        val x = GFilter(filter)
        x.convType = Net.CrossCorrelation
        x.setTensorFormat(Net.getCUDNNformat(convNode.tensorFormat, net.opts.tensorFormat));
        x
      } else {
        filter
      })
      
      if (convNode.hasBias) {
        val n = layer.param.getBlobs(1).getShape().getDim(0).toInt
        modelMats += blob2Mat(layer.param.getBlobs(1)).reshapeView(n, 1, 1, 1)
      } else {
        modelMats += null
      }
    }
    
    (Array[Node](convNode), modelMats)
  }
  
  private def translatePooling(layer:CaffeLayer) = {
    val poolingParam = layer.param.getPoolingParam()
    new PoolingNode {
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
  
  private def translateLRN(layer:CaffeLayer) = {
    val lrnParam = layer.param.getLrnParam()
    if (lrnParam.getNormRegion() == NormRegion.WITHIN_CHANNEL) {
      new LRNwithinNode {
        dim = lrnParam.getLocalSize()
        alpha = lrnParam.getAlpha()
        beta = lrnParam.getBeta()
      }
    } else {
      new LRNacrossNode {
        dim = lrnParam.getLocalSize()
        alpha = lrnParam.getAlpha()
        beta = lrnParam.getBeta()
        k = lrnParam.getK()
      }
    }
  }
  
  private def translateInnerProduct(layer:CaffeLayer) = {
    val ipp = layer.param.getInnerProductParam()
    val linNode = new LinNode {
      outdim = ipp.getNumOutput()
      hasBias = ipp.getBiasTerm()
    }
    var modelMats:Seq[Mat] = null
    if (layer.param.getBlobsCount() > 0) {
      if (!linNode.hasBias) {
        if (layer.param.getBlobsCount() != 1) {
          throw new IllegalArgumentException("Linear layer without bias needs 1 matrix")
        }
        modelMats = Array(blob2MatTranspose(layer.param.getBlobs(0)), null)
      } else {
        if (layer.param.getBlobsCount() != 2) {
          throw new IllegalArgumentException("Linear layer without bias needs 2 matrices")
        }
        if (layer.param.getBlobs(0).getShape().getDim(0) != layer.param.getBlobs(1).getShape().getDim(0)) {
          throw new IllegalArgumentException("Weight and bias dimensions for linear layer don't agree")
        }
        if ((layer.param.getBlobs(0).getDoubleDataCount() > 0) != (layer.param.getBlobs(1).getDoubleDataCount() > 0)) {
          throw new IllegalArgumentException("Weight and bias matrices must both be double data or both be single data")
        }
        
        val outDim = layer.param.getBlobs(0).getShape().getDim(0).intValue()
        val weightMat = blob2MatTranspose(layer.param.getBlobs(0))
        val biasMat = blob2MatTranspose(layer.param.getBlobs(1)).reshape(outDim, 1)
        modelMats = Array(weightMat, biasMat)
      }
    }
    (Array[Node](linNode), modelMats)
  }
  
  private def translateBatchNorm(layer:CaffeLayer, scaleParam:Caffe.ScaleParameter) = {
    val bnParam = layer.param.getBatchNormParam()

    val mode = if (layer.inputs(0).param.getType() == "Convolution") {
      BatchNormLayer.Spatial
    } else {
      BatchNormLayer.PerActivation
    }
    
    if (scaleParam ne null) {
      new BatchNormScaleNode {
        epsilon = bnParam.getEps()
        expAvgFactor = bnParam.getMovingAverageFraction()
        batchNormMode = mode
        hasBias = scaleParam.getBiasTerm()
      }
    } else {
      new BatchNormNode {
        epsilon = bnParam.getEps()
        expAvgFactor = bnParam.getMovingAverageFraction()
        batchNormMode = mode
      }
    }
  }
  
  private def addTransformNodes(transformParam:Caffe.TransformationParameter, subjectNode:Node) = {
    val newNodeList = new mutable.ListBuffer[Node]
    newNodeList += subjectNode

    if (transformParam.hasCropSize()) {
      val cropSize = transformParam.getCropSize()
      // TODO: use the correct dimensions
      val sizeMat = 0 \ cropSize \ cropSize \ 0
      // TODO: do I have to worry about offsets
      if (transformParam.getMirror()) {
        newNodeList += new CropMirrorNode {
          inputs(0) = newNodeList.last
          sizes = sizeMat
        }
      } else {
        newNodeList += new CropNode {
          inputs(0) = newNodeList.last
          sizes = sizeMat
        }
      }
    }

    // TODO: implement mean

    if (transformParam.hasScale()) {
      val constNode = new ConstantNode {
        value = transformParam.getScale()
        cache = true // TODO: verify
      }
      val mulNode = newNodeList.last ∘ constNode
      newNodeList += constNode
      newNodeList += mulNode
    }

    newNodeList.toArray
  }
  
  /** Converts the given blob into a Mat. Does not perform any transposition. */
  private def blob2Mat(blob:Caffe.BlobProto):Mat = {
    val dims = blob.getShape().getDimList().map(_.intValue()).toArray
    if (blob.getDoubleDataCount() > 0) {
      // TODO: should I bother with GDMat
      new DMat(dims, blob.getDoubleDataList().map(_.doubleValue()).toArray)
    } else {
      // TODO: should I bother with GFMat
      new FMat(dims, blob.getDataList().map(_.floatValue()).toArray)
    }
  }

  /** Converts the given blob into a Mat, transposing data from row-major order to column-major order. */
  private def blob2MatTranspose(blob:Caffe.BlobProto):Mat = {
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
