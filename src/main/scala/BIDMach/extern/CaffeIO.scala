package BIDMach.extern

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,JSON,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMach._
import BIDMach.networks.layers._
import scala.collection.JavaConversions._
import scala.collection.mutable
import _root_.caffe.Caffe
import _root_.caffe.Caffe.LRNParameter.NormRegion
import com.google.protobuf.TextFormat

object CaffeIO {
  def mkNodeSetFromProtobuf(fin:Readable) = {
    val caffeBuilder = Caffe.NetParameter.newBuilder()
    TextFormat.merge(fin, caffeBuilder)
    
    // Translate every layer and build a mapping of blobs to layers feeding into them
    val nodes = new mutable.ArrayBuffer[Node]
    val nodesWithTop = new mutable.HashMap[String,mutable.Buffer[Node]]
    for (layer <- caffeBuilder.getLayerList()) {
      layer.getType() match {
        case "Convolution" => {
          val convParam = layer.getConvolutionParam()
          nodes += new ConvNode {
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
          }
        }
        case "Pooling" => {
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
          }
        }
        case "LRN" => {
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

        case "Data" => nodes += new InputNode
        case "MemoryData" => nodes += new InputNode
        case "HDF5Data" => nodes += new InputNode

        case "InnerProduct" => {
          val ipp = layer.getInnerProductParam()
          nodes += new LinNode {
            outdim = ipp.getNumOutput()
            hasBias = ipp.getBiasTerm()
          }
        }
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
      for (t <- layer.getTopList()) {
        nodesWithTop.getOrElseUpdate(t, new mutable.ArrayBuffer) += nodes.last
      }
    }
    
    // Assign layer inputs based on which Caffe blobs each layer links to.
    // When several layers reuse the same blob, order the layers in the order they were specified
    // in the prototxt.
    val blobIterIndices = new mutable.HashMap[String,Int]
    for ((layer, curNode) <- caffeBuilder.getLayerList() zip nodes) {
      // XXX this should account for multiple bottom blobs
      if (layer.getBottomCount() >= 1) {
        val bottom = layer.getBottom(0)
        // TODO: check this code further
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
        node.batchNormMode = BatchNormLayer.SPATIAL
      } else {
        node.batchNormMode = BatchNormLayer.PER_ACTIVATION
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

    new NodeSet(newNodes.toArray)
  }
}
