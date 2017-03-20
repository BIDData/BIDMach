package BIDMach.extern

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,JSON,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach._
import BIDMach.networks.layers._
import java.util.Arrays
import scala.collection.JavaConversions._
import scala.collection.mutable
import _root_.caffe.Caffe
import _root_.caffe.Caffe.LRNParameter.NormRegion
import com.google.protobuf.TextFormat

object CaffeIO {
  def mkNodeSetFromProtobuf(fin:Readable) = {
    val caffeBuilder = Caffe.NetParameter.newBuilder()
    TextFormat.merge(fin, caffeBuilder)
    
    val nodeSet = new NodeSet(caffeBuilder.getLayerCount())
    val nodesWithTop = new mutable.HashMap[String,mutable.Buffer[Node]]
    for (i <- 0 until caffeBuilder.getLayerCount()) {
      val layer = caffeBuilder.getLayer(i)
      layer.getType() match {
        case "Convolution" => {
          val convParam = layer.getConvolutionParam()
          // TODO: handle null
          nodeSet(i) = new ConvNode {
            noutputs = convParam.getNumOutput()
            hasBias = convParam.getBiasTerm()
            if (convParam.hasPadW()) {
              pad = convParam.getPadW() \ convParam.getPadH()
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
            } else {
              stride = irow(convParam.getStrideList().map(_.intValue()).toList)
            }
            dilation = irow(convParam.getDilationList().map(_.intValue()).toList)
          }
        }
        case "Pooling" => {
          val poolingParam = layer.getPoolingParam()
          // TODO: handle null
          nodeSet(i) = new PoolingNode {
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
          // TODO: handle null
          if (lrnParam.getNormRegion() == NormRegion.WITHIN_CHANNEL) {
            nodeSet(i) = new LRNwithinNode {
              dim = lrnParam.getLocalSize()
              alpha = lrnParam.getAlpha()
              beta = lrnParam.getBeta()
            }
          } else {
            nodeSet(i) = new LRNacrossNode {
              dim = lrnParam.getLocalSize()
              alpha = lrnParam.getAlpha()
              beta = lrnParam.getBeta()
              k = lrnParam.getK()
            }
          }
        }
        case "BatchNorm" => {
          val bnParam = layer.getBatchNormParam()
          // TODO: handle null
          nodeSet(i) = new BatchNormNode {
            epsilon = bnParam.getEps()
          }
        }

        case "SoftmaxWithLoss" => nodeSet(i) = new SoftmaxOutputNode
        case "HingeLoss" => nodeSet(i) = new GLMNode { links = irow(3) }
        case "Accuracy" => nodeSet(i) = new AccuracyNode

        case "ReLU" => nodeSet(i) = new RectNode
        case "Sigmoid" => nodeSet(i) = new SigmoidNode
        case "TanH" => nodeSet(i) = new TanhNode
        case "BNLL" => nodeSet(i) = new SoftplusNode

        case "Data" => nodeSet(i) = new InputNode
        case "MemoryData" => nodeSet(i) = new InputNode
        case "HDF5Data" => nodeSet(i) = new InputNode

        case "InnerProduct" => {
          val ipp = layer.getInnerProductParam()
          // TODO: handle ipp null case
          nodeSet(i) = new LinNode { outdim = ipp.getNumOutput(); hasBias = ipp.getBiasTerm() }
        }
        case "Split" => nodeSet(i) = new CopyNode
        case "Softmax" => nodeSet(i) = new SoftmaxNode
        case "Dropout" => {
          val dropoutParam = layer.getDropoutParam()
          // TODO: handle null
          nodeSet(i) = new DropoutNode { frac = dropoutParam.getDropoutRatio() }
        }
        // TODO: implement base, shift, scale for the following two
        case "Exp" => nodeSet(i) = new ExpNode
        case "Log" => nodeSet(i) = new LnNode
        case "Scale" => {
          val scaleParam = layer.getScaleParam()
          nodeSet(i) = new ScaleNode { hasBias = scaleParam.getBiasTerm() }
        }
      }
      for (t <- caffeBuilder.getLayer(i).getTopList()) {
        nodesWithTop.getOrElseUpdate(t, new mutable.ArrayBuffer).append(nodeSet(i))
      }
    }
    
    val blobIterIndices = new mutable.HashMap[String,Int]
    for (i <- 0 until caffeBuilder.getLayerCount()) {
      val layer = caffeBuilder.getLayer(i)
      // XXX this should account for multiple bottom blobs
      if (layer.getBottomCount() >= 1) {
        val bottom = layer.getBottom(0)
        // TODO: check this code further
        if (layer.getTopList().contains(bottom)) {
          val j = blobIterIndices.getOrElse(bottom, 0)
          nodeSet(i).inputs(0) = nodesWithTop(bottom)(j)
          blobIterIndices(bottom) = j + 1
        } else {
          nodeSet(i).inputs(0) = nodesWithTop(bottom)(nodesWithTop(bottom).length - 1)
        }
      }
    }
    
    nodeSet
  }
}
