package BIDMach.extern

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,JSON,SMat,SDMat,TMat,FFilter,Filter,GFilter}
import BIDMat.MatFunctions._
import BIDMach._
import BIDMach.datasources.DataSource
import BIDMach.datasources.FileSource
import BIDMach.mixins.L1Regularizer
import BIDMach.models.GLM
import BIDMach.networks.Net
import BIDMach.networks.layers._
import BIDMach.updaters._
import scala.collection.JavaConversions._
import scala.collection.generic.FilterMonadic
import scala.collection.mutable
import scala.language.implicitConversions
import scala.Option
import scala.util.control.Breaks._
import scala.util.Try
import java.io.FileReader
import java.io.InputStream
import java.lang.IllegalArgumentException
import _root_.caffe.Caffe
import _root_.caffe.Caffe.LRNParameter.NormRegion
import _root_.caffe.Caffe.PoolingParameter.PoolMethod
import com.google.protobuf.{CodedInputStream,TextFormat}
import jcuda.jcudnn.cudnnPoolingMode

class CaffeModel private(net:Net, netParam:Caffe.NetParameterOrBuilder, _layers:Seq[CaffeLayer]) {
  import CaffeModel._
  
  private[extern] val layers = _layers

  def predictor(data:Mat, labels:Mat) = {
    val (nn, opts) = Net.predictor(net, data, labels)
    switchLayersToTest(nn.model.asInstanceOf[Net])
    (nn, opts)
  }
  
  def predictor(infn:String, outfn:String):(Learner, Net.FilePredOptions) = {
    predictor(List(FileSource.simpleEnum(infn,1,0)), List(FileSource.simpleEnum(outfn,1,0)));
  }

  def predictor(infn: String, inlb: String, outfn: String): (Learner, Net.FilePredOptions) = {
    predictor(List(FileSource.simpleEnum(infn, 1, 0), FileSource.simpleEnum(inlb, 1, 0)), List(FileSource.simpleEnum(outfn, 1, 0)));
  }

  def predLabels(infn: String, inlb: String): (Learner, Net.FilePredOptions) = {
    predictor(List(FileSource.simpleEnum(infn, 1, 0), FileSource.simpleEnum(inlb, 1, 0)), null);
  }
  
  def predictor(infiles:List[(Int)=>String], outfiles:List[(Int)=>String]):(Learner, Net.FilePredOptions) = {
    val (nn, opts) = Net.predictor(net, infiles, outfiles)
    switchLayersToTest(nn.model.asInstanceOf[Net])
    (nn, opts)
  }
  
  private def switchLayersToTest(newNet:Net) = {
    // It's assumed that model layers between train and test are the same
    val (_, testNodes) = parseProtobuf(netParam, Caffe.Phase.TEST, newNet)
    newNet.opts.nodeset = new NodeSet(testNodes.toArray)
  }
  
  def loadWeights(weightsFile:InputStream) = {
    val cis = CodedInputStream.newInstance(weightsFile)
    cis.setSizeLimit(1 << 30)
    val weightNetParam = Caffe.NetParameter.parseFrom(cis)
    
    // Build a map of names to layers for the weights
    val weightLayerForName = Map(weightNetParam.getLayerList().map(layer => (layer.getName(), layer)):_*)
    val modelMats = new mutable.ArrayBuffer[Mat]
    var i = 0
    while (i < layers.length) {
      val layer = layers(i)
      var incr = 1

      // If layer corresponds to a ModelNode, extract the model mats for this layer
      if (layer.inodeFirst != -1 && net.opts.nodeset(layer.inodeFirst).isInstanceOf[ModelNode]) {
        val weightLayer = weightLayerForName.get(layer.param.getName()) match {
          case Some(wl) => wl
          case None => throw new IllegalArgumentException(s"Layer ${layer.param.getName()} not found in weights file")
        }
        
        layer.param.getType() match {
          case "Convolution" => modelMats ++= getConvLayerMats(weightLayer, net)
          case "BatchNorm" => {
            if (i + 1 < layers.length && layers(i + 1).param.getType() == "Scale") {
              // We are loading data into a BatchNormScaleLayer
              assert(layer.inodeFirst == layers(i + 1).inodeFirst && layer.inodeLast == layers(i + 1).inodeLast)
              incr = 2
              val scaleWeightLayer = weightLayerForName.get(layers(i + 1).param.getName()) match {
                case Some(wl) => wl
                case None => throw new IllegalArgumentException(s"Layer ${layers(i + 1).param.getName()} not found in weights file")
              }
              modelMats ++= getScaleMats(scaleWeightLayer)
              // Theoretically this line should be outside the if statement, but at present the BatchNorm layer isn't a ModelLayer.
              modelMats ++= getBatchNormMats(weightLayer)
            }
          }
          case "Scale" => modelMats ++= getScaleMats(weightLayer)
          case "InnerProduct" => modelMats ++= getInnerProductLayerMats(weightLayer)
          case _ =>
        }
      }
      
      i += incr
    }
    net.setmodelmats(modelMats.toArray)
    net.opts.nmodelmats = modelMats.length
    net.refresh = false
  }
}

object CaffeModel {
  def loadFromSolver(solverFile:Readable, opts:Learner.Opts with Grad.Opts, net:Net, means:Mat = null,
                     l1regopts:L1Regularizer.Opts = null) = {
    val caffeBuilder = Caffe.SolverParameter.newBuilder()
    TextFormat.merge(solverFile, caffeBuilder)
    
    require(caffeBuilder.hasNet() || caffeBuilder.hasNetParam(), "A solver file must specify a net or inline net param")
    val caffeModel = if (caffeBuilder.hasNet()) {
      loadModel(new FileReader(caffeBuilder.getNet()), net, means)
    } else {
      val (layers, nodes) = parseProtobuf(caffeBuilder.getNetParam(), Caffe.Phase.TRAIN, net, means)
      net.opts.nodeset = new NodeSet(nodes.toArray)
  
      new CaffeModel(net, caffeBuilder.getNetParam(), layers)
    }
    // TODO: implement train_net, test_net, train_net_param, test_net_param
    
    require(caffeBuilder.getTestInitialization(), "test_initialization = false is not supported")
    
    val maxIter = caffeBuilder.getMaxIter()
    opts.pstep = caffeBuilder.getDisplay().asInstanceOf[Float] / maxIter
    // TODO: set opts.npasses from maxIter
    
    val baseLr = caffeBuilder.getBaseLr()
    val gamma = caffeBuilder.getGamma()
    opts.lrate = baseLr
    caffeBuilder.getLrPolicy() match {
      case "fixed" | "" => {}
      case "step" => {
        val stepSize = caffeBuilder.getStepsize()
        opts.lr_policy =
          (ipass:Float, istep:Float, prog:Float) => (baseLr * Math.pow(gamma, Math.floor(istep / stepSize))).asInstanceOf[Float]
      }
      case "exp" => {
        opts.lr_policy =
          (ipass:Float, istep:Float, prog:Float) => (baseLr * Math.pow(gamma, istep)).asInstanceOf[Float]
      }
      case "inv" => {
        val power = caffeBuilder.getPower()
        opts.lr_policy =
          (ipass:Float, istep:Float, prog:Float) => (baseLr * Math.pow(1 + gamma * istep, -power)).asInstanceOf[Float]
      }
      case "multistep" => {
        var currentStep = 0
        val stepValues = caffeBuilder.getStepvalueList()
        opts.lr_policy = (ipass:Float, istep:Float, prog:Float) => {
          if (currentStep < stepValues.size() && istep >= stepValues.get(currentStep)) {
            currentStep += 1
          }
          (baseLr * Math.pow(gamma, currentStep)).asInstanceOf[Float]
        }
      }
      case "poly" => {
        val power = caffeBuilder.getPower()
        opts.lr_policy =
          (ipass:Float, istep:Float, prog:Float) => (baseLr * Math.pow(1 - istep/maxIter, power)).asInstanceOf[Float]
      }
      case "sigmoid" => {
        val stepSize = caffeBuilder.getStepsize()
        opts.lr_policy = (ipass:Float, istep:Float, prog:Float) => {
          (baseLr * (1 / (1 + Math.exp(-gamma * (istep - stepSize))))).asInstanceOf[Float]
        }
      }
    }
    
    caffeBuilder.getRegularizationType() match {
      case "L1" => {
        require(l1regopts ne null, "L1 regularization opts must be given to use L1 regularization")
        l1regopts.reg1weight = caffeBuilder.getWeightDecay()
      }
      case "L2" => if (caffeBuilder.hasWeightDecay()) opts.l2reg = caffeBuilder.getWeightDecay()
    }
    if (caffeBuilder.hasClipGradients()) opts.max_grad_norm = caffeBuilder.getClipGradients()
    
    net.opts.useGPU = (caffeBuilder.getSolverMode() == Caffe.SolverParameter.SolverMode.GPU)
    
    opts.texp = null
    val adaOptsOption = Try(opts.asInstanceOf[ADAGrad.Opts]).toOption
    caffeBuilder.getType() match {
      case "SGD" => {
        opts.vel_decay = caffeBuilder.getMomentum()
        opts.waitsteps = 0
        for (adaOpt <- adaOptsOption) {
          // Don't do AdaGrad etc. stuff
          adaOpt.vexp = 0f
        }
      }
      case "Nesterov" => {
        opts.nesterov_vel_decay = caffeBuilder.getMomentum()
        opts.waitsteps = 0
        for (adaOpt <- adaOptsOption) {
          // Don't do AdaGrad etc. stuff
          adaOpt.vexp = 0f
        }
      }
      case "AdaGrad" => {
        require(adaOptsOption.nonEmpty, "AdaGrad solver type requires a Learner.Opts of type ADAGrad.Opts")
        val adaOpt = adaOptsOption.get
        opts.texp = 0.5f
        if (caffeBuilder.hasDelta()) adaOpt.epsilon = caffeBuilder.getDelta()
      }
      case "RMSProp" => {
        require(adaOptsOption.nonEmpty, "RMSProp solver type requires a Learner.Opts of type ADAGrad.Opts")
        val adaOpt = adaOptsOption.get
        if (caffeBuilder.hasDelta()) adaOpt.epsilon = caffeBuilder.getDelta()
        adaOpt.gsq_decay = caffeBuilder.getRmsDecay()
      }
      case "AdaDelta" => {
        require(adaOptsOption.nonEmpty, "AdaDelta solver type requires a Learner.Opts of type ADAGrad.Opts")
        val adaOpt = adaOptsOption.get
        if (caffeBuilder.hasDelta()) adaOpt.epsilon = caffeBuilder.getDelta()
        adaOpt.gsq_decay = caffeBuilder.getMomentum()
        Mat.consoleLogger.warning("AdaDelta: RMS(delta_x) is not currently implemented")
      }
      case "Adam" => throw new NotImplementedError("Adam is not implemented yet")
    }
    
    net.opts.debug = if (caffeBuilder.getDebugInfo()) 1 else 0
    
    caffeModel
  }
  
  def loadModel(modelFile:Readable, net:Net, means:Mat = null) = {
    val caffeBuilder = Caffe.NetParameter.newBuilder()
    TextFormat.merge(modelFile, caffeBuilder)

    val (layers, nodes) = parseProtobuf(caffeBuilder, Caffe.Phase.TRAIN, net, means)
    net.opts.nodeset = new NodeSet(nodes.toArray)

    new CaffeModel(net, caffeBuilder, layers)
  }

  private def parseProtobuf(netParam:Caffe.NetParameterOrBuilder, phase:Caffe.Phase, net:Net, means:Mat = null) = {
    // Caffe only supports CrossCorrelation convolution
    net.opts.convType = Net.CrossCorrelation
    // The Caffe tensor format is NCHW
    net.opts.tensorFormat = Net.TensorNCHW
    
    val layersForPhase = filterLayers(netParam.getLayerList(), phase)
    val layers = toposort(resolveLayerLinks(layersForPhase))

    // Translate every layer and build a mapping of blobs to layers feeding into them
    val nodes = new mutable.ArrayBuffer[Node]
    // SoftmaxOutputNode for categorical classification
    var softmaxOutputNode:SoftmaxOutputNode = null
    var hasAccuracy = false
    implicit def singleNode(node:Node):Array[Node] = Array(node)
    var i = 0
    while (i < layers.length) {
      val layer = layers(i)
      var incr = 1

      // Translate layer according to its layer type
      val newNodes:Array[Node] = layer.param.getType() match {
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

        case "Concat" => {
          val concatParam = layer.param.getConcatParam()
          if (concatParam.getAxis() != 1) {
            throw new NotImplementedError("Only axis=1 for Concat layers is currently supported")
          }
          new StackNode(layer.param.getBottomCount())
        }

        case "MultinomialLogisticLoss" => {
          softmaxOutputNode = new SoftmaxOutputNode { lossType = SoftmaxOutputLayer.CaffeMultinomialLogisticLoss }
          softmaxOutputNode
        }
        case "SoftmaxWithLoss" => {
          softmaxOutputNode = new SoftmaxOutputNode
          softmaxOutputNode
        }
        case "EuclideanLoss" => new GLMNode { links = GLM.linear }
        case "HingeLoss" => {
          if (layer.param.getHingeLossParam().getNorm() != Caffe.HingeLossParameter.Norm.L1) {
            throw new UnsupportedOperationException("Only L1 loss is supported")
          }
          new GLMNode { links = GLM.svm }
        }
        case "Accuracy" => {
          hasAccuracy = true
          Array()
        }

        case "ReLU" => new RectNode
        case "Sigmoid" => new SigmoidNode
        case "TanH" => new TanhNode
        case "BNLL" => new SoftplusNode

        case "Data" => {
          val dataParam = layer.param.getDataParam()
          
          if (net.opts.isInstanceOf[DataSource.Opts]) {
            net.opts.asInstanceOf[DataSource.Opts].batchSize = dataParam.getBatchSize()
          }
          
          addTransformNodes(layer.param, means, new InputNode)
        }
        case "MemoryData" => new InputNode
        case "HDF5Data" => new InputNode

        case "InnerProduct" => {
          val ipp = layer.param.getInnerProductParam()
          new LinNode {
            outdim = ipp.getNumOutput()
            hasBias = ipp.getBiasTerm()
            fillWeightInitOpts(this, ipp.getWeightFiller(), if (hasBias) ipp.getBiasFiller() else null)
          }
        }
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
        
        case "Eltwise" => translateEltwise(layer)
        
        // TODO: change once we implement all layer types
        case unknownType => throw new NotImplementedError("\"%s\" is not implemented yet" format unknownType)
      }
      
      // Set lr scaling and model mat for ModelNodes
      newNodes match {
        case Array(modelNode:ModelNode) => {
          if (layer.param.getParamCount() >= 1) {
            modelNode.lr_scale = layer.param.getParam(0).getLrMult()
            if (layer.param.getParamCount() >= 2) {
              modelNode.bias_scale = layer.param.getParam(1).getLrMult()
            }
          }
          if (layer.param.getParamList().filter(_.hasDecayMult()).nonEmpty) {
            Mat.consoleLogger.warning("The decay_mult option is not implemented")
          }
        }
        case _ =>
      }
      
      // Set accuracy score option if there was one
      if (hasAccuracy && (softmaxOutputNode ne null)) {
        softmaxOutputNode.scoreType = SoftmaxOutputLayer.AccuracyScore
      }
      
      layer.inodeFirst = nodes.length
      layer.inodeLast = nodes.length + newNodes.length - 1

      if (!newNodes.isEmpty) {
        if (newNodes(0).isInstanceOf[OutputNode]) {
          // In Caffe, output layers have two inputs (the penultimate layer and the data layer),
          // while in BIDMach they only have one.
          newNodes(0).inputs(0) = nodes(layer.inputs(0).inodeLast)
        } else {
          for ((input, i) <- layer.inputs.zipWithIndex) {
            newNodes(0).inputs(i) = nodes(input.inodeLast)
          }
        }
      }
      
      nodes ++= newNodes
      i += incr
    }

    (layers, nodes)
  }
  
  /** Filters out layers in the given {@code LayerParameter} sequence to only contain ones
   *  wanted for the given {@code phase}.
   */
  private def filterLayers(layerList:Seq[Caffe.LayerParameter], phase:Caffe.Phase) = {
    layerList.withFilter(layer => {
      check(!(layer.getIncludeCount() > 0 && layer.getExcludeCount() > 0), layer,
            "only include rules xor exclude rules can be specified")
      if (layer.getIncludeCount() > 0) {
        layer.getIncludeList().exists(netStateRule => stateMatchesRule(netStateRule, phase))
      } else {
        !layer.getExcludeList().exists(netStateRule => stateMatchesRule(netStateRule, phase))
      }
    })
  }
  
  private def stateMatchesRule(netStateRule:Caffe.NetStateRule, phase:Caffe.Phase) = {
    var matches = true
    for (fieldDesc <- netStateRule.getAllFields().keys) {
      fieldDesc.getName() match {
        case "phase" => matches &= (netStateRule.getPhase() == phase)
        case _ => println(s"Warning: net state rule ${fieldDesc.getName()} is not implemented")
      }
    }
    matches
  }
  
  /** Creates a list of {@code Layer} objects, setting their {@code lowers} and {@code uppers}
   *  attributes based on their links to blobs in the protobuf.
   */
  private def resolveLayerLinks(layers:FilterMonadic[Caffe.LayerParameter, Seq[Caffe.LayerParameter]]):Seq[CaffeLayer] = {
    val layerBuf = new mutable.ArrayBuffer[CaffeLayer]
    for (layerParam <- layers) {
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
      var inplaceUsed = false
      for (bottom <- layer.param.getBottomList()) {
        if (layer.param.getTopList().contains(bottom)) {
          check(!inplaceUsed, layer.param, "A layer can only have at most one in-place input")
          inplaceUsed = true
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
          check(!previsited.contains(layer), layer.param, "Cycle detected")
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
      
      fillWeightInitOpts(this, convParam.getWeightFiller(), if (hasBias) convParam.getBiasFiller() else null)
    }

    Array[Node](convNode)
  }
  
  private def getConvLayerMats(layerParam:Caffe.LayerParameter, net:Net) = {
    val convParam = layerParam.getConvolutionParam()
    val modelMats = new mutable.ArrayBuffer[Mat]

    // TODO: avoid duplication with translateConvolution
    val hasBias = convParam.getBiasTerm()
    val stride0 = if (convParam.hasStrideW()) {
      convParam.getStrideW() 
    } else if (convParam.getStrideCount() == 0) {
      1
    } else {
      convParam.getStride(0)
    }
    val pad0 = if (convParam.hasPadW()) {
      convParam.getPadW() 
    } else if (convParam.getPadCount() == 0) {
      0
    } else {
      convParam.getPad(0)
    }
    
    if (!hasBias) {
      check(layerParam.getBlobsCount() == 1, layerParam, "convolution layer without bias needs 1 matrix")
    } else {
      check(layerParam.getBlobsCount() == 2, layerParam, "convolution layer with bias needs 2 matrices")
    }

    // TODO: avoid duplicating code with ConvLayer here
    val shape = layerParam.getBlobs(0).getShape().getDimList().map(_.intValue()).toArray
    val filter = FFilter2Ddn(shape(3), shape(2), shape(1), shape(0), stride0, pad0)
    // TODO: is this an abstraction barrier violation
    layerParam.getBlobs(0).getDataList().map(_.floatValue()).copyToArray(filter.data)
    modelMats += (if (net.opts.useGPU && Mat.hasCUDA > 0 && Mat.hasCUDNN) {
      val x = GFilter(filter)
      x.convType = Net.CrossCorrelation
      x.setTensorFormat(jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW);
      x
    } else {
      filter
    })
    
    if (hasBias) {
      val n = layerParam.getBlobs(1).getShape().getDim(0).toInt
      modelMats += blob2Mat(layerParam.getBlobs(1)).reshapeView(n, 1, 1, 1)
    } else {
      modelMats += null
    }

    modelMats
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
  
  private def getInnerProductLayerMats(layerParam:Caffe.LayerParameter) = {
    if (!layerParam.getInnerProductParam().getBiasTerm()) {
      check(layerParam.getBlobsCount() == 1, layerParam, "linear layer without bias needs 1 matrix")
      Array(blob2MatTranspose(layerParam.getBlobs(0)), null)
    } else {
      check(layerParam.getBlobsCount() == 2, layerParam, "linear layer without bias needs 2 matrices")
      check(layerParam.getBlobs(0).getShape().getDim(0) == layerParam.getBlobs(1).getShape().getDim(0),
            layerParam, "weight and bias dimensions for linear layer don't agree")
      
      val outDim = layerParam.getBlobs(0).getShape().getDim(0).intValue()
      val weightMat = blob2MatTranspose(layerParam.getBlobs(0))
      val biasMat = blob2MatTranspose(layerParam.getBlobs(1)).reshapeView(outDim, 1)
      Array(weightMat, biasMat)
    }
  }
  
  private def translateBatchNorm(layer:CaffeLayer, scaleParam:Caffe.ScaleParameter) = {
    val bnParam = layer.param.getBatchNormParam()
    
    if (scaleParam ne null) {
      new BatchNormScaleNode {
        epsilon = bnParam.getEps()
        expAvgFactor = bnParam.getMovingAverageFraction()
        // It appears that Caffe always uses Spatial activations
        batchNormMode = BatchNormLayer.Spatial
        hasBias = scaleParam.getBiasTerm()
      }
    } else {
      new BatchNormNode {
        epsilon = bnParam.getEps()
        expAvgFactor = bnParam.getMovingAverageFraction()
        batchNormMode = BatchNormLayer.Spatial
      }
    }
  }
  
  private def translateEltwise(layer:CaffeLayer) = {
    val eltwiseParam = layer.param.getEltwiseParam()
    check(eltwiseParam.getOperation() == Caffe.EltwiseParameter.EltwiseOp.SUM || eltwiseParam.getCoeffCount() == 0,
        layer.param, "Eltwise layer only takes coefficients for summation.")
    eltwiseParam.getOperation() match {
      case Caffe.EltwiseParameter.EltwiseOp.SUM => {
        if (eltwiseParam.getCoeffCount() == 0) {
          new AddNode(layer.param.getBottomCount())
        } else {
          check(eltwiseParam.getCoeffCount() == layer.param.getBottomCount(),
              layer.param, "Eltwise Layer takes one coefficient per bottom blob.")
          val coefNodes = for (c <- eltwiseParam.getCoeffList()) yield {
            new ConstantNode { value = c.floatValue(); cache = true }
          }
          val addNode = new AddNode(layer.param.getBottomCount())
          coefNodes.copyToArray(addNode.inputs)
          new CompoundNode {
            grid = NodeMat(coefNodes) \ addNode
          }
        }
      }
      case Caffe.EltwiseParameter.EltwiseOp.PROD => new MulNode(layer.param.getBottomCount())
      case Caffe.EltwiseParameter.EltwiseOp.MAX => new MaxNode(layer.param.getBottomCount())
    }
  }
  
  private def getBatchNormMats(layerParam:Caffe.LayerParameter) = {
    check(layerParam.getBlobsCount() == 3, layerParam, "batch norm needs 2 matrices and scale factor")
    check(layerParam.getBlobs(2).getDataCount() > 0, layerParam, "batch norm layer doesn't have a scale factor")

    val c = layerParam.getBlobs(0).getShape().getDim(0).toInt
    check(c == layerParam.getBlobs(1).getShape().getDim(0).toInt, layerParam, "batch norm matrices aren't the same shape")
    val scale = {
      val rawScale = layerParam.getBlobs(2).getData(0)
      if (rawScale == 0) 0f else 1f / rawScale
    }
    val runningMeans = blob2Mat(layerParam.getBlobs(0)).reshapeView(c, 1, 1, 1)
    runningMeans ~ runningMeans * scale
    val runningVariances = blob2Mat(layerParam.getBlobs(1)).reshapeView(c, 1, 1, 1)
    runningVariances ~ runningVariances * scale
    Array(runningMeans, runningVariances)
  }
  
  private def getScaleMats(layerParam:Caffe.LayerParameter) = {
    val hasBias = layerParam.getScaleParam().hasBiasTerm()
    
    if (hasBias) {
      check(layerParam.getBlobsCount() == 2, layerParam, "scale layer with bias needs 2 matrices")
    } else {
      check(layerParam.getBlobsCount() == 1, layerParam, "scale layer without bias needs 1 matrix")
    }
    
    val c = layerParam.getBlobs(0).getShape().getDim(0).toInt
    val scaleMat = blob2Mat(layerParam.getBlobs(0)).reshapeView(c, 1, 1, 1)
    val biasMat = if (hasBias) {
      check(layerParam.getBlobs(1).getShape().getDim(0).toInt == c, layerParam, "scale layer matrices aren't the same shape")
      blob2Mat(layerParam.getBlobs(0)).reshapeView(c, 1, 1, 1)
    } else {
      zeros(c \ 1 \ 1 \ 1)
    }
    Array(scaleMat, biasMat)
  }
  
  private def addTransformNodes(layerParam:Caffe.LayerParameter, means:Mat, subjectNode:Node) = {
    val transformParam = layerParam.getTransformParam()
    val newNodeList = new mutable.ListBuffer[Node]
    newNodeList += subjectNode

    check(!(transformParam.hasMeanFile() && transformParam.getMeanValueCount() > 0),
          layerParam, "you cannot specify both a mean value file and mean value counts")
    if (transformParam.getMeanValueCount() > 0) {
      val numMeanValues = transformParam.getMeanValueCount()
      val meanValues:FMat = if (numMeanValues == 1) {
    	// One global mean value to subtract from every datum
        transformParam.getMeanValue(0)
      } else {
        // Each mean value applies to a channel. Since data matrices are written in CWHN order,
        // we create a Cx1x1x1 mean value matrix that we subtract from the input.
        new FMat(Array(numMeanValues, 1, 1, 1), transformParam.getMeanValueList().map(_.floatValue()).toArray)
      }
      val constNode = new ConstantNode {
        value = meanValues
        cache = true // TODO: verify
      }
      val subNode = newNodeList.last - constNode
      newNodeList += constNode
      newNodeList += subNode
    } else if (transformParam.hasMeanFile()) {
      check(means ne null, layerParam, "need to specify means")
      val constNode = new ConstantNode {
        value = means
        cache = true // TODO: verify
      }
      val subNode = newNodeList.last - constNode
      newNodeList += constNode
      newNodeList += subNode
    }

    if (transformParam.hasScale()) {
      val constNode = new ConstantNode {
        value = transformParam.getScale()
        cache = true // TODO: verify
      }
      val mulNode = newNodeList.last ∘ constNode
      newNodeList += constNode
      newNodeList += mulNode
    }

    if (transformParam.hasCropSize()) {
      val cropSize = transformParam.getCropSize()
      // XXX: don't hardcode the # of channels
      val sizeMat = 3 \ cropSize \ cropSize \ 0
      if (transformParam.getMirror()) {
        newNodeList += new CropMirrorNode {
          inputs(0) = newNodeList.last
          sizes = sizeMat
          offsets = 0 \ -1 \ -1 \ 0
          // We want the height and width dimensions of randoffsets to be equal to the size of the
          // image. Since we don't have access to that here, we will set the height and width
          // dimensions to some number that will get capped to the image height and width.
          randoffsets = 0 \ Int.MaxValue \ Int.MaxValue \ 0
        }
      } else {
        newNodeList += new CropNode {
          inputs(0) = newNodeList.last
          sizes = sizeMat
          offsets = 0 \ -1 \ -1 \ 0
          randoffsets = 0 \ Int.MaxValue \ Int.MaxValue \ 0
        }
      }
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
  
  private def fillWeightInitOpts(weightInitOpts:WeightInitOpts,
                                 weightFillerParam:Caffe.FillerParameter,
                                 biasFillerParam:Caffe.FillerParameter = null) = {
    def translate(fp:Caffe.FillerParameter) = {
      fp.getType() match {
        case "xavier" => {
          if (fp.getVarianceNorm() != Caffe.FillerParameter.VarianceNorm.FAN_IN) {
            Mat.consoleLogger.warning(s"Xavier initialization option ${fp.getVarianceNorm()} is not implemented")
          }
          (Net.xavier, 1f)
        }
        case "gaussian" => {
          if (fp.getMean() != 0f) {
            throw new NotImplementedError("Gaussian initialization with non-zero mean isn't supported")
          }
          (Net.gaussian, fp.getStd())
        }
        case "constant" => (Net.constant, fp.getValue())
        case _ => throw new NotImplementedError(s"Initialization method '${fp.getType()}' is not implemented")
      }
    }
    
    val (initfn, initv) = translate(weightFillerParam)
    weightInitOpts.initfn = initfn
    weightInitOpts.initv = initv
    if (biasFillerParam ne null) {
      val (initbiasfn, initbiasv) = translate(biasFillerParam)
      weightInitOpts.initbiasfn = initbiasfn
      weightInitOpts.initbiasv = initbiasv
    }
  }
  
  private def check(requirement:Boolean, layerParam:Caffe.LayerParameter, message: => Any) = {
    require(requirement, s"Layer ${layerParam.getName()}: ${message}")
  }
}

private class CaffeLayer(val param:Caffe.LayerParameter) {
  val inputs = new mutable.ArrayBuffer[CaffeLayer]
  var inodeFirst = -1
  var inodeLast = -1
  
  override def toString() = {
    val strInputs = "[" + inputs.map(_.param.getName()).mkString(", ") + "]"
    s"CaffeLayer(param.name=${param.getName()}, inputs=${strInputs}, inodeFirst=${inodeFirst}, inodeLast=${inodeLast})"
  }
}
