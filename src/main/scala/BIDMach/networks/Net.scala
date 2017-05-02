package BIDMach.networks

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GFilter,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,JSON,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.datasinks._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach._
import BIDMach.networks.layers._
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import scala.util.hashing.MurmurHash3;
import java.util.HashMap;

/**
 * Basic Net class. Learns a supervised map from input blocks to output (target) data blocks. 
 *
 * The network topology is specified by opts.layers which is a sequence of "NodeSet" objects. There is a NodeSet
 * Class for each Layer class, which holds the params for defining that layer. There is also an inputs parameter which points
 * to the set of Node instances that mirror the final network structure. 
 *
 */

class Net(override val opts:Net.Opts = new Net.Options) extends Model(opts) {
  var layers:Array[Layer] = null;
  var layermat:LayerMat = null;
  var output_layers:Array[Layer] = null;
  var targmap:Mat = null;
  var mask:Mat = null;
  var bufmat:Mat = null;
  var modelMap:HashMap[String,Int] = null;
  var batchSize = -1;
  var imodel = 0;
  var initialize = false;

  override def init() = {
//	  mats = datasource.next;
	  var nfeats = mats(0).nrows;
	  batchSize = mats(0).ncols
	  targmap = if (opts.targmap.asInstanceOf[AnyRef] != null) convertMat(opts.targmap) else null;
	  mask = if (opts.dmask.asInstanceOf[AnyRef] != null) convertMat(opts.dmask) else null;
	  createLayers;
    if (output_layers == null) {
      output_layers = layers(layers.length-1) match {
        case a:OutputLayer => Array(layers(layers.length-1));
        case _ => Array[Layer]();
      }
    }
	  if (modelMap == null) {
	  	modelMap = new HashMap[String,Int];
	  }
	  imodel = 0;
	  layers.map((x:Layer) => if (x != null)x.getModelMats(this));
	  if (refresh) {
	  	setmodelmats(new Array[Mat](imodel + modelMap.size));
	  }
	  if (updatemats == null) updatemats = new Array[Mat](modelmats.length);
	  for (i <- 0 until modelmats.length) {
	  	if (modelmats(i).asInstanceOf[AnyRef] != null) modelmats(i) = convertMat(modelmats(i));
	  	if (updatemats(i).asInstanceOf[AnyRef] != null) {
        updatemats(i) = convertMat(updatemats(i));
        updatemats(i).clear;
	  	}
	  };
	  copyMats(mats, gmats);
	  val pb = putBack;
	  putBack = -1;
    initialize = true;
    evalbatch(gmats, 0, 0);
    initialize = false;
    putBack = pb;
//	  datasource.reset;
  }
  
  def createLayers = {
    if (opts.nodeset.asInstanceOf[AnyRef] != null) {   // create the layers from the nodeset
    	val nodes = opts.nodeset.nodes;
    	layers = new Array[Layer](opts.nodeset.nnodes);
    	for (i <- 0 until opts.nodeset.nnodes) {
    		layers(i) = nodes(i).create(this);
    		nodes(i).myLayer = layers(i);
    	}
    	for (i <- 0 until opts.nodeset.nnodes) {
    		for (j <- 0 until nodes(i).inputs.length) {
    			if (nodes(i).inputs(j) != null) {
    				val nodeTerm = nodes(i).inputs(j);
    				layers(i).setInput(j, new LayerTerm(nodeTerm.node.myLayer, nodeTerm.term));
    			}
    		}
    	}
    } else {                                           // create a LayerMat that mirrors the NodeMat
      val nrows = opts.nodemat.nrows;
      val ncols = opts.nodemat.ncols;
      layermat = LayerMat(nrows, ncols);
      var nnodes = opts.nodemat.data.map((x:Node) => if (x.asInstanceOf[AnyRef] == null) 0 else 1).reduce(_+_); // count non-null nodes
      layers = new Array[Layer](nnodes);
      var nnlayers = 0;
      for (i <- 0 until ncols) {
        for (j <- 0 until nrows) {
          val node = opts.nodemat(j, i);
          if (node.asInstanceOf[AnyRef] != null) {
            layermat(j, i) = node.create(this);
            node.myLayer = layermat(j, i);
            layers(nnlayers) = layermat(j, i);
            nnlayers += 1;
          }
        }
    	}
      for (i <- 0 until ncols) {
        for (j <- 0 until nrows) {
          val node = opts.nodemat(j, i);
          if (node.asInstanceOf[AnyRef] != null) {
          	for (k <- 0 until node.inputs.length) {
          		if (node.inputs(k) != null) {
          			val nodeTerm = node.inputs(k);
          			layermat(j, i).setInput(k, new LayerTerm(nodeTerm.node.myLayer, nodeTerm.term));
          		}
          	}
          }
        }
      }
    }
  }
  
  def assignInputs(gmats:Array[Mat], ipass:Int, pos:Long) {
    for (i <- 0 until (gmats.length - output_layers.length)) {
    	layers(i).output = gmats(i);
    }
  }
  
  def assignTargets(gmats:Array[Mat], ipass:Int, pos:Long) {
  	if (targmap.asInstanceOf[AnyRef] != null) {
  		layers(layers.length-1).target = targmap * gmats(0);
  	} else if (gmats.length > 1 && output_layers.length > 0) {
  	  for (i <- 0 until output_layers.length)
  	  	output_layers(i).target = full(gmats(gmats.length-output_layers.length+i));
  	}
  }
  
  def forward = {
		  if (mask.asInstanceOf[AnyRef] != null) {
			  modelmats(0) ~ modelmats(0) ∘ mask;
		  }
		  var i = 0;
		  while (i < layers.length) {
			  if (opts.debug > 0) {
				  println("dobatch forward %d %s" format (i, layers(i).getClass))
			  }
			  layers(i).forward;
			  i += 1;
		  }
  }
  
  def cleargrad {
  	if (opts.aopts == null) {
  		for (j <- 0 until updatemats.length) updatemats(j).clear;
  	}
  }
  
  /** 
   *  Set the derivative of the output layer to 1's. Assumes we are maximizing likelihood. 
   *  If todo > 0, put ones in the first todo columns and zeros elsewhere. This is to deal 
   *  with incomplete minibatches. 
   */
  
  def setderiv(todo:Int=0) = {
    val dolayers = if (output_layers.length > 0) output_layers else Array(layers(layers.length-1));
    var j = 0;
    while (j < dolayers.length) {
    	val deriv = dolayers(j).deriv;
      if (todo == 0) {
      	deriv.set(1);
      } else {
        deriv <-- (ones(deriv.nrows, todo) \ zeros(deriv.nrows, deriv.ncols - todo));
      }
    	j += 1;
    }    
  }
  
  def backward(ipass:Int = 0, pos:Long = 0) {
    var i = layers.length;
    while (i > 1) {
    	i -= 1;
    	if (opts.debug > 0) {
    		println("dobatch backward %d %s" format (i, layers(i).getClass))
    	}
    	layers(i).backward(ipass, pos);
    }
    if (mask.asInstanceOf[AnyRef] != null) {
    	updatemats(0) ~ updatemats(0) ∘ mask;
    }
  }
  
  
  def dobatch(gmats:Array[Mat], ipass:Int, pos:Long):Unit = {
    if (batchSize < 0) batchSize = gmats(0).ncols;
    if (batchSize == gmats(0).ncols) {                                    // discard odd-sized minibatches
    	assignInputs(gmats, ipass, pos);
    	assignTargets(gmats, ipass, pos);
    	forward;
    	cleargrad;
    	setderiv();
    	backward(ipass, pos);
    }
  }
  
  def evalbatch(mats:Array[Mat], ipass:Int, pos:Long):FMat = {  
  	if (batchSize < 0) batchSize = gmats(0).ncols;
  	if (batchSize == gmats(0).ncols) { 
  		assignInputs(gmats, ipass, pos);
  		assignTargets(gmats, ipass, pos);
  		if (mask.asInstanceOf[AnyRef] != null) {
  			modelmats(0) ~ modelmats(0) ∘ mask;
  		}
  		var i = 0;
  		while (i < layers.length) {
  		  if (opts.debug > 0) {
  		    println("evalbatch forward %d %s" format (i, layers(i).getClass))
  		  }
  			layers(i).forward;
  			i += 1;
  		}
  		if (putBack >= 0) {
  			output_layers(output_layers.length-1).output.colslice(0, gmats(0).ncols, gmats(1));
  		}
      val scores = zeros(output_layers.length, 1);
  		var j = 0;
      while (j < output_layers.length) {
        scores(j) = output_layers(j).score.v;
        if (ogmats != null && j < ogmats.length) ogmats(j) = output_layers(j).output;
        j += 1;
      }
      scores;
  	} else {
  	  zeros(output_layers.length, 1);
  	}
  }
  
  override def saveMetaData(fname:String) = {
    import java.io._
    val str = BIDMat.JSON.toJSON(modelMap, true);
    val writer = new PrintWriter(new File(fname + "metadata.json"));
    writer.print(str);
    writer.close;
  }
  
  override def loadMetaData(fname:String) = {
    import java.io._
    val fr = new BufferedReader(new FileReader(fname+"metadata.json"));
    val strbuf = new StringBuffer;
    var line:String = null;
    while ({line = fr.readLine(); line != null}) {
      strbuf.append(line).append("\n");
    }
    modelMap = JSON.fromJSON(strbuf.toString).asInstanceOf[HashMap[String,Int]];
  }
  
  /* 
   * Deal with annoying sub-sized minibatches
   */
  
  def extendData(mat:Mat, batchSize:Int):Mat = {
    val nrows = mat.nrows;
    val ncols = mat.ncols;
    val bsize = batchSize - ncols;
    if (bsize > 0) {
    	val newGUID = MurmurHash3.mix(MurmurHash3.mix((mat.GUID >> 32).toInt, mat.GUID.toInt),"extendData".##);
    	mat match {
    	case a:GMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = gzeros(nrows, bsize); a \ bufmat}
    	case a:GDMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = gdzeros(nrows, bsize); a \ bufmat}
    	case a:GIMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = gizeros(nrows, bsize); a \ bufmat}   
    	case a:GLMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = glzeros(nrows, bsize); a \ bufmat}
    	case a:GSMat => {val b = new GSMat(nrows, ncols, a.nnz, a.pir, a.pic, a.pjc, a.pdata, a.realnnz); b.setGUID(newGUID); b}
    	case a:GSDMat => {val b = new GSDMat(nrows, ncols, a.nnz, a.pir, a.pic, a.pjc, a.pdata, a.realnnz); b.setGUID(newGUID); b}
    	case a:FMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = zeros(nrows, bsize); a \ bufmat}
    	case a:DMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = dzeros(nrows, bsize); a \ bufmat}
    	case a:IMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = izeros(nrows, bsize); a \ bufmat}
    	case a:LMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = lzeros(nrows, bsize); a \ bufmat}
    	case a:SMat => {val b = new SMat(nrows, ncols, a.nnz, a.ir, a.jc, a.data); b.setGUID(newGUID); b}
    	case a:SDMat => {val b = new SDMat(nrows, ncols, a.nnz, a.ir, a.jc, a.data); b.setGUID(newGUID); b}
    	}
    } else {
      mat;
    }
  }
}

object Net  {
  trait Opts extends Model.Opts {
    var links:IMat = null;
    var nweight:Float = 0.1f;
    var dropout:Float = 0.5f;
    var predict:Boolean = false;
    var targetNorm:Float = 1f;
    var targmap:Mat = null;
    var dmask:Mat = null;
    var hasBias:Boolean = false;
    var aopts:ADAGrad.Opts = null;
    var nmodelmats = 0;
    var nodeset:NodeSet = null;
    var nodemat:NodeMat = null;
    var withInteractions = false;
    var tmatShape:(Int,Int) => (Array[Int], Array[Int], Array[Int], Array[Int]) = null;
    var tensorFormat:Int = Net.TensorNHWC;
  }
  
  class Options extends Opts {} 
    
  final val UseNetFormat = 0;
  final val TensorNCHW = 1;
  final val TensorNHWC = 2;
  
  def getCUDNNformat(layerFormat:Int, netFormat:Int):Int = {
    layerFormat match {
      case TensorNCHW => cudnnTensorFormat.CUDNN_TENSOR_NCHW;
      case TensorNHWC => cudnnTensorFormat.CUDNN_TENSOR_NHWC;
      case UseNetFormat => {
        netFormat match {
        case TensorNCHW => cudnnTensorFormat.CUDNN_TENSOR_NCHW;
        case TensorNHWC => cudnnTensorFormat.CUDNN_TENSOR_NHWC;         
        }
      }
    }
  }
  /**
   * Build a net with a stack of nodes. node(0) is an input node, node(n-1) is a GLM node. 
   * Intermediate nodes are Linear followed by nonlinear, starting and ending with Linear. 
   * First Linear node width is given as an argument, then it tapers off by taper.
   */
  
  def dnodes2(nslabs:Int, width:Int, taper:Float, ntargs:Int, opts:Opts, nonlin:Int = 1):NodeSet = {
    val widths = int(width * (taper ^ row(0 -> (nslabs-1)))) \ ntargs;
    powerNet(widths, opts, 0, nonlin);
  }
  
  /**
   * Build a stack of nodes. node(0) is an input node, node(n-1) is a GLM node. 
   * Intermediate nodes are linear, Nonlinear and Norm, starting and ending with Linear. 
   * First Linear node width is given as an argument, then it tapers off by taper.
   */
  
  def dnodes3(nslabs:Int, width:Int, taper:Float, ntargs:Int, opts:Opts, nonlin:Int = 1):NodeSet = {
    val widths = int(width * (taper ^ row(0 -> (nslabs-1)))) \ ntargs;
    powerNet(widths, opts, 1, nonlin);
  }
  
  /**
   * Build a stack of nodes. node(0) is an input node, node(n-1) is a GLM node. 
   * Intermediate nodes are Linear, Nonlinear, Norm, Dropout, starting and ending with Linear. 
   * First Linear node width is given as an argument, then it tapers off by taper.
   */
  
  def dnodes4(nslabs:Int, width:Int, taper:Float, ntargs:Int, opts:Opts, nonlin:Int = 1):NodeSet = {
    val widths = int(width * (taper ^ row(0 -> (nslabs-1)))) \ ntargs;
    powerNet(widths, opts, 2, nonlin);
  }
  
  /**
   * Build a net with a stack of nodes. node(0) is an input node, node(n-1) is a GLM node. 
   * Intermediate nodes are Linear followed by Nonlinear, with optional Norm and Dropout, 
   * starting and ending with Linear. 
   * The widths argument specifies the sequence of output dimensions for the Linear nodes. 
   * If a tmatShape argument is given, then that shape is used for the first linear layer. 
   */
  
  def powerNet(widths:IMat, opts:Opts, addons:Int, nonlin:Int = 1):NodeSet = {
    val thickness = 2 + addons;
    val depth = 3 + (widths.length - 1) * thickness;  
    val nodes = new NodeSet(depth);
    nodes(0) = new InputNode;
    nodes(1) = new LinNode{inputs(0) = nodes(0); outdim = widths(0); hasBias = opts.hasBias; aopts = opts.aopts; 
         withInteractions = opts.withInteractions; tmatShape = opts.tmatShape};
    for (i <- 2 until depth - 1) {
    	((i-1) % thickness) match {
    	case 0 => {
    		val w = widths((i-1)/thickness);
    		nodes(i) = new LinNode{inputs(0) = nodes(i-1); outdim = w; hasBias = opts.hasBias; aopts = opts.aopts;};
    	}
    	case 1 => {
    		nonlin match {
    		case 1 => nodes(i) = new TanhNode{inputs(0) = nodes(i-1)};
    		case 2 => nodes(i) = new SigmoidNode{inputs(0) = nodes(i-1)};
    		case 3 => nodes(i) = new RectNode{inputs(0) = nodes(i-1)};
    		case 4 => nodes(i) = new SoftplusNode{inputs(0) = nodes(i-1)};
    		}
    	}
    	case 2 => {
    		nodes(i) = new DropoutNode{inputs(0) = nodes(i-1); frac = opts.dropout};
    	}
    	case 3 => {
    		nodes(i) = new NormNode{inputs(0) = nodes(i-1); targetNorm = opts.targetNorm; weight = opts.nweight};
    	}
    	}
    }
    nodes(depth-1) = new GLMNode{inputs(0) = nodes(depth-2); links = opts.links};
    nodes;
  }

  def powerShape(tailHeight:Float)(headCount:Int, nfeats:Int):(Array[Int], Array[Int], Array[Int], Array[Int]) = {
    powerShape(tailHeight, 1f, true, true)(headCount, nfeats);
  }

  def powerShape(tailHeight:Float, power:Float)(headCount:Int, nfeats:Int):(Array[Int], Array[Int], Array[Int], Array[Int]) = {
    powerShape(tailHeight, power, true, true)(headCount, nfeats);
  }

  def powerShape(tailHeight:Float, power:Float, leftAlign:Boolean)(headCount:Int, nfeats:Int):(Array[Int], Array[Int], Array[Int], Array[Int]) = {
    powerShape(tailHeight, power, leftAlign, true)(headCount, nfeats);  
  }

  def powerShape(tailHeight:Float, power:Float, leftAlign:Boolean, horizontal:Boolean)(headCount:Int, nfeats:Int):(Array[Int], Array[Int], Array[Int], Array[Int]) = { 
    if (horizontal) powerShapeH(tailHeight, power,leftAlign)(headCount,nfeats)
    else  powerShapeV(tailHeight, power,leftAlign)(headCount,nfeats)
  }
  
  def powerShapeH(tailHeight:Float, power:Float, leftAlign:Boolean)(headCount:Int, nfeats:Int):(Array[Int], Array[Int], Array[Int], Array[Int]) = {
    var nblocks = 1;
    var tc = tailHeight;
    var ymin = 0;
    while (tc < headCount) {
      val ymax = math.min(headCount, math.round(tc - 1e-5f));
      if (ymax - ymin > 0) nblocks += 1;
      ymin = ymax;
      tc *= 2;
    }
    val y = new Array[Int](nblocks);
    val x = new Array[Int](nblocks);
    val h = new Array[Int](nblocks);
    val w = new Array[Int](nblocks);
    val ratio = math.pow(0.5, power);
    var xmax = nfeats;
    ymin = 0;
    tc = tailHeight;
    var i = 0;
    while (i < nblocks) {
    	val newx = (xmax * ratio).toInt;
      val xmin = if (leftAlign) 0 else newx; 
      val ymax = math.min(headCount, math.round(tc - 1e-5f));
      if (ymax - ymin > 0) {
      	x(i) = xmin;
      	y(i) = ymin;
      	w(i) = xmax - xmin;
      	h(i) = ymax - ymin;
      	i += 1;
      }
      xmax = newx;
      ymin = ymax;
      tc *= 2;
    }
    (y, x, h, w)
  }

  def powerShapeV(tailHeight:Float, power:Float, leftAlign:Boolean)(headCount:Int, nfeats:Int):(Array[Int], Array[Int], Array[Int], Array[Int]) = {
    var nblocks = 1;
    var tc = tailHeight;
    var ymin = 0;
    while (tc < headCount) {
      val ymax = math.min(headCount, math.round(tc - 1e-5f));
      if (ymax - ymin > 0) nblocks += 1;
      ymin = ymax;
      tc *= 2;
    }
    val y = new Array[Int](nblocks);
    val x = new Array[Int](nblocks);
    val h = new Array[Int](nblocks);
    val w = new Array[Int](nblocks);
    val ratio = math.pow(0.5, power);
    var xmax = nfeats;
    ymin = 0;
    tc = tailHeight;
    var i = nblocks - 1;
    while (i >= 0) {
    	val newx = if (i == 0) 0 else (xmax * ratio).toInt;
      val ymax = math.min(headCount, math.round(tc - 1e-5f));
      if (ymax > 0) {
      	x(i) = newx;
      	y(i) = 0;
      	w(i) = xmax - newx;
      	h(i) = ymax;
      	i -= 1;
      }
      xmax = newx;
      tc *= 2;
    }
    (y, x, h, w)
  }
  
  def mkNetModel(fopts:Model.Opts) = {
    new Net(fopts.asInstanceOf[Net.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
    new ADAGrad(nopts.asInstanceOf[ADAGrad.Opts])
  } 
  
  def mkRegularizer(nopts:Mixin.Opts):Array[Mixin] = {
    Array(new L1Regularizer(nopts.asInstanceOf[L1Regularizer.Opts]))
  }
    
  class LearnOptions extends Learner.Options with Net.Opts with MatSource.Opts with ADAGrad.Opts with L1Regularizer.Opts

  def learner(mat0:Mat, targ:Mat) = {
    val opts = new LearnOptions;
    if (opts.links == null) {
      opts.links = izeros(1,targ.nrows);
      opts.links.set(1);
    }
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1);
  	val nn = new Learner(
  	    new MatSource(Array(mat0, targ), opts), 
  	    new Net(opts), 
  	    Array(new L1Regularizer(opts)),
  	    new ADAGrad(opts), 
  	    null,
  	    opts)
    (nn, opts)
  }
  
  def learnerX(mat0:Mat, targ:Mat) = {
    val opts = new LearnOptions;
    opts.links = izeros(1,targ.nrows);
    opts.links.set(1);
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatSource(Array(mat0, targ), opts), 
  	    new Net(opts), 
  	    null,
  	    null, 
  	    null,
  	    opts)
    (nn, opts)
  }
  
  class FDSopts extends Learner.Options with Net.Opts with FileSource.Opts with ADAGrad.Opts with L1Regularizer.Opts
  
  def learner(fn1:String, fn2:String):(Learner, FDSopts) = learner(List(FileSource.simpleEnum(fn1,1,0),
  		                                                                  FileSource.simpleEnum(fn2,1,0)));
  
  def learner(fn1:String):(Learner, FDSopts) = learner(List(FileSource.simpleEnum(fn1,1,0)));

  def learner(fnames:List[(Int)=>String]):(Learner, FDSopts) = {   
    val opts = new FDSopts;
    opts.fnames = fnames
    opts.batchSize = 100000;
    opts.eltsPerSample = 500;
    implicit val threads = threadPool(4);
    val ds = new FileSource(opts)
  	val nn = new Learner(
  			ds, 
  	    new Net(opts), 
  	    Array(new L1Regularizer(opts)),
  	    new ADAGrad(opts), 
  	    null,
  	    opts)
    (nn, opts)
  } 
  
  def learnerX(fn1:String, fn2:String):(Learner, FDSopts) = learnerX(List(FileSource.simpleEnum(fn1,1,0),
  		                                                                  FileSource.simpleEnum(fn2,1,0)));
  
  def learnerX(fn1:String):(Learner, FDSopts) = learnerX(List(FileSource.simpleEnum(fn1,1,0)));
  
  def learnerX(fnames:List[(Int)=>String]):(Learner, FDSopts) = {   
    val opts = new FDSopts
    opts.fnames = fnames
    opts.batchSize = 100000;
    opts.eltsPerSample = 500;
    val ds = new FileSource(opts);
    // val net = dnodes(3, 0, 1f, opts.targmap.nrows, opts)                   // default to a 3-node network
  	val nn = new Learner(ds, 
  	                     new Net(opts), 
  	                     null,
  	                     null, 
  	                     null,
  	                     opts)
    (nn, opts)
  }

  
  class PredOptions extends Learner.Options with Net.Opts with MatSource.Opts with MatSink.Opts;
  
  def predictor(model0:Model, data0:Mat, labels0:Mat):(Learner, PredOptions) = {
    val model = model0.asInstanceOf[Net];
    val mopts = model.opts;
    val opts = new PredOptions;
    opts.batchSize = mopts.asInstanceOf[DataSource.Opts].batchSize;
    opts.links = mopts.links;
    opts.nodeset = mopts.nodeset;
    opts.nodemat = mopts.nodemat;
    if (opts.nodeset.asInstanceOf[AnyRef] != null) {
    	opts.nodeset.nodes.foreach({case nx:LinNode => nx.aopts = null; case _ => Unit})
    }
    if (opts.nodemat.asInstanceOf[AnyRef] != null) {
    	opts.nodemat.data.foreach({case nx:LinNode => nx.aopts = null; case _ => Unit})
    }
    opts.hasBias = mopts.hasBias;
    opts.tensorFormat = mopts.tensorFormat;
    opts.dropout = 1f;
    
    val newmod = new Net(opts);
    newmod.refresh = false;
    newmod.copyFrom(model)
    val nn = new Learner(
        new MatSource(Array(data0, labels0), opts), 
        newmod, 
        null,
        null, 
        new MatSink(opts),
        opts);
    (nn, opts)
  }
  
  class FilePredOptions extends Learner.Options with Net.Opts with FileSource.Opts with FileSink.Opts;
  
  def predictor(model0:Model, infn:String, outfn:String):(Learner, FilePredOptions) = {
    predictor(model0, List(FileSource.simpleEnum(infn,1,0)), List(FileSource.simpleEnum(outfn,1,0)));
  }

  def predictor(model0:Model, infn:String, inlb:String, outfn:String):(Learner, FilePredOptions) = {
    predictor(model0, List(FileSource.simpleEnum(infn,1,0),FileSource.simpleEnum(inlb,1,0)), List(FileSource.simpleEnum(outfn,1,0)));
  }
  
  def predictor(model0:Model, infiles:List[(Int)=>String], outfiles:List[(Int)=>String]):(Learner, FilePredOptions) = {
    val model = model0.asInstanceOf[Net];
    val mopts = model.opts;
    val opts = new FilePredOptions;
    opts.fnames = infiles;
    opts.ofnames = outfiles;
    opts.links = mopts.links;
    opts.nodeset = mopts.nodeset;
    opts.nodemat = mopts.nodemat;
    if (opts.nodeset.asInstanceOf[AnyRef] != null) {
    	opts.nodeset.nodes.foreach({case nx:LinNode => nx.aopts = null; case _ => Unit})
    }
    if (opts.nodemat.asInstanceOf[AnyRef] != null) {
    	opts.nodemat.data.foreach({case nx:LinNode => nx.aopts = null; case _ => Unit})
    }
    opts.hasBias = mopts.hasBias;
    opts.tensorFormat = mopts.tensorFormat;
    opts.dropout = 1f;
    
    val newmod = new Net(opts);
    newmod.refresh = false;
    newmod.copyFrom(model);
    val dsource = new FileSource(opts);
    val dsink = new FileSink(opts);
    val nn = new Learner(
        dsource, 
        newmod, 
        null,
        null, 
        dsink,
        opts);
    (nn, opts)
  }
  
  class LearnParOptions extends ParLearner.Options with Net.Opts with FileSource.Opts with ADAGrad.Opts with L1Regularizer.Opts;
  
  def learnPar(fn1:String, fn2:String):(ParLearnerF, LearnParOptions) = {learnPar(List(FileSource.simpleEnum(fn1,1,0), FileSource.simpleEnum(fn2,1,0)))}
  
  def learnPar(fnames:List[(Int) => String]):(ParLearnerF, LearnParOptions) = {
    val opts = new LearnParOptions;
    opts.batchSize = 10000;
    opts.lrate = 1f;
    opts.fnames = fnames;
    implicit val threads = threadPool(4)
    val nn = new ParLearnerF(
        new FileSource(opts), 
        opts, mkNetModel _,
        opts, mkRegularizer _,
        opts, mkUpdater _, 
        null, null,
        opts)
    (nn, opts)
  }
}


