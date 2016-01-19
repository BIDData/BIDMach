package BIDMach.networks

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,JSON,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.datasinks._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach._
import BIDMach.networks.layers._
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
  var output_layers:Array[Layer] = null;
  var targmap:Mat = null;
  var mask:Mat = null;
  var bufmat:Mat = null;
  var modelMap:HashMap[String,Int] = null;
  var batchSize = -1;
  var imodel = 0;
  var initialize = false;

  override def init() = {
	  mats = datasource.next;
	  var nfeats = mats(0).nrows;
	  batchSize = mats(0).ncols
	  targmap = if (opts.targmap.asInstanceOf[AnyRef] != null) convertMat(opts.targmap) else null;
	  mask = if (opts.dmask.asInstanceOf[AnyRef] != null) convertMat(opts.dmask) else null;
	  createLayers;
    if (output_layers == null) output_layers = Array(layers(layers.length-1));
	  if (modelMap == null) {
	  	modelMap = new HashMap[String,Int];
	  }
	  imodel = 0;
	  layers.map(_.getModelMats(this));
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
	  if (useGPU) copyMats(mats, gmats);
	  val pb = putBack;
	  putBack = -1;
    initialize = true;
    evalbatch(gmats, 0, 0);
    initialize = false;
    putBack = pb;
	  datasource.reset;
  }
  
  def createLayers = {
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
  }
  
  def assignInputs(gmats:Array[Mat], ipass:Int, pos:Long) {
  	layers(0).output = gmats(0);
  }
  
  def assignTargets(gmats:Array[Mat], ipass:Int, pos:Long) {
  	if (targmap.asInstanceOf[AnyRef] != null) {
  		layers(layers.length-1).target = targmap * gmats(0);
  	} else if (gmats.length > 1) {
  		layers(layers.length-1).target = full(gmats(1));
  	}
  }
  
  
  def dobatch(gmats:Array[Mat], ipass:Int, pos:Long):Unit = {
    if (batchSize < 0) batchSize = gmats(0).ncols;
    if (batchSize == gmats(0).ncols) {                                    // discard odd-sized minibatches
    	assignInputs(gmats, ipass, pos);
    	assignTargets(gmats, ipass, pos);
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
      var j = 0;
      while (j < output_layers.length) {
    	  output_layers(j).deriv.set(1);
    	  j += 1;
      }
      if (opts.aopts == null) {
    	  for (j <- 0 until updatemats.length) updatemats(j).clear;
      }
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
  }
  
  // no backward pass
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
        if (ogmats != null && j < ogmats.length) ogmats(j) = output_layers(j).output.asMat;
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
    	case a:FMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = zeros(nrows, bsize); a \ bufmat}
    	case a:DMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = dzeros(nrows, bsize); a \ bufmat}
    	case a:IMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = izeros(nrows, bsize); a \ bufmat}
    	case a:LMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = lzeros(nrows, bsize); a \ bufmat}
    	case a:GMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = gzeros(nrows, bsize); a \ bufmat}
    	case a:GDMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = gdzeros(nrows, bsize); a \ bufmat}
    	case a:GIMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = gizeros(nrows, bsize); a \ bufmat}   
    	case a:GLMat => {if (bufmat.asInstanceOf[AnyRef] == null) bufmat = glzeros(nrows, bsize); a \ bufmat}
    	case a:SMat => {val b = new SMat(nrows, ncols, a.nnz, a.ir, a.jc, a.data); b.setGUID(newGUID); b}
    	case a:SDMat => {val b = new SDMat(nrows, ncols, a.nnz, a.ir, a.jc, a.data); b.setGUID(newGUID); b}
    	case a:GSMat => {val b = new GSMat(nrows, ncols, a.nnz, a.ir, a.ic, a.jc, a.data, a.realnnz); b.setGUID(newGUID); b}
    	case a:GSDMat => {val b = new GSDMat(nrows, ncols, a.nnz, a.ir, a.ic, a.jc, a.data, a.realnnz); b.setGUID(newGUID); b}
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
  }
  
  class Options extends Opts {}
  
  
  /**
   * Build a net with a stack of nodes. node(0) is an input node, node(n-1) is a GLM node. 
   * Intermediate nodes are Linear alternating with Rect, starting and ending with Linear. 
   * First Linear node width is given as an argument, then it tapers off by taper.
   */
  
  def dnodes(depth0:Int, width:Int, taper:Float, ntargs:Int, opts:Opts, nonlin:Int = 1):NodeSet = {
    val depth = (depth0/2)*2 + 1;              // Round up to an odd number of nodes 
    val nodes = new NodeSet(depth);
    var w = width;
    nodes(0) = new InputNode;
    for (i <- 1 until depth - 2) {
    	if (i % 2 == 1) {
    		nodes(i) = new LinNode{inputs(0) = nodes(i-1); outdim = w; hasBias = opts.hasBias; aopts = opts.aopts};
    		w = (taper*w).toInt;
    	} else {
    	  nonlin match {
    	    case 1 => nodes(i) = new TanhNode{inputs(0) = nodes(i-1)};
    	    case 2 => nodes(i) = new SigmoidNode{inputs(0) = nodes(i-1)};
    	    case 3 => nodes(i) = new RectNode{inputs(0) = nodes(i-1)};
    	    case 4 => nodes(i) = new SoftplusNode{inputs(0) = nodes(i-1)};
    	  }
    	}
    }
    nodes(depth-2) = new LinNode{inputs(0) = nodes(depth-3); outdim = ntargs; hasBias =  opts.hasBias; aopts = opts.aopts};
    nodes(depth-1) = new GLMNode{inputs(0) = nodes(depth-2); links = opts.links};
    nodes;
  }
  
  /**
   * Build a stack of nodes. node(0) is an input node, node(n-1) is a GLM node. 
   * Intermediate nodes are linear, Rect, Norm, starting and ending with Linear. 
   * First Linear node width is given as an argument, then it tapers off by taper.
   */
  
  def dnodes3(depth0:Int, width:Int, taper:Float, ntargs:Int, opts:Opts, nonlin:Int = 1):NodeSet = {
    val depth = (depth0/3)*3;              // Round up to an odd number of nodes 
    val nodes = new NodeSet(depth);
    var w = width;
    nodes(0) = new InputNode;
    for (i <- 1 until depth - 2) {
    	if (i % 3 == 1) {
    		nodes(i) = new LinNode{inputs(0) = nodes(i-1); outdim = w; hasBias = opts.hasBias; aopts = opts.aopts};
    		w = (taper*w).toInt;
    	} else if (i % 3 == 2) {
    	  nonlin match {
    	    case 1 => nodes(i) = new TanhNode{inputs(0) = nodes(i-1)};
    	    case 2 => nodes(i) = new SigmoidNode{inputs(0) = nodes(i-1)};
    	    case 3 => nodes(i) = new RectNode{inputs(0) = nodes(i-1)};
    	    case 4 => nodes(i) = new SoftplusNode{inputs(0) = nodes(i-1)};
    	  }
    	} else {
    		nodes(i) = new NormNode{inputs(0) = nodes(i-1); targetNorm = opts.targetNorm; weight = opts.nweight};
    	}
    }
    nodes(depth-2) = new LinNode{inputs(0) = nodes(depth-3); outdim = ntargs; hasBias = opts.hasBias; aopts = opts.aopts};
    nodes(depth-1) = new GLMNode{inputs(0) = nodes(depth-2); links = opts.links};
    nodes;
  }
  
  /**
   * Build a stack of nodes. node(0) is an input node, node(n-1) is a GLM node. 
   * Intermediate nodes are Linear, Rect, Norm, Dropout, starting and ending with Linear. 
   * First Linear node width is given as an argument, then it tapers off by taper.
   */
  
  def dnodes4(depth0:Int, width:Int, taper:Float, ntargs:Int, opts:Opts, nonlin:Int = 1):NodeSet = {
    val depth = ((depth0+1)/4)*4 - 1;              // Round up to an odd number of nodes 
    val nodes = new NodeSet(depth);
    var w = width;
    nodes(0) = new InputNode;
    for (i <- 1 until depth - 2) {
    	(i % 4) match {
    	  case 1 => {
    	  	nodes(i) = new LinNode{inputs(0) = nodes(i-1); outdim = w; hasBias = opts.hasBias; aopts = opts.aopts};
    	  	w = (taper*w).toInt;
    	  }
    	  case 2 => {
    	  	nonlin match {
    	  	case 1 => nodes(i) = new TanhNode{inputs(0) = nodes(i-1)};
    	  	case 2 => nodes(i) = new SigmoidNode{inputs(0) = nodes(i-1)};
    	  	case 3 => nodes(i) = new RectNode{inputs(0) = nodes(i-1)};
    	  	case 4 => nodes(i) = new SoftplusNode{inputs(0) = nodes(i-1)};
    	  	}
    	  }
    	  case 3 => {
    	  	nodes(i) = new NormNode{inputs(0) = nodes(i-1); targetNorm = opts.targetNorm; weight = opts.nweight};
      }
    	  case _ => {
    	  	nodes(i) = new DropoutNode{inputs(0) = nodes(i-1); frac = opts.dropout};
    	  }
    	}
    }
    nodes(depth-2) = new LinNode{inputs(0) = nodes(depth-3); outdim = ntargs; hasBias =  opts.hasBias; aopts = opts.aopts};
    nodes(depth-1) = new GLMNode{inputs(0) = nodes(depth-2); links = opts.links};
    nodes;
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
    implicit val threads = threadPool(4);
    val ds = new FileSource(opts)
    val net = dnodes(3, 0, 1f, opts.targmap.nrows, opts)                   // default to a 3-node network
  	val nn = new Learner(
  			ds, 
  	    new Net(opts), 
  	    null,
  	    null, 
  	    null,
  	    opts)
    (nn, opts)
  }

  
  class PredOptions extends Learner.Options with Net.Opts with MatSource.Opts with MatSink.Opts
  
  def predictor(model0:Model, mat0:Mat):(Learner, PredOptions) = {
    val model = model0.asInstanceOf[Net];
    val mopts = model.opts;
    val opts = new PredOptions;
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1);
    opts.links = mopts.links;
    opts.nodeset = mopts.nodeset.clone;
    opts.nodeset.nodes.foreach({case nx:LinNode => nx.aopts = null; case _ => Unit})
    opts.hasBias = mopts.hasBias;
    opts.dropout = 1f;
    
    val newmod = new Net(opts);
    newmod.refresh = false;
    newmod.copyFrom(model)
    val nn = new Learner(
        new MatSource(Array(mat0), opts), 
        newmod, 
        null,
        null, 
        new MatSink(opts),
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


