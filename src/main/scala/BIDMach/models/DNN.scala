package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach._

/**
 * Basic DNN class. Learns a supervised map from input blocks to output (target) data blocks. There are currently 4 layer types:
 - InputLayer: just a placeholder for the first layer which is loaded with input data blocks. No learnable params. 
 - FCLayer: Fully-Connected Linear layer. Has a matrix of learnable params which is the input-output map. 
 - RectLayer: Rectifying one-to-one layer. No params.
 - GLMLayer: a one-to-one layer with GLM mappings (linear, logistic, abs-logistic and SVM). No learnable params. 
 *
 * The network topology is specified by opts.layers which is a sequence of "LayerSpec" objects. There is a LayerSpec
 * Class for each Layer class, which holds the params for defining that layer. Currently only two LayerSpec types need params:
 - FC: holds the output dimensions of the FClayer (input dimension set by previous layer). 
 - GLM: holds the links matrix (integer specs for loss types, see GLM), for the output of that layer. Its size should match the
 * number of targets. 
 */

class DNN(override val opts:DNN.Opts = new DNN.Options) extends Model(opts) {
  var layers:Array[Layer] = null

  override def init() = {
    if (refresh) {
    	mats = datasource.next;
    	var nfeats = mats(0).nrows;
    	datasource.reset;
    	layers = new Array[Layer](opts.layers.length);
    	var imodel = 0;
    	val nmodelmats = opts.layers.count(_ match {case x:DNN.ModelLayerSpec => true; case _ => false});
    	setmodelmats(Array[Mat](nmodelmats));
    	updatemats = new Array[Mat](nmodelmats);

    	for (i <- 0 until opts.layers.length) {
    		opts.layers(i) match {
    		case fcs:DNN.FC => {
    			layers(i) = new FCLayer(imodel);
    			modelmats(imodel) = if (useGPU) gnormrnd(0, 1, fcs.outsize, nfeats) else normrnd(0, 1, fcs.outsize, nfeats);
    			updatemats(imodel) = if (useGPU) gzeros(fcs.outsize, nfeats) else zeros(fcs.outsize, nfeats);
    			nfeats = fcs.outsize;
    			imodel += 1;
    		}
    		case rls:DNN.Rect => {
    			layers(i) = new RectLayer;
    		}
    		case ils:DNN.Input => {
    			layers(i) = new InputLayer;
    		}
    		case ols:DNN.GLM => {
    			layers(i) = new GLMLayer(ols.links);
    		}
    		}
    		if (i > 0) layers(i).input = layers(i-1)
    	}
    }
  }
  
  
  def doblock(gmats:Array[Mat], ipass:Int, i:Long):Unit = {
    layers(0).data = gmats(0);
    layers(layers.length-1).target = gmats(1);
    var i = 1
    while (i < layers.length) {
      layers(i).forward;
      i += 1;
    }
    while (i > 1) {
      i -= 1;
      layers(i).backward;
    }
  }
  
  def evalblock(mats:Array[Mat], ipass:Int, here:Long):FMat = {  
    layers(0).data = gmats(0);
    val targ = gmats(1);
    layers(layers.length-1).data = targ;
    var i = 1;
    while (i < layers.length) {
      layers(i).forward;
      i += 1;
    }
    if (putBack >= 0) {targ <-- layers(layers.length-1).data}
    layers(layers.length-1).score
  }
  
  class Layer {
    var data:Mat = null;
    var target:Mat = null;
    var deriv:Mat = null;
    var input:Layer = null;
    def forward = {};
    def backward = {};
    def score:FMat = zeros(1,1);
  }
  
  class FCLayer(val imodel:Int) extends Layer {
    override def forward = {
      data = modelmats(imodel) * input.data;
    }
    
    override def backward = {
      if (imodel > 0) input.deriv = modelmats(imodel) ^* deriv;
      updatemats(imodel) = deriv *^ input.data;
    }
  }
  
  class RectLayer extends Layer {
    override def forward = {
      data = max(input.data, 0f)
    }
    
    override def backward = {
      input.deriv = deriv ∘ (input.data > 0f);
    }
  }
  
  class InputLayer extends Layer {
  }
  
  class GLMLayer(val links:IMat) extends Layer {
    val ilinks = if (useGPU) GIMat(links) else links;
    var totflops = 0L;
    for (i <- 0 until links.length) {
      totflops += GLM.linkArray(links(i)).fnflops
    }
    
    override def forward = {
      data = input.data + 0f
      GLM.preds(data, data, ilinks, totflops)
    }
    
    override def backward = {
      input.deriv = data + 1f
      GLM.derivs(data, target, input.deriv, ilinks, totflops)
      if (deriv.asInstanceOf[AnyRef] != null) {
        input.deriv ~ input.deriv ∘ deriv;
      }
    }
    
    override def score:FMat = { 
      val v = GLM.llfun(data, target, ilinks, totflops);
      FMat(mean(v, 2));
    }
  }
}

object DNN  {
  trait Opts extends Model.Opts {
	var layers:Seq[LayerSpec] = null;
    var links:IMat = null;
  }
  
  class Options extends Opts {}
  
  class LayerSpec {}
  
  class ModelLayerSpec extends LayerSpec{}
  
  class FC(val outsize:Int) extends ModelLayerSpec {}
  
  class Rect extends LayerSpec {}
  
  class Input extends LayerSpec {}
  
  class GLM(val links:IMat) extends LayerSpec {}
  
  /**
   * Build a stack of layers. layer(0) is an input layer, layer(n-1) is a GLM layer. 
   * Intermediate layers are FC alternating with Rect, starting and ending with FC. 
   * First FC layer width is given as an argument, then it tapers off by taper.
   */
  
  def dlayers(depth0:Int, width:Int, taper:Float, ntargs:Int, opts:Opts) = {
    val depth = (depth0/2)*2 + 1;              // Round up to an odd number of layers 
    val layers = new Array[LayerSpec](depth);
    var w = width
    for (i <- 1 until depth - 2) {
    	if (i % 2 == 1) {
    		layers(i) = new FC(w);
    		w = (taper*w).toInt;
    	} else {
    		layers(i) = new Rect;
    	}
    }
    layers(0) = new Input;
    layers(depth-2) = new FC(ntargs);
    layers(depth-1) = new GLM(opts.links);
    opts.layers = layers
    layers
  }

  def learner(mat0:Mat, targ:Mat, d:Int) = {
    class xopts extends Learner.Options with DNN.Opts with MatDS.Opts with ADAGrad.Opts
    val opts = new xopts
    if (opts.links == null) opts.links = izeros(1,targ.nrows)
    opts.links.set(d)
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    dlayers(3, 0, 1f, targ.nrows, opts)                   // default to a 3-layer network
  	val nn = new Learner(
  	    new MatDS(Array(mat0, targ), opts), 
  	    new DNN(opts), 
  	    null,
  	    new ADAGrad(opts), 
  	    opts)
    (nn, opts)
  }

  def learner(fnames:List[(Int)=>String], d:Int) = {
    class xopts extends Learner.Options with DNN.Opts with SFilesDS.Opts with ADAGrad.Opts
    val opts = new xopts
    opts.dim = d
    opts.fnames = fnames
    opts.batchSize = 100000;
    opts.eltsPerSample = 500;
    implicit val threads = threadPool(4);
    val ds = new SFilesDS(opts)
//    dlayers(3, 0, 1f, targ.nrows, opts)                   // default to a 3-layer network
  	val nn = new Learner(
  			ds, 
  	    new DNN(opts), 
  	    null,
  	    new ADAGrad(opts), 
  	    opts)
    (nn, opts)
  } 
  
  class LearnOptions extends Learner.Options with DNN.Opts with MatDS.Opts with ADAGrad.Opts with L1Regularizer.Opts
    
    // This function constructs a learner and a predictor. 
  def learner(mat0:Mat, targ:Mat, mat1:Mat, preds:Mat, d:Int):(Learner, LearnOptions, Learner, LearnOptions) = {
    val mopts = new LearnOptions;
    val nopts = new LearnOptions;
    mopts.lrate = 1f
    mopts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    mopts.autoReset = false
    if (mopts.links == null) mopts.links = izeros(targ.nrows,1)
    nopts.links = mopts.links
    mopts.links.set(d)
    nopts.batchSize = mopts.batchSize
    nopts.putBack = 1
    dlayers(3, 0, 1f, targ.nrows, mopts)                   // default to a 3-layer network
    val model = new DNN(mopts)
    val mm = new Learner(
        new MatDS(Array(mat0, targ), mopts), 
        model, 
        Array(new L1Regularizer(mopts)),
        new ADAGrad(mopts), mopts)
    val nn = new Learner(
        new MatDS(Array(mat1, preds), nopts), 
        model, 
        null,
        null, 
        nopts)
    (mm, mopts, nn, nopts)
  }

}


