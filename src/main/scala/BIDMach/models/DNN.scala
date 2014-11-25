package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._

class DNN(override val opts:DNN.Opts = new DNN.Options) extends Model(opts) {
  var layers:Array[Layer] = null

  override def init() = {
    mats = datasource.next;
    datasource.reset;
    var nfeats = mats(0).ncols
    layers = new Array[Layer](opts.layers.length);
    var imodel = 0;
    val nmats = opts.layers.count(_ match {case x:DNN.ModelLayerSpec => true; case _ => false})
    modelmats = new Array[Mat](nmats);
    updatemats = new Array[Mat](nmats);
    
	for (i <- 0 until opts.layers.length) {
	  opts.layers(i) match {
	    case fcs:DNN.FC => {
	      layers(i) = new FCLayer(imodel);
	      modelmats(imodel) = if (useGPU) gzeros(fcs.outsize, nfeats) else zeros(fcs.outsize, nfeats);
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
    layers(layers.length-1).data = gmats(1);
    var i = 1
    while (i < layers.length) {
      layers(i).forward;
      i += 1;
    }
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
      input.deriv = modelmats(imodel) ^* deriv;
      updatemats(imodel) = deriv *^ input.data;
    }
  }
  
  class RectLayer extends Layer {
    override def forward = {
      data = max(input.data, 0)
    }
    
    override def backward = {
      input.deriv = deriv ∘ (input.data > 0);
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
      data = input.data + 0
      GLM.preds(data, data, ilinks, totflops)
    }
    
    override def backward = {
      input.deriv = data + 1
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
  }
  
  class Options extends Opts {}
  
  class LayerSpec {}
  
  class ModelLayerSpec extends LayerSpec{}
  
  class FC(val outsize:Int) extends ModelLayerSpec {}
  
  class Rect extends LayerSpec {}
  
  class Input extends LayerSpec {}
  
  class GLM(val links:IMat) extends LayerSpec {}

  def learner(mat0:Mat, d:Int) = {
    class xopts extends Learner.Options with DNN.Opts with MatDS.Opts with ADAGrad.Opts
    val opts = new xopts
    opts.dim = d
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts), 
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
    implicit val threads = threadPool(4)
  	val nn = new Learner(
  	    new SFilesDS(opts), 
  	    new DNN(opts), 
  	    null,
  	    new ADAGrad(opts), 
  	    opts)
    (nn, opts)
  } 

}


