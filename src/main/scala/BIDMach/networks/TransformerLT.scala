package BIDMach.networks

import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.datasinks._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach.networks.layers._
import BIDMach._

/*
 * Transformer-LT network. 
 */

@SerialVersionUID(100L)
class TransformerLT(override val opts:TransformerLT.Opts = new TransformerLT.Options) extends Model(opts) {
  
  var table:Array[Mat] = null;
  var dtable:Array[Mat] = null;
  var batchSize:Int = 0;
  var txNets:Array[Net] = null;
  val kmodels = 5
  var cacheState = false;
  var cacheGPUstate = false;
  var useCache = false;
  var useGPUCache = true;

  override def init() = {
	useGPU = opts.useGPU && Mat.hasCUDA > 0;
	useDouble = opts.useDouble;
    cacheState = Mat.useCache;
    Mat.useCache = useCache;
    cacheGPUstate = Mat.useGPUcache;
    Mat.useGPUcache = useGPUCache;
  }  
  
  def wrapUp() = { 
    Mat.useCache = cacheState;
    Mat.useGPUcache = cacheGPUstate;
  }

  def createTables() { 
    table = new Array[Mat](opts.depth);
    dtable = new Array[Mat](opts.depth);
    setmodelmats(new Array[Mat](opts.depth * kmodels * 2));
    updatemats = new Array[Mat](opts.depth * kmodels * 2);
    for (i <- 0 until opts.depth) { 
      table(i) = convertMat(zeros(opts.dim, opts.seqlength + opts.degree));
      dtable(i) = convertMat(zeros(opts.dim, opts.seqlength + opts.degree));
      for (j <- 0 until kmodels) { 
        modelmats(2 * (j + kmodels * i)) = convertMat(zeros(opts.dim,opts.dim));
      }
    }

  }


  def attach(net:Net, level:Int = 0) { 
    net.layers(0).output = table(level);
    net.layers(0).deriv = dtable(level);
    net.layers(5).asInstanceOf[ModelLayer].imodel = level * kmodels;
    net.layers(6).asInstanceOf[ModelLayer].imodel = level * kmodels + 2;
    net.layers(7).asInstanceOf[ModelLayer].imodel = level * kmodels + 4;
    net.layers(8).asInstanceOf[ModelLayer].imodel = level * kmodels + 2;
    net.layers(9).asInstanceOf[ModelLayer].imodel = level * kmodels + 4;
    net.layers(36).asInstanceOf[ModelLayer].imodel = level * kmodels + 6;
    net.layers(39).asInstanceOf[ModelLayer].imodel = level * kmodels + 8;
  }

  def createTxNet(seqlength:Int) = {
    val net = new Net();
    val nopts = net.opts;
    nopts.useGPU = useGPU;
    net.useGPU = useGPU;

    import BIDMach.networks.layers.Node._
    val innerdim = opts.indim;
    val dim =      opts.dim;
    val degree =   opts.degree;
    val basedims = irow(opts.dim, seqlength);
    val headdims = irow(opts.dim/opts.nheads, opts.nheads, degree, seqlength/degree);
    val headdims2 = irow(opts.dim/opts.nheads, opts.nheads, degree*2, seqlength/degree);
    val headdims_2d = irow(opts.dim/opts.nheads, opts.nheads * seqlength);
    val headdims2_2d = irow(opts.dim/opts.nheads, opts.nheads * seqlength * 2);
    val headdimsx = irow(opts.dim/opts.nheads, degree, opts.nheads, seqlength/degree);
    val headdimsx2 = irow(opts.dim/opts.nheads, degree*2, opts.nheads, seqlength/degree);
    val headinds__ = irow(0->(opts.nheads * seqlength)).reshapeView(opts.nheads, degree, seqlength/degree);
    val headinds_ = headinds__.transpose(1\0\2).reshapeView(1,opts.nheads*seqlength);
    val headinds2__ = irow(0->(opts.nheads * seqlength * 2)).reshapeView(opts.nheads, degree*2, seqlength/degree);
    val headinds2_ = headinds2__.transpose(1\0\2).reshapeView(1,opts.nheads*seqlength*2);
    val invheadinds__ = irow(0->(opts.nheads * seqlength)).reshapeView(degree, opts.nheads, seqlength/degree);
    val invheadinds_ = invheadinds__.transpose(1\0\2).reshapeView(1,opts.nheads*seqlength);
    val headperm2 = irow(2,0,1,3);
    val hasBias  = opts.hasBias;

    val cmask_ =   zeros((degree*2) \ degree \ opts.nheads \ (seqlength/degree));
    val col = icol(0->degree);
    for (i <- 0 until seqlength/degree) { 
      for (j <- 0 until opts.nheads) { 
        for (k <- 0 until degree) { 
          cmask_(k + 1 + col, k, j, i) = 1f;
        }
      }
    }
    val smask_ =   (1f - cmask_) *@ -1000f;

    val in_qkv =      input;
    val this_in =     colslice(in_qkv)(degree, seqlength+degree);
    val last_in =     colslice(in_qkv)(0, seqlength);
    val headinds =    constant(headinds_)(true);
    val headinds2 =   constant(headinds2_)(true);

    val proj_q_this = linear(this_in)(outdim=innerdim, hasBias=hasBias); // layers 5-9
    val proj_k_this = linear(this_in)(outdim=innerdim, hasBias=hasBias);
    val proj_v_this = linear(this_in)(outdim=innerdim, hasBias=hasBias);   
    val proj_k_last = linear(last_in)(outdim=innerdim, hasBias=hasBias);
    val proj_v_last = linear(last_in)(outdim=innerdim, hasBias=hasBias);   

    val queries_2d =  reshape(proj_q_this)(headdims_2d,false);
    val keys_this =   reshape(proj_k_this)(headdims,false);
    val vals_this =   reshape(proj_v_this)(headdims,false);
    val keys_last =   reshape(proj_k_last)(headdims,false);
    val vals_last =   reshape(proj_v_last)(headdims,false);

    val keys =        stack(keys_last, keys_this)(2);
    val vals =        stack(vals_last, vals_this)(2);
    val keys_2d =     reshape(keys)(headdims2_2d,false);
    val vals_2d =     reshape(vals)(headdims2_2d,false);
    val queriesx_2d = colperm(queries_2d, headinds);

    val keysx_2d =    colperm(keys_2d, headinds2); // layer 20
    val valsx_2d =    colperm(vals_2d, headinds2);
    val queriesx =    reshape(queriesx_2d)(headdimsx,false);
    val keysx =       reshape(keysx_2d)(headdimsx2,false);
    val valsx =       reshape(valsx_2d)(headdimsx2,false);

    val prod =        keysx ^* queriesx;
    val cmask =       constant(cmask_)(true);
    val smask =       constant(smask_)(true);
    val mprod =       prod *@ cmask;
    val oprod =       mprod + smask;

    val weights =     softmaxx(oprod)(); // layer 30
    val wvals =       valsx * weights;
    val wvals_2d =    reshape(wvals)(headdims_2d,false);
    val invheadinds = constant(invheadinds_)(true);
    val pvals_2d =    colperm(wvals_2d, invheadinds);

    val pvals =       reshape(pvals_2d)(basedims,false);
    val mhattn =      linear(pvals)(outdim=dim, hasBias=hasBias); // layer 36
    val norm1 =       layerNorm(mhattn)();
    val sum1 =        norm1 + this_in;
    val ffwd1 =       linear(sum1)(outdim=dim, hasBias=hasBias);  // layer 39

    val relu1 =       relu(ffwd1)();
    val norm2 =       layerNorm(relu1)();
    val sum2 =        sum1 + norm2;
    
    val nodes     = in_qkv       \ this_in      \ last_in      \ headinds     \ headinds2    on
                    proj_q_this  \ proj_k_this  \ proj_v_this  \ proj_k_last  \ proj_v_last  on
                    queries_2d   \ keys_this    \ vals_this    \ keys_last    \ vals_last    on
                    keys         \ vals         \ keys_2d      \ vals_2d      \ queriesx_2d  on
                    keysx_2d     \ valsx_2d     \ queriesx     \ keysx        \ valsx        on
                    prod         \ cmask        \ smask        \ mprod        \ oprod        on
                    weights      \ wvals        \ wvals_2d     \ invheadinds  \ pvals_2d     on
                    pvals        \ mhattn       \ norm1        \ sum1         \ ffwd1        on
                    relu1        \ norm2        \ sum2         \ null         \ null;

    nopts.nodemat = nodes.t;
 
    net.output_nodes = Array(sum2);
    net.createLayers;
    net;
  }

  override def dobatch(gmats:Array[Mat], ipass:Int, pos:Long):Unit = {
    if (batchSize < 0) batchSize = gmats(0).ncols;
    if (batchSize == gmats(0).ncols) {                                    // discard odd-sized minibatches
    }
  }
  
  override def evalbatch(mats:Array[Mat], ipass:Int, pos:Long):FMat = {  
    if (batchSize < 0) batchSize = gmats(0).ncols;
    if (batchSize == gmats(0).ncols) { 
    }
    zeros(1, 1);
  }
}

@SerialVersionUID(100L)
object TransformerLT {
  trait Opts extends Model.Opts {
    var seqlength = 16384;
    dim = 512;
    var indim = 512;
    var degree = 128;
    var nheads = 8;
    var depth = 32;
    var stride = 4;
    var firststrided = 10;
    var nstrided = 6;
    var hasBias = true;
  }
  
@SerialVersionUID(100L)
  class Options extends Opts {}
  
@SerialVersionUID(100L)
  class LearnOptions extends Learner.Options with TransformerLT.Opts with MatSource.Opts with Grad.Opts

  def learner(mat0:Mat, mat1:Mat, regularize:Boolean = false) = {
    val opts = new LearnOptions;
    opts.batchSize = 128;
  	val nn = new Learner(
  	    new MatSource(Array(mat0, mat1), opts), 
  	    new TransformerLT(opts), 
        null,
  	    new Grad(opts), 
  	    null,
  	    opts)
    (nn, opts)
  }

  def testsetup(opts:Opts = new Options):TransformerLT = { 
    opts.depth = 2;
    val trans = new TransformerLT(opts);
    trans.init();
    trans.createTables();
    val net = trans.createTxNet(opts.seqlength);
    net.setmodelmats(trans.modelmats);
    net.updatemats = trans.updatemats;
    trans.attach(net, 0);
    trans.txNets = Array(net);
    trans;
  }

  def testfwd(trans:TransformerLT, n:Int) = { 
    for (i <- 0 until n) { 
      trans.txNets(0).forward
    }
  }
  
  
  
  def load(fname:String):TransformerLT = {
  	val mm = new TransformerLT;
  	mm.loadMetaData(fname);
  	mm.load(fname);
  	mm
  }

}

