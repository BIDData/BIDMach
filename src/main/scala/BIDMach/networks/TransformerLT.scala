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


  override def init() = {
  }  

  def createTables(opts:TransformerLT.Opts) { 
    table = new Array[Mat](opts.depth);
    dtable = new Array[Mat](opts.depth);
    for (i <- 0 until opts.depth) { 
      table(i) = convertMat(zeros(opts.dim, opts.seqlength + opts.degree));
      dtable(i) = convertMat(zeros(opts.dim, opts.seqlength + opts.degree));
    }
  }

  def createTxNet(seqlength:Int) = {
    val net = new Net();
    val nopts = net.opts;

    import BIDMach.networks.layers.Node._
    val innerdim = opts.indim;
    val dim =      opts.dim;
    val degree =   opts.degree;
    val basedims = irow(opts.dim, seqlength);
    val headdims = irow(opts.dim/opts.nheads, opts.nheads, degree, seqlength/degree);
    val headperm = irow(0,2,1,3);
    val invheadperm = headperm;
    val headperm2 = irow(2,0,1,3);
    val hasBias  = opts.hasBias;

    val cmask0 =   zeros((degree*2) \ degree \ opts.nheads \ (seqlength/degree));
    val col = icol(0->degree);
    for (i <- 0 until seqlength/degree) { 
      for (j <- 0 until opts.nheads) { 
        for (k <- 0 until degree) { 
          cmask0(k + 1 + col, k, j, i) = 1f;
        }
      }
    }
    val smask0 =   (1f - cmask0) *@ -1000f;

    val in_qkv =      input;
    val this_in =     colslice(in_qkv)(degree, seqlength+degree);
    val last_in =     colslice(in_qkv)(0, seqlength);
    val cmask =       constant(cmask0);
    val smask =       constant(smask0);

    val proj_q_this = linear(this_in)(outdim=innerdim, hasBias=hasBias);
    val proj_k_this = linear(this_in)(outdim=innerdim, hasBias=hasBias);
    val proj_v_this = linear(this_in)(outdim=innerdim, hasBias=hasBias);   
    val proj_k_last = linear(last_in)(outdim=innerdim, hasBias=hasBias);
    val proj_v_last = linear(last_in)(outdim=innerdim, hasBias=hasBias);   

    val rqueries =    reshape(proj_q_this)(headdims,false);
    val rkeys_this =  reshape(proj_k_this)(headdims,false);
    val rvals_this =  reshape(proj_v_this)(headdims,false);
    val rkeys_last =  reshape(proj_k_last)(headdims,false);
    val rvals_last =  reshape(proj_v_last)(headdims,false);

    val queries =     transpose(rqueries)(headperm);
    val keys_this =   transpose(rkeys_this)(headperm2);
    val vals_this =   transpose(rvals_this)(headperm2);
    val keys_last =   transpose(rkeys_last)(headperm2);
    val vals_last =   transpose(rvals_last)(headperm2);

    val keys =        keys_last over keys_this;
    val vals =        vals_last over vals_this;
    val prod =        keys * queries;
    val mprod =       prod *@ cmask;
    val oprod =       prod + smask;

    val weights =     softmaxx(oprod)();
    val wvals =       vals ^* weights;
    val pvals =       transpose(wvals)(invheadperm);
    val rpvals =      reshape(pvals)(basedims,false);
    val mhattn =      linear(rpvals)(outdim=dim, hasBias=hasBias);

    val norm1 =       layerNorm(mhattn)();
    val sum1 =        norm1 + this_in;
    val ffwd1 =       linear(sum1)(outdim=dim, hasBias=hasBias);
    val relu1 =       relu(ffwd1)();
    val norm2 =       layerNorm(relu1)();
    val sum2 =        sum1 + norm2;
    
    nopts.nodemat = in_qkv       \ this_in      \ last_in      \ cmask        \ smask        on
                    proj_q_this  \ proj_k_this  \ proj_v_this  \ proj_k_last  \ proj_v_last  on
                    rqueries     \ rkeys_this   \ rvals_this   \ rkeys_last   \ rvals_last   on
                    queries      \ keys_this    \ vals_this    \ keys_last    \ vals_last    on
                    keys         \ vals         \ prod         \ mprod        \ oprod        on
                    weights      \ wvals        \ pvals        \ rpvals       \ mhattn       on
                    norm1        \ sum1         \ ffwd1        \ relu1        \ norm2        on 
                    sum2         \ null         \ null         \ null         \ null;
 
    net.output_nodes = Array(sum2);
    net.createLayers;
    net
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
  
  def load(fname:String):TransformerLT = {
  	val mm = new TransformerLT;
  	mm.loadMetaData(fname);
  	mm.load(fname);
  	mm
  }

}

