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
class TransformerLT(override val opts:TransformerLT.Opts = new TransformerLT.Options) extends Net(opts) {
  
  var table:Array[Mat] = null;
  var dtable:Array[Mat] = null;
  
  def createTables(opts:TransformerLT.Opts) { 
    table = new Array[Mat](opts.depth);
    dtable = new Array[Mat](opts.depth);
    for (i <- 0 until opts.depth) { 
      table(i) = convertMat(zeros(opts.dim, opts.seqlength));
      dtable(i) = convertMat(zeros(opts.dim, opts.seqlength));
    }
  }


  /**
   * TODO: Modify 
   */

  def constructNet(nqueries:Int, nvals:Int) = {
    import BIDMach.networks.layers.Node._
    val innerdim = opts.indim;
    val dim =      opts.dim;
    val headdims = irow(opts.dim/opts.nheads, opts.nheads, nqueries);
    val headdims2 = irow(opts.dim/opts.nheads, opts.nheads, nvals);
    val headperm = irow(0,2,1);
    val hasBias  = opts.hasBias;

    val cmask0 =   zeros(nvals, nqueries);
    val col = icol(0->nqueries);
    for (i <- 0 until nqueries) { 
      cmask0(i + col, i) = 1;
    }
    val smask0 =   (1f - cmask0) *@ -100f;
    
    val in_q =     input;
    val in_k =     input;
    val in_v =     input;

    val cmask =    constant(cmask0);
    val smask =    constant(smask0);

    val proj_q =   linear(in_q)(outdim=innerdim, hasBias=hasBias);
    val proj_k =   linear(in_k)(outdim=innerdim, hasBias=hasBias);
    val proj_v =   linear(in_v)(outdim=innerdim, hasBias=hasBias);   

    val rqueries = reshape(proj_q)(headdims,false);
    val rkeys =    reshape(proj_k)(headdims2,false);
    val rvalues =  reshape(proj_v)(headdims2,false);

    val queries =  transpose(rqueries)(headperm);
    val keys =     transpose(rkeys)(headperm);
    val values =   transpose(rvalues)(headperm);

    val prod =     keys ^* queries;
    val mprod =    prod *@ cmask;
    val oprod =    prod + smask;
    val weights =  softmaxx(oprod)();
    val wvals =    values * weights;

    val pvals =    transpose(wvals)(headperm);
    val mhattn =   linear(pvals)(outdim=dim, hasBias=hasBias);
    val sum1 =     mhattn + in_q;
    val norm1 =    layerNorm(sum1)();

    val ffwd1 =    linear(wvals)(outdim=dim, hasBias=hasBias);
    val relu1 =    relu(ffwd1)();
    val sum2 =     relu1 + norm1;
    val norm2 =    layerNorm(sum2)();
    
    val grid =     in_q       \ in_v      \ in_k      \ cmask    on
                   smask      \ proj_q    \ proj_k    \ proj_v   on
                   rqueries   \ rkeys     \ rvalues   \ null     on
                   queries    \ keys      \ values    \ prod     on
                   mprod      \ oprod     \ weights   \ wvals    on
                   pvals      \ mhattn    \ sum1      \ norm1    on
                   ffwd1      \ relu1     \ sum2      \ norm2;
  }

  override def assignInputs(gmats:Array[Mat], ipass:Int, pos:Long) { 
  }

  override def assignTargets(gmats:Array[Mat], ipass:Int, pos:Long) {
  } 
  
  override def dobatch(gmats:Array[Mat], ipass:Int, pos:Long):Unit = {
    if (batchSize < 0) batchSize = gmats(0).ncols;
    if (batchSize == gmats(0).ncols) {                                    // discard odd-sized minibatches
      assignInputs(gmats, ipass, pos);
      assignTargets(gmats, ipass, pos);
    }
  }
  
  override def evalbatch(mats:Array[Mat], ipass:Int, pos:Long):FMat = {  
    if (batchSize < 0) batchSize = gmats(0).ncols;
    if (batchSize == gmats(0).ncols) { 
      assignInputs(gmats, ipass, pos);
    }
    zeros(1, 1);
  }
}

@SerialVersionUID(100L)
object TransformerLT {
  trait Opts extends Net.Opts {
    var seqlength = 16384;
    dim = 512;
    var degree = 128;
    var indim = 512;
    var nheads = 8;
    var depth = 32;
    var stride = 4;
    var firststrided = 10;
    var nstrided = 6;
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

