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
  var txNets:Array[Net] = null;
  var frontEnd:Net = null;
  var backEnd:Net = null;
  var lastScores:FMat = null;
  
  val kmodels = 5
  var cacheState = false;
  var cacheGPUstate = false;
  var useCache = false;
  var useGPUCache = true;
  var seqptr = 0;
  var batchSize = -1;

  override def init() = {
	useGPU = opts.useGPU && Mat.hasCUDA > 0;
	useDouble = opts.useDouble;
    cacheState = Mat.useCache;
    Mat.useCache = useCache;
    cacheGPUstate = Mat.useGPUcache;
    Mat.useGPUcache = useGPUCache;
    seqptr = 0;

    createTables();

    frontEnd = createFrontEnd();
    frontEnd.setmodelmats(modelmats);
    frontEnd.updatemats = updatemats;

    backEnd = createBackEnd();
    backEnd.setmodelmats(modelmats);
    backEnd.updatemats = updatemats;

    attachEnds();

    val net = createTxNet(opts.seqlength);
    net.setmodelmats(modelmats);
    net.updatemats = updatemats;
    txNets = Array(net);

  }  
  
  def wrapUp() = { 
    Mat.useCache = cacheState;
    Mat.useGPUcache = cacheGPUstate;
  }

  override def dobatch(gmats:Array[Mat], ipass:Int, pos:Long):Unit = {
    if (batchSize < 0) batchSize = gmats(0).ncols;
    if (batchSize == gmats(0).ncols) {                       // discard odd-sized minibatches
      assignInputsAndTargets(gmats, ipass, pos);
      if (seqptr >= opts.seqlength) { 
        forward();
        backward();
        wrapInput();
        seqptr = 0;
      }
    }
  }

  override def evalbatch(gmats:Array[Mat], ipass:Int, pos:Long):FMat = {
    if (batchSize < 0) batchSize = gmats(0).ncols;
    if (batchSize == gmats(0).ncols) {                       // discard odd-sized minibatches
      assignInputsAndTargets(gmats, ipass, pos);
      if (lastScores.asInstanceOf[AnyRef] == null) lastScores = zeros(backEnd.score_layers.length, batchSize);
      if (seqptr == 0) { 
        forward(true);
        lastScores = zeros(backEnd.score_layers.length, batchSize);
  		for (i <- 0 until backEnd.score_layers.length) {
  		  lastScores(i,?) = backEnd.score_layers(i).score;
  		}
      }
      lastScores
    } else { 
      zeros(backEnd.score_layers.length, 1);
    }
  }

  def assignInputsAndTargets(gmats:Array[Mat], ipass:Int, pos:Long) = {
    val src = gmats(0);
    if (opts.seqlength % batchSize != 0) { 
      throw new RuntimeException("TransformerLT sequence length must be a multiple of batch size");
    }
    src ~ src - ((src >= opts.nvocab) ∘ (src - opts.OOVsym)) // Map OOV words to the OOV symbol

    val inlayer = frontEnd.layers(0)
    if (inlayer.output.asInstanceOf[AnyRef] == null) inlayer.output = convertMat(izeros(1, opts.seqlength+opts.degree))
    val inmat = inlayer.output
    if (seqptr + batchSize < opts.seqlength) { 
      src.colslice(0, batchSize, inmat, seqptr + opts.degree + 1);
    } else { 
      src.colslice(0, batchSize - 1, inmat, seqptr + opts.degree + 1);
    }

    val backin = backEnd.layers(0)
    if (backin.output.asInstanceOf[AnyRef] == null) { 
      backin.output = convertMat(zeros(opts.dim, opts.seqlength))
      backin.deriv = convertMat(zeros(opts.dim, opts.seqlength))
    }

    val backout = backEnd.output_layers(0)
    if (backout.target.asInstanceOf[AnyRef] == null) backout.target = convertMat(izeros(1, opts.seqlength))
    val target = backout.target;
    src.colslice(0, batchSize, target, seqptr);

    seqptr += batchSize;
  }
  
  def wrapInput() { 
    val target = backEnd.output_layers(0).target;
    val inmat = frontEnd.layers(0).output
    target.colslice(opts.seqlength - opts.degree - 1, opts.seqlength, inmat, 0);
  }

  def createTables() { 
    table = new Array[Mat](opts.depth+1);
    dtable = new Array[Mat](opts.depth+1);
    setmodelmats(new Array[Mat](opts.depth * kmodels * 2 + 4));
    updatemats = new Array[Mat](opts.depth * kmodels * 2 + 4);
    for (i <- 0 until opts.depth) { 
      table(i) = convertMat(rand(opts.dim, opts.seqlength + opts.degree));
      dtable(i) = convertMat(rand(opts.dim, opts.seqlength + opts.degree));
      for (j <- 0 until 3) { 
        modelmats(2 * (j + kmodels * i)) = convertMat(zeros(opts.indim, opts.dim));
        modelmats(2 * (j + kmodels * i) + 1) = convertMat(zeros(opts.indim, 1));
        updatemats(2 * (j + kmodels * i)) = convertMat(zeros(opts.indim, opts.dim));
        updatemats(2 * (j + kmodels * i) + 1) = convertMat(zeros(opts.indim, 1));
      }
      modelmats(2 * (3 + kmodels * i)) = convertMat(zeros(opts.dim, opts.indim));
      modelmats(2 * (3 + kmodels * i) + 1) = convertMat(zeros(opts.dim, 1));
      updatemats(2 * (3 + kmodels * i)) = convertMat(zeros(opts.dim, opts.indim));
      updatemats(2 * (3 + kmodels * i) + 1) = convertMat(zeros(opts.dim, 1));

      modelmats(2 * (4 + kmodels * i)) = convertMat(zeros(opts.dim, opts.dim));
      modelmats(2 * (4 + kmodels * i) + 1) = convertMat(zeros(opts.dim, 1));
      updatemats(2 * (4 + kmodels * i)) = convertMat(zeros(opts.dim, opts.dim));
      updatemats(2 * (4 + kmodels * i) + 1) = convertMat(zeros(opts.dim, 1));
    }
    modelmats(2 * kmodels * opts.depth) = convertMat(zeros(opts.dim, opts.nvocab));
    modelmats(2 * kmodels * opts.depth + 1) = convertMat(zeros(opts.dim, 1));
    modelmats(2 * kmodels * opts.depth + 2) = convertMat(zeros(opts.nvocab, opts.dim));
    modelmats(2 * kmodels * opts.depth + 3) = convertMat(zeros(opts.nvocab, 1));
    updatemats(2 * kmodels * opts.depth) = convertMat(zeros(opts.dim, opts.nvocab));
    updatemats(2 * kmodels * opts.depth + 1) = convertMat(zeros(opts.dim, 1));
    updatemats(2 * kmodels * opts.depth + 2) = convertMat(zeros(opts.nvocab, opts.dim));
    updatemats(2 * kmodels * opts.depth + 3) = convertMat(zeros(opts.nvocab, 1));
    table(opts.depth) = convertMat(rand(opts.dim, opts.seqlength + opts.degree));
    dtable(opts.depth) = convertMat(rand(opts.dim, opts.seqlength + opts.degree));
  }

  def attachEnds() { 
    frontEnd.layers(2).output = table(0);
    frontEnd.layers(2).deriv = dtable(0);
    frontEnd.layers(2).asInstanceOf[ModelLayer].imodel = opts.depth * kmodels * 2;
    backEnd.layers(1).asInstanceOf[ModelLayer].imodel = opts.depth * kmodels * 2 + 2;
  }

  def attach(net:Net, level:Int = 0) { 
    net.layers(0).output = table(level);
    net.layers(0).deriv = dtable(level);
    net.layers(5).asInstanceOf[ModelLayer].imodel = level * kmodels * 2;
    net.layers(6).asInstanceOf[ModelLayer].imodel = level * kmodels * 2  + 2;
    net.layers(7).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 4;
    net.layers(8).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 2;
    net.layers(9).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 4;
    net.layers(36).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 6;
    net.layers(39).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 8;
  }

  def forward(predicting:Boolean=false) { 
    var tmppred = frontEnd.predicting;
    frontEnd.predicting = predicting;
    frontEnd.forward;
    frontEnd.predicting = tmppred;
    val net = txNets(0);
    for (level <- 0 until opts.depth) { 
      attach(net, level);
      val tmppred = net.predicting;
      net.predicting = predicting;
      net.forward;
      net.predicting = tmppred;
      table(level+1).colslice(opts.seqlength, opts.seqlength+opts.degree, table(level+1), 0);
      net.layers(net.layers.length-1).output.colslice(0, opts.seqlength, table(level+1), opts.degree);
    }
    tmppred = backEnd.predicting;
    table(opts.depth).colslice(opts.degree, opts.seqlength+opts.degree, backEnd.layers(0).output, 0);
    backEnd.predicting = predicting;
    backEnd.forward;
    backEnd.predicting = tmppred;
  }

  def backward() { 
    val net = txNets(0);
    backEnd.backward();
    backEnd.layers(0).deriv.colslice(0, opts.seqlength, dtable(opts.depth), opts.degree);
    for (level <- (opts.depth -1) to 0 by -1) { 
      attach(net, level);
      dtable(level+1).colslice(opts.degree, opts.seqlength+opts.degree, net.layers(net.layers.length-1).deriv, 0)
      net.forward;
      net.backward();
    }
    frontEnd.backward();
  }

  def createFrontEnd() = {
    val net = new Net();
    val nopts = net.opts;
    nopts.useGPU = useGPU;
    net.useGPU = useGPU;
    val dim =       opts.dim;
    val hasBias =   opts.hasBias;

    import BIDMach.networks.layers.Node._

    val in =        input;
    val smat =      oneHot(in)(opts.nvocab);
    val out =       linear(smat)(outdim=dim, hasBias=hasBias);

    nopts.nodemat = in \ smat \ out;
 
    net.output_nodes = Array(out);
    net.createLayers;
    net;
  }

  def createBackEnd() = {
    val net = new Net();
    val nopts = net.opts;
    nopts.useGPU = useGPU;
    net.useGPU = useGPU;
    val dim =       opts.dim;
    val hasBias =   opts.hasBias;

    import BIDMach.networks.layers.Node._

    val in =        input;
    val prod =      linear(in)(outdim=opts.nvocab, hasBias=false);
    val out =       softmaxout(prod)(scoreType=1, lossType=1);

    nopts.nodemat = in \ prod \ out;
 
    net.output_nodes = Array(out);
    net.createLayers;
    net;
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
    var nvocab = 32768;
    var hasBias = true;
    var PADsym = 1;      // Padding symbol
    var OOVsym = 2;      // OOV symbol
    var STARTsym = 0;    // Start symbol
  }
  
@SerialVersionUID(100L)
  class Options extends Opts {}
  
@SerialVersionUID(100L)
  class LearnOptions extends Learner.Options with TransformerLT.Opts with MatSource.Opts with Grad.Opts

@SerialVersionUID(100L)
  class FSopts extends Learner.Options with TransformerLT.Opts with FileSource.Opts with Grad.Opts


  def learner(mat0:Mat, mat1:Mat) = {
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

  def learner(fnames:List[(Int)=>String]) = {
    val opts = new FSopts;
    opts.fnames = fnames
  	val nn = new Learner(
  	    new FileSource(opts),
  	    new TransformerLT(opts), 
        null,
  	    new Grad(opts), 
  	    null,
  	    opts)
    (nn, opts)
  }

  def learner(fn1:String):(Learner, FSopts) = learner(List(FileSource.simpleEnum(fn1,1,0)))

  def testsetup(opts:Opts = new Options):TransformerLT = { 
    val trans = new TransformerLT(opts);
    trans.init();
    trans;
  }

  def testfwd(trans:TransformerLT, n:Int) = { 
    for (i <- 0 until n) { 
      trans.txNets(0).forward
    }
  }

  def testbwd(trans:TransformerLT, n:Int) = { 
    for (i <- 0 until n) { 
      trans.txNets(0).forward
      trans.txNets(0).setderiv();
      trans.txNets(0).backward()
    }
  }

  def testbackward(trans:TransformerLT, n:Int) = { 
    for (i <- 0 until n) { 
      trans.forward();
      trans.backward();
    }
  }
  
  
  
  def load(fname:String):TransformerLT = {
  	val mm = new TransformerLT;
  	mm.loadMetaData(fname);
  	mm.load(fname);
  	mm
  }

}

