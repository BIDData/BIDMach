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
  var posMat:FMat = null;
  
  val kmodels = 6
  var cacheState = false;
  var cacheGPUstate = false;
  var useCache = false;
  var useGPUCache = true;
  var seqptr = 0;
  var batchSize = -1;

  var linear0_nodenum = 0
  var linear5_nodenum = 0
  var linear6_nodenum = 0
  var linear7_nodenum = 0
  var be_model_nodenum = 0;
  var fe_model_nodenum = 0;
  var step = 0L

  override def init() = {
    useGPU = opts.useGPU && Mat.hasCUDA > 0;
    useDouble = opts.useDouble;
    cacheState = Mat.useCache;
    Mat.useCache = useCache;
    cacheGPUstate = Mat.useGPUcache;
    Mat.useGPUcache = useGPUCache;
    seqptr = 0;

    createTables();
    createModelmats();

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
        step += 1;
      }
    }
  }

  override def evalbatch(gmats:Array[Mat], ipass:Int, pos:Long):FMat = {
    if (batchSize < 0) batchSize = gmats(0).ncols;
    if (batchSize == gmats(0).ncols) {                       // discard odd-sized minibatches
      assignInputsAndTargets(gmats, ipass, pos);
      if (lastScores.asInstanceOf[AnyRef] == null) lastScores = zeros(backEnd.score_layers.length, batchSize);
      if (seqptr >= opts.seqlength) { 
        forward(true);
        wrapInput();
        seqptr = 0;
        step += 1;
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

    if (!opts.useRelPos) { 
      TransformerLT.posEncoding(pos, posMat, opts.posMagnitude, opts.posScale);
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
    for (i <- 0 to opts.depth) { 
      table(i) = convertMat(zeros(opts.dim, opts.seqlength + opts.degree));
      dtable(i) = convertMat(zeros(opts.dim, opts.seqlength + opts.degree));
    }
    posMat = zeros(opts.dim, opts.seqlength + opts.degree);
  }

  def createModelmats() { 
    setmodelmats(new Array[Mat](opts.depth * kmodels * 2 + 4));
    updatemats = new Array[Mat](opts.depth * kmodels * 2 + 4);
    for (i <- 0 until opts.depth) { 

      // Query, Key, Value embedding model matrices
      for (j <- 0 until 3) {     
        val m0 = convertMat(normrnd(0, 1/math.sqrt(opts.dim).toFloat, opts.indim, opts.dim));
        modelmats(2 * (j + kmodels * i)) = m0;
        modelmats(2 * (j + kmodels * i) + 1) = convertMat(zeros(m0.nrows, 1));             // Bias vector
        updatemats(2 * (j + kmodels * i)) = convertMat(zeros(m0.dims))                     // Matches model dims
        updatemats(2 * (j + kmodels * i) + 1) = convertMat(zeros(m0.nrows, 1));            // Bias update vector
      }

      // MHattn linear map matrices
      val m1 = convertMat(normrnd(0, 1/math.sqrt(opts.indim).toFloat, opts.dim, opts.indim));
      modelmats(2 * (3 + kmodels * i)) = m1
      modelmats(2 * (3 + kmodels * i) + 1) = convertMat(zeros(m1.nrows, 1));
      updatemats(2 * (3 + kmodels * i)) = convertMat(zeros(m1.dims));
      updatemats(2 * (3 + kmodels * i) + 1) = convertMat(zeros(m1.nrows, 1));

      // First feedforward net matrices
      val m2 = convertMat(normrnd(0, 1/math.sqrt(opts.dim).toFloat, opts.outdim, opts.dim));
      modelmats(2 * (4 + kmodels * i)) = m2;
      modelmats(2 * (4 + kmodels * i) + 1) = convertMat(zeros(m2.nrows, 1));
      updatemats(2 * (4 + kmodels * i)) = convertMat(zeros(m2.dims));
      updatemats(2 * (4 + kmodels * i) + 1) = convertMat(zeros(m2.nrows, 1));

      // Second feedforward net matrices
      val m3 = convertMat(normrnd(0, 1/math.sqrt(opts.dim).toFloat, opts.dim, opts.outdim));
      modelmats(2 * (5 + kmodels * i)) = m3;
      modelmats(2 * (5 + kmodels * i) + 1) = convertMat(zeros(m3.nrows, 1));
      updatemats(2 * (5 + kmodels * i)) = convertMat(zeros(m3.dims))
      updatemats(2 * (5 + kmodels * i) + 1) = convertMat(zeros(m3.nrows, 1));
    }
    // Front-end model matrices
    val m4 = convertMat(normrnd(0, opts.posMagnitude, opts.dim, opts.nvocab));
    modelmats(2 * kmodels * opts.depth) = m4;
    modelmats(2 * kmodels * opts.depth + 1) = convertMat(zeros(m4.nrows, 1));
    updatemats(2 * kmodels * opts.depth) = convertMat(zeros(m4.dims));
    updatemats(2 * kmodels * opts.depth + 1) = convertMat(zeros(m4.nrows, 1));

    // Back-end model matrices
    val m5 = convertMat(normrnd(0, 1/math.sqrt(opts.dim).toFloat, opts.nvocab, opts.dim));
    modelmats(2 * kmodels * opts.depth + 2) = m5;
    modelmats(2 * kmodels * opts.depth + 3) = convertMat(zeros(m5.nrows, 1));
    updatemats(2 * kmodels * opts.depth + 2) = convertMat(zeros(m5.dims));
    updatemats(2 * kmodels * opts.depth + 3) = convertMat(zeros(m5.nrows, 1));
  }

  def attachEnds() { 
    frontEnd.layers(frontEnd.layers.length-1).output = table(0);
    frontEnd.layers(frontEnd.layers.length-1).deriv = dtable(0);
    frontEnd.layers(fe_model_nodenum).asInstanceOf[ModelLayer].imodel = opts.depth * kmodels * 2;
    backEnd.layers(be_model_nodenum).asInstanceOf[ModelLayer].imodel = opts.depth * kmodels * 2 + 2;
  }

  def attach(net:Net, level:Int = 0) { 
    net.layers(0).output = table(level);
    net.layers(0).deriv = dtable(level);
    net.layers(linear0_nodenum).asInstanceOf[ModelLayer].imodel = level * kmodels * 2;
    net.layers(linear0_nodenum+1).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 2;
    net.layers(linear0_nodenum+2).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 4;
    net.layers(linear0_nodenum+3).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 2;
    net.layers(linear0_nodenum+4).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 4;
    net.layers(linear5_nodenum).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 6;
    net.layers(linear6_nodenum).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 8;
    net.layers(linear7_nodenum).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 10;
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
      setseed((5434*level+2354*step).toInt);    // Needed for dropout to be consistent
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
    backEnd.layers(backEnd.layers.length-1).deriv.set(1f);
    backEnd.backward();
    backEnd.layers(0).deriv.colslice(0, opts.seqlength, dtable(opts.depth), opts.degree);
    for (level <- (opts.depth -1) to 0 by -1) { 
      attach(net, level);
      setseed((5434*level+2354*step).toInt);
      net.forward;
      dtable(level+1).colslice(opts.degree, opts.seqlength+opts.degree, net.layers(net.layers.length-1).deriv, 0)
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
    Net.initDefaultNodeSet;

    val in =           input;
    val smat =         oneHot(in)(opts.nvocab);
    fe_model_nodenum = Net.getDefaultNodeNum
    val embed =        linear(smat)(outdim=dim, hasBias=hasBias);
    val posenc =       constant(posMat)(false);
    val out =          embed + posenc;

    nopts.nodeset =    Net.getDefaultNodeSet
 
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
    Net.initDefaultNodeSet;

    val in =           input;
    be_model_nodenum = Net.getDefaultNodeNum
    val prod =         linear(in)(outdim=opts.nvocab, hasBias=false);
    val out =          softmaxout(prod)(scoreType=opts.scoreType, lossType=1);

    nopts.nodeset =    Net.getDefaultNodeSet;
 
    net.createLayers;
    net;
  }


  def createTxNet(seqlength:Int) = {
    val net = new Net();
    val nopts = net.opts;
    nopts.useGPU = useGPU;
    net.useGPU = useGPU;

    import BIDMach.networks.layers.Node._
    val indim = opts.indim;
    val dim =      opts.dim;
    val degree =   opts.degree;
    val basedims = irow(dim, seqlength);
    val basedimsi = irow(indim, seqlength);
    val headdims = irow(indim/opts.nheads, opts.nheads, degree, seqlength/degree);
    val headdims2 = irow(indim/opts.nheads, opts.nheads, degree*2, seqlength/degree);
    val headdims_2d = irow(indim/opts.nheads, opts.nheads * seqlength);
    val headdims2_2d = irow(indim/opts.nheads, opts.nheads * seqlength * 2);
    val headdimsx = irow(indim/opts.nheads, degree, opts.nheads, seqlength/degree);
    val headdimsx2 = irow(indim/opts.nheads, degree*2, opts.nheads, seqlength/degree);
    val headinds__ = irow(0->(opts.nheads * seqlength)).reshapeView(opts.nheads, degree, seqlength/degree);
    val headinds_ = headinds__.transpose(1\0\2).reshapeView(1,opts.nheads*seqlength);
    val headinds2__ = irow(0->(opts.nheads * seqlength * 2)).reshapeView(opts.nheads, degree*2, seqlength/degree);
    val headinds2_ = headinds2__.transpose(1\0\2).reshapeView(1,opts.nheads*seqlength*2);
    val invheadinds__ = irow(0->(opts.nheads * seqlength)).reshapeView(degree, opts.nheads, seqlength/degree);
    val invheadinds_ = invheadinds__.transpose(1\0\2).reshapeView(1,opts.nheads*seqlength);
    val headperm2 = irow(2,0,1,3);
    val hasBias  = opts.hasBias;

    val cmask_ =  zeros((degree*2) \ degree \ opts.nheads \ (seqlength/degree));
    val col = icol(0->degree);
    for (i <- 0 until seqlength/degree) { 
      for (j <- 0 until opts.nheads) { 
        for (k <- 0 until degree) { 
          cmask_(k + col + 1, k, j, i) = 1f
        }
      }
    }
    val smask_ =  (1f - cmask_) *@ -1e37f;
    if (opts.useRelPos) smask_ ~ smask_ + TransformerLT.getRelPos(cmask_, headScale=0.5f);
    val v = 1/math.sqrt(indim/opts.nheads).toFloat;
    cmask_ ~ cmask_ * v;

    Net.initDefaultNodeSet;

    // Split the input into current and previous degree blocks (to be stacked later). Apply posencoding
    // but not to the residual link this_in
    val in_qkv =      input;
    val this_in_nopos=colslice(in_qkv)(degree, seqlength+degree);
//    val last_in =     colslice(in_qkv)(0, seqlength);
//    val posenc =      constant(posMat)(false);
//    val in_qkv_pos =  in_qkv + posenc
    val in_qkv_pos =  in_qkv
    val this_in =     colslice(in_qkv_pos)(degree, seqlength+degree);
    val last_in =     colslice(in_qkv_pos)(0, seqlength);

    // Query/Key/Value embedding
    linear0_nodenum = Net.getDefaultNodeNum
    val proj_q_this = linear(this_in)(outdim=indim, hasBias=hasBias);
    val proj_k_this = linear(this_in)(outdim=indim, hasBias=hasBias);
    val proj_v_this = linear(this_in)(outdim=indim, hasBias=hasBias);   
    val proj_k_last = linear(last_in)(outdim=indim, hasBias=hasBias);
    val proj_v_last = linear(last_in)(outdim=indim, hasBias=hasBias);   

    // Reshape queries and keys. Keys and Vals are reshaped differently so they can be stacked to 2*degree height
    val queries_2d =  reshape(proj_q_this)(headdims_2d,false);
    val keys_this =   reshape(proj_k_this)(headdims,false);
    val vals_this =   reshape(proj_v_this)(headdims,false);
    val keys_last =   reshape(proj_k_last)(headdims,false);
    val vals_last =   reshape(proj_v_last)(headdims,false);

    // Now stack keys and values.
    val keys =        stack(keys_last, keys_this)(2);
    val vals =        stack(vals_last, vals_this)(2);
    val keys_2d =     reshape(keys)(headdims2_2d,false);
    val vals_2d =     reshape(vals)(headdims2_2d,false);

    // Now transpose from (dim/nheads, nheads, degree, n) to (dim/nheads, degree, nheads, n) using colperm
    val headinds =    constant(headinds_)(true);
    val headinds2 =   constant(headinds2_)(true);
    val queriesx_2d = colperm(queries_2d, headinds);
    val keysx_2d =    colperm(keys_2d, headinds2); 
    val valsx_2d =    colperm(vals_2d, headinds2);
    val queriesx =    reshape(queriesx_2d)(headdimsx,false);
    val keysx =       reshape(keysx_2d)(headdimsx2,false);
    val valsx =       reshape(valsx_2d)(headdimsx2,false);

    // Query/Key products and masking
    val prod =        keysx ^* queriesx;
    val cmask =       constant(cmask_)(true);
    val smask =       constant(smask_)(true);
    val mprod =       prod *@ cmask;
    val oprod =       mprod + smask;

    // Apply softmax, then apply attention to the values.
    val weights =     softmaxx(oprod)();
    val wvals =       valsx * weights;
    val wvals_2d =    reshape(wvals)(headdims_2d,false);
    val invheadinds = constant(invheadinds_)(true);
    val pvals_2d =    colperm(wvals_2d, invheadinds);
    val pvals =       reshape(pvals_2d)(basedimsi,false);

    // Apply output embedding to the attention-weighted values
    linear5_nodenum = Net.getDefaultNodeNum;
    val mhattn =      linear(pvals)(outdim=dim, hasBias=hasBias); 
    val drop1 =       dropout(mhattn)(opts.dropout)
    val sum1 =        drop1 + this_in_nopos;
    val norm1 =       layerNorm(sum1)();
//    val norm1 =       layerNorm(drop1)();
//    val sum1 =        norm1 + this_in_nopos;

    // Feedforward output layer
    linear6_nodenum = Net.getDefaultNodeNum
    val ffwd1 =       linear(norm1)(outdim=opts.outdim, hasBias=true);
//    val ffwd1 =       linear(sum1)(outdim=opts.outdim, hasBias=true);
    val relu1 =       relu(ffwd1)();
    linear7_nodenum = Net.getDefaultNodeNum
    val ffwd2 =       linear(relu1)(outdim=dim, hasBias=true);
    val drop2 =       dropout(ffwd2)(opts.dropout)
    val sum2 =        norm1 + drop2;
    val norm2 =       layerNorm(sum2)();

//    val norm2 =       layerNorm(drop2)();
//    val sum2 =        sum1 + norm2;

    nopts.nodeset =   Net.getDefaultNodeSet

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
    var outdim = 2048;
    var degree = 128;
    var nheads = 8;
    var depth = 32;
    var stride = 4;
    var firststrided = 10;
    var nstrided = 6;
    var nvocab = 32768;
    var hasBias = false;
    var scoreType = SoftmaxOutputLayer.CrossEntropyScore
    var PADsym = 1;      // Padding symbol
    var OOVsym = 2;      // OOV symbol
    var STARTsym = 0;    // Start symbol
    var dropout = 0.9f;
    var useRelPos = false;
    var posEvery = true;
    var posScale = 1f;
    var posMagnitude = 1f;
  }


  def posEncoding(startpos:Long, mat:FMat, outScale:Float=1f, posScale:Float=1f, maxr:Float=10000) = { 
    val d = mat.nrows;
    val n = mat.ncols;
    val p = 0.883f;
    val pos = DMat(row(0->n)) + startpos.toDouble;
    for (i <- 0 until d/2) { 
//      val rate = math.pow(maxr, -i*2.0/d) * posScale;
      val rate = math.pow((i*2.0+1)/d, 1.0/(1.0-p)/posScale) * 0.54;
      mat(i*2, ?) = FMat(sin(pos * rate)) * outScale;
      mat(i*2+1, ?) = FMat(cos(pos * rate)) * outScale;
    }
    mat
  }

  def getRelPos(cmask:FMat, exponent:Float=0.883f, headScale:Float=0):FMat = { 
    val cd = cmask.dims;
    val mat = zeros(cd);
    val degree = cd(1);
    val nheads = cd(2);
    val len = cd(3);
    for (ii <- 0 until len) { 
      for (jj <- 0 until nheads) { 
        for (i <- 0 until degree) { 
          for (j <- 0 until degree*2) { 
            val dt = degree - j + i;
            if (cmask(j, i, jj, ii) != 0) {
              mat(j, i, jj, ii) = math.pow(1 + dt, (exponent-1) * (1 + headScale*jj)).toFloat;
            }
          }
        }
      }
    }
    mat
  }

  
@SerialVersionUID(100L)
  class Options extends Opts {}
  
@SerialVersionUID(100L)
  class LearnOptions extends Learner.Options with TransformerLT.Opts with MatSource.Opts with ADAGrad.Opts

@SerialVersionUID(100L)
  class FSopts extends Learner.Options with TransformerLT.Opts with FileSource.Opts with ADAGrad.Opts


  def learner(mat0:Mat, useADAGrad:Boolean) = {
    val opts = new LearnOptions;
    opts.batchSize = 128;
  	val nn = new Learner(
  	    new MatSource(Array(mat0), opts), 
  	    new TransformerLT(opts), 
        null,
  	    if (useADAGrad) { new ADAGrad(opts) } else { new Grad(opts) }, 
  	    null,
  	    opts)
    (nn, opts)
  }

  def learner(mat0:Mat):(Learner, LearnOptions) = learner(mat0, true);

  def learner(fnames:List[(Int)=>String], useADAGrad:Boolean) = {
    val opts = new FSopts;
    opts.fnames = fnames
  	val nn = new Learner(
  	    new FileSource(opts),
  	    new TransformerLT(opts), 
        null,
  	    if (useADAGrad) { new ADAGrad(opts) } else { new Grad(opts) }, 
  	    null,
  	    opts)
    (nn, opts)
  }

  def learner(fn1:String, useADAGrad:Boolean):(Learner, FSopts) = learner(List(FileSource.simpleEnum(fn1,1,0)), useADAGrad)

  def learner(fn1:String):(Learner, FSopts) = learner(List(FileSource.simpleEnum(fn1,1,0)), true)

  
  
  def load(fname:String):TransformerLT = {
  	val mm = new TransformerLT;
  	mm.loadMetaData(fname);
  	mm.load(fname);
  	mm
  }

}
