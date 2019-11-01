package BIDMach.networks
// resScale now computes a convex combination

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
  var allScores:FMat = null;
  var inData:Mat = null;
  var linkMask:IMat = null;

  // Masking matrices
  var convMask:FMat = null;
  var fullMask:FMat = null;
  var nfullMask:FMat = null;
  var maskRowInds:IMat = null;
  var maskColInds:IMat = null;

  // Pos encoding matrices
  var posMat:FMat = null;
  var encBase:FMat = null;
  var encRates:DMat = null;
  var encPeriods:DMat = null;
  var encStart:DMat = null;
  var encTheta:DMat = null;

  
//  val kmodels = 6
  val kmodels = 8
  var cacheState = false;
  var cacheGPUstate = false;
  var useCache = false;
  var useGPUCache = true;
  var batchSize = -1;
  var lastPos = -1L;

  var linear0_nodenum = 0
  var linear5_nodenum = 0
  var linear6_nodenum = 0
  var linear7_nodenum = 0
  var scale1_nodenum = 0
  var scale2_nodenum = 0
  var be_model_nodenum = 0;
  var fe_model_nodenum = 0;
  var step = 0L

  var updateTime = 0.0
  var posEncTime = 0.0

  var time1 = 0.0
  var time2 = 0.0
  var time3 = 0.0
  var time4 = 0.0

  var time5 = 0.0
  var time6 = 0.0
  var time7 = 0.0

  override def init() = {
    useGPU = opts.useGPU && Mat.hasCUDA > 0;
    useDouble = opts.useDouble;
    cacheState = Mat.useCache;
    Mat.useCache = useCache;
    cacheGPUstate = Mat.useGPUcache;
    Mat.useGPUcache = useGPUCache;
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
    if (table.asInstanceOf[AnyRef] == null) createTables();
    if (batchSize == gmats(0).ncols) {                       // discard odd-sized minibatches
      val t1 = toc
      assignInputs(gmats);
      val t2 = toc
      forward(pos, false);
      val t3 = toc
      backward(pos);
      val t4 = toc
      wrapData(pos);
      val t5 = toc
      time1 += t2 - t1
      time2 += t3 - t2
      time3 += t4 - t3
      time4 += t5 - t4
    }
  }

  override def evalbatch(gmats:Array[Mat], ipass:Int, pos:Long):FMat = {
    if (batchSize < 0) batchSize = gmats(0).ncols;
    if (table.asInstanceOf[AnyRef] == null) createTables();
    if (batchSize == gmats(0).ncols) {                       // discard odd-sized minibatches
      val t1 = toc
      assignInputs(gmats);
      val t2 = toc
      forward(pos, true);
      val t3 = toc
      wrapData(pos);
      val t4 = toc
      time1 += t2 - t1
      time2 += t3 - t2
      time4 += t4 - t3
      allScores
    } else { 
      zeros(backEnd.score_layers.length, 1);
    }
  }

  def assignInputs(gmats:Array[Mat]) { 
    val src = gmats(0);
    if (batchSize % opts.seqlength != 0) { 
      throw new RuntimeException("TransformerLT batch size must be a multiple of sequence length");
    }
    src ~ src - ((src >= opts.nvocab) ∘ (src - opts.OOVsym)) // Map OOV words to the OOV symbol
    src.colslice(0, batchSize, inData, opts.degree + 1);
  }

  def wrapData(pos:Long) = { 
    if (pos != lastPos) { 
      val copyDataEnd = inData.colslice(batchSize, batchSize + opts.degree + 1); // Wrap overlapping input
      copyDataEnd.colslice(0, opts.degree + 1, inData, 0);
      for (level <- 0 to opts.depth) { 
        val copyEnd = table(level).colslice(batchSize, batchSize + opts.degree); // Wrap overlapping table data
        copyEnd.colslice(0, opts.degree, table(level), 0);
      }
      lastPos = pos
    }
  }

  def forwardFrontEnd(pos:Long, predicting:Boolean) = {
    val inlayer = frontEnd.layers(0)
    if (inlayer.output.asInstanceOf[AnyRef] == null) inlayer.output = convertMat(izeros(1, opts.seqlength+opts.degree))
    var tmppred = frontEnd.predicting;
    frontEnd.predicting = predicting;
    val inmat = frontEnd.layers(0).output

    var ipos = 0;
    while (ipos < batchSize) { 
      inData.colslice(ipos, ipos + opts.seqlength + opts.degree, inmat, 0);
//      TransformerLT.posEncoding(pos + ipos, posMat, opts.posMagnitude, opts.posScale);
      posEncoding(pos + ipos);
      frontEnd.forward
      val outmat = frontEnd.layers(frontEnd.layers.length-1).output
      outmat.colslice(0, opts.seqlength + opts.degree, table(0), ipos)
      ipos += opts.seqlength;
    }
    frontEnd.predicting = tmppred;
  }

  def forwardMainNet(pos:Long, predicting:Boolean, level:Int) = {
    val net = txNets(0);
    val inlayer = net.layers(0)
    if (inlayer.output.asInstanceOf[AnyRef] == null) {
      inlayer.output = convertMat(zeros(opts.dim, opts.seqlength+opts.degree))
      inlayer.deriv = convertMat(zeros(opts.dim, opts.seqlength+opts.degree))
    }
    var tmppred = net.predicting;
    net.predicting = predicting;
    val inmat = net.layers(0).output;
    val indata = table(level);
    val outdata = table(level+1);

    var ipos = 0;
    while (ipos < batchSize) {
      indata.colslice(ipos, ipos + opts.seqlength + opts.degree, inmat, 0);
      if (opts.updateMasks) { 
        inData.colslice(ipos, ipos + opts.seqlength + opts.degree, frontEnd.layers(0).output, 0);
        updateMasks(frontEnd.layers(0).output);
      }
      net.forward
      val outmat = net.layers(net.layers.length-1).output
      if (linkMask.asInstanceOf[AnyRef] == null || linkMask(level+1,1) == 0) { 
	    outmat.colslice(0, opts.seqlength, outdata, ipos + opts.degree)
      } else { 
	    val tmp = table(linkMask(level+1,1)).colslice(ipos + opts.degree, ipos + opts.degree + opts.seqlength);
        tmp ~ tmp *@ opts.resScale
	    tmp ~ tmp + (outmat *@ (1f - opts.resScale))
	    tmp.colslice(0, opts.seqlength, outdata, ipos + opts.degree)
      }
      ipos += opts.seqlength;
    }
    net.predicting = tmppred;
  }

  def forwardBackEnd(predicting:Boolean) = { 
    var tmppred = backEnd.predicting;
    backEnd.predicting = predicting;

    val backin = backEnd.layers(0)
    if (backin.output.asInstanceOf[AnyRef] == null) { 
      backin.output = convertMat(zeros(opts.dim, opts.seqlength))
      backin.deriv = convertMat(zeros(opts.dim, opts.seqlength))
    }

    val backout = backEnd.output_layers(0)
    if (backout.target.asInstanceOf[AnyRef] == null) backout.target = convertMat(izeros(1, opts.seqlength))
    val target = backout.target;

    var ipos = 0;
    while (ipos < batchSize) { 
      table(opts.depth).colslice(ipos + opts.degree, ipos + opts.seqlength + opts.degree, backin.output, 0);
      inData.colslice(ipos + opts.degree + 1, ipos + opts.seqlength + opts.degree + 1, target, 0);
      backEnd.forward
      if (allScores.asInstanceOf[AnyRef] == null) allScores = zeros(backEnd.score_layers.length, batchSize);
      for (i <- 0 until backEnd.score_layers.length) { 
        allScores(i,ipos->(ipos+opts.seqlength)) = backEnd.score_layers(i).score;
      }
      ipos += opts.seqlength;
    }
    backEnd.predicting = tmppred;
  }


  def backwardFrontEnd(pos:Long) = {
    val inmat = frontEnd.layers(0).output
    val outderiv = frontEnd.layers(frontEnd.layers.length-1).deriv
    val inderiv = frontEnd.layers(0).deriv
    var ipos = 0;

    while (ipos < batchSize) { 
//      TransformerLT.posEncoding(pos + ipos, posMat, opts.posMagnitude, opts.posScale);
      posEncoding(pos + ipos);
      inData.colslice(ipos, ipos + opts.seqlength + opts.degree, inmat, 0);
      frontEnd.forward
      dtable(0).colslice(ipos, ipos + opts.degree + opts.seqlength, outderiv, 0);
      frontEnd.backward();
      ipos += opts.seqlength;
    }
  }

  def backwardMainNet(pos:Long, level:Int) = {
    val net = txNets(0);
    val inmat = net.layers(0).output
    val outderiv = net.layers(net.layers.length-1).deriv
    val inderiv = net.layers(0).deriv
    val intable = table(level);
    val indtable = dtable(level);
    val outdtable = dtable(level+1);

    var ipos = 0;
    indtable.clear
    while (ipos < batchSize) { 
      if (opts.updateMasks) { 
        inData.colslice(ipos, ipos + opts.seqlength + opts.degree, frontEnd.layers(0).output, 0);
        updateMasks(frontEnd.layers(0).output);
      }
//      TransformerLT.posEncoding(pos + ipos, posMat, opts.posMagnitude, opts.posScale);
      intable.colslice(ipos, ipos + opts.seqlength + opts.degree, inmat, 0);
      net.forward
      outdtable.colslice(ipos + opts.degree, ipos + opts.degree + opts.seqlength, outderiv, 0);
      net.backward();
      val tmp = indtable.colslice(ipos, ipos + opts.degree + opts.seqlength);
      tmp ~ tmp + inderiv;
      tmp.colslice(0, opts.degree + opts.seqlength, indtable, ipos);
      ipos += opts.seqlength;
    }
    if (linkMask.asInstanceOf[AnyRef] != null && linkMask(level,0) > 0) { 
      indtable ~ indtable *@ (1f - opts.resScale)
      indtable ~ indtable + (dtable(linkMask(level,0)) *@ opts.resScale);
    }
  }

  def backwardBackEnd() = {
    val inmat = backEnd.layers(0).output
    val outderiv = backEnd.layers(backEnd.layers.length-1).deriv
    val inderiv = backEnd.layers(0).deriv
    val intable = table(opts.depth)
    val indtable = dtable(opts.depth)
    val target = backEnd.output_layers(0).target;

    var ipos = 0;
    indtable.clear
    while (ipos < batchSize) { 
      intable.colslice(ipos + opts.degree, ipos + opts.seqlength + opts.degree, inmat, 0);
      inData.colslice(ipos + opts.degree + 1, ipos + opts.seqlength + opts.degree + 1, target, 0);
      backEnd.forward
      outderiv.set(1f);
      backEnd.backward();
      inderiv.colslice(0, opts.seqlength, indtable, ipos + opts.degree);
      ipos += opts.seqlength;
    }
  }

  def createTables() { 
    table = new Array[Mat](opts.depth+1);
    dtable = new Array[Mat](opts.depth+1);
    for (i <- 0 to opts.depth) { 
      table(i) = convertMat(zeros(opts.dim, batchSize + opts.degree));
      dtable(i) = convertMat(zeros(opts.dim, batchSize + opts.degree));
    }
    inData = convertMat(izeros(1, batchSize + opts.degree + 1));
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

      // scaleNorm matrices
      val adims = irow(1,1);
      modelmats(2 * (6 + kmodels * i)) = convertMat(ones(adims)*opts.normInit);
      modelmats(2 * (6 + kmodels * i) + 1) = convertMat(zeros(adims));
      updatemats(2 * (6 + kmodels * i)) = convertMat(zeros(adims))
      updatemats(2 * (6 + kmodels * i) + 1) = convertMat(zeros(adims))

      modelmats(2 * (7 + kmodels * i)) = convertMat(ones(adims)*opts.normInit);
      modelmats(2 * (7 + kmodels * i) + 1) = convertMat(zeros(adims));
      updatemats(2 * (7 + kmodels * i)) = convertMat(zeros(adims))
      updatemats(2 * (7 + kmodels * i) + 1) = convertMat(zeros(adims))
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

    // Position encoding matrix
    posMat = convertMat(zeros(opts.dim, opts.seqlength + opts.degree)).asInstanceOf[FMat];

    // Mask matrices
    val (cm, ncm, cmask) = TransformerLT.boundaryMats(opts.degree, opts.nheads, opts.seqlength);
    fullMask = convertMat(zeros((opts.degree*2) \ opts.degree \ opts.nheads \ (opts.seqlength/opts.degree))).asInstanceOf[FMat];
    nfullMask = convertMat(zeros((opts.degree*2) \ opts.degree \ opts.nheads \ (opts.seqlength/opts.degree))).asInstanceOf[FMat];;
    maskRowInds = convertMat(cm).asInstanceOf[IMat];;
    maskColInds = convertMat(ncm).asInstanceOf[IMat];;
    convMask = convertMat(cmask).asInstanceOf[FMat];
    updateMasks(null);

    if (opts.resLinks.asInstanceOf[AnyRef] != null) { 
      linkMask = izeros(opts.depth + 1, 2);
      linkMask(opts.resLinks(?,0), 0) = opts.resLinks(?,1);
      linkMask(opts.resLinks(?,1), 1) = opts.resLinks(?,0);
    }
  }

  def updateMasks(inmat:Mat) { 
    val t1 = toc
    if (inmat.asInstanceOf[AnyRef] != null) { 
      val iinmat = float(inmat);
      val ndoc = cumsum(iinmat.t == opts.boundaryWord.toFloat).t;
      val cc = (ndoc(maskRowInds) == ndoc(maskColInds));
      fullMask ~ cc *@ convMask; 
    } else { 
      fullMask <-- convMask
    }
    nfullMask ~ fullMask *@ -1f;
    nfullMask ~ nfullMask + 1f;
    nfullMask ~ nfullMask *@ -1e37f;
    val v = 1/math.sqrt(opts.dim/opts.nheads).toFloat;
    fullMask ~ fullMask *@ v
    val t2 = toc;
    updateTime += t2 - t1;
  }

  def attachEnds() { 
//    frontEnd.layers(frontEnd.layers.length-1).output = table(0);
//    frontEnd.layers(frontEnd.layers.length-1).deriv = dtable(0);
    frontEnd.layers(fe_model_nodenum).asInstanceOf[ModelLayer].imodel = opts.depth * kmodels * 2;
    backEnd.layers(be_model_nodenum).asInstanceOf[ModelLayer].imodel = opts.depth * kmodels * 2 + 2;
  }

  def attach(net:Net, level:Int = 0) { 
//    net.layers(0).output = table(level);
//    net.layers(0).deriv = dtable(level);
    net.layers(linear0_nodenum).asInstanceOf[ModelLayer].imodel = level * kmodels * 2;
    net.layers(linear0_nodenum+1).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 2;
    net.layers(linear0_nodenum+2).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 4;
    net.layers(linear0_nodenum+3).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 2;
    net.layers(linear0_nodenum+4).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 4;
    net.layers(linear5_nodenum).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 6;
    net.layers(linear6_nodenum).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 8;
    net.layers(linear7_nodenum).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 10;
    net.layers(scale1_nodenum).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 12;
    net.layers(scale2_nodenum).asInstanceOf[ModelLayer].imodel = level * kmodels * 2 + 14;
  }

  def forward(pos:Long, predicting:Boolean=false) { 
    val net = txNets(0);
    step += 1;
    forwardFrontEnd(pos, predicting);
    for (level <- 0 until opts.depth) { 
      attach(net, level);
      setseed((5434*level+2354*step).toInt);
      forwardMainNet(pos, predicting, level);
    }
    forwardBackEnd(predicting);
  }

  def backward(pos:Long) { 
    val net = txNets(0);
    val t1 = toc
    backwardBackEnd();
    val t2 = toc
    for (level <- (opts.depth -1) to 0 by -1) { 
      attach(net, level);
      setseed((5434*level+2354*step).toInt);
      backwardMainNet(pos, level);
    }
    val t3 = toc
    backwardFrontEnd(pos);
    val t4 = toc
    time5 += t2 - t1  
    time6 += t3 - t2
    time7 += t4 - t3  
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

/*    val cmask_ =  zeros((degree*2) \ degree \ opts.nheads \ (seqlength/degree));
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
    cmask_ ~ cmask_ * v; */

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
    val cmask =       constant(fullMask)(false);
    val smask =       constant(nfullMask)(false);
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
    scale1_nodenum =   Net.getDefaultNodeNum
//    val norm1 =       scale(sum1)(modelDims=0\1)
    val norm1 =       autoNorm(sum1)(decay=opts.decay)
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
    scale2_nodenum =   Net.getDefaultNodeNum
//     val norm2 =       scale(sum2)(modelDims=0\1);
    val norm2 =       autoNorm(sum2)(decay=opts.decay);

    nopts.nodeset =   Net.getDefaultNodeSet

    net.createLayers;
    net;
  }

  def posEncoding(startpos:Long) = { 
    val t1 = toc
    val d = opts.dim
    val n = opts.seqlength + opts.degree;
    val p = 0.883f;
    // Create a table encBase of d * n angles for zero start position
    // and a table encRates of the rates
    if (encBase.asInstanceOf[AnyRef] == null) { 
      val baseFMat = zeros(d, n);
      val rateDMat = dzeros(d, 1);
      val posFMat = row(0->n);
      for (i <- 0 until d/2) { 
	    val rate0 = math.pow((i*2.0+1)/d, 1.0/(1.0-p)/opts.posScale) * 0.54;
//      val rate = math.pow(maxr, -i*2.0/d) * posScale;
	    val rate = if (rate0 > 1.0/opts.posDistance) rate0 else 0.0;
	    rateDMat(2*i, 0) = rate;
	    rateDMat(2*i+1, 0) = rate;
	    baseFMat(i*2, ?) = posFMat * rate.toFloat;
	    baseFMat(i*2+1, ?) = posFMat * rate.toFloat + (Math.PI/2).toFloat; // Add pi/2 to the angle for cos rows
      }
      encBase = convertMat(baseFMat).asInstanceOf[FMat];
      encRates = if (useGPU) GDMat(rateDMat) else rateDMat;
      encStart = if (useGPU) gdzeros(1,1) else dzeros(1,1);
    }
    encStart.set(startpos.toDouble);
    encTheta = encStart *@ encRates;   // Encode the start position as an angle for each embedding dim
    encTheta ~ encTheta - (trunc(encTheta *@ (0.5/Math.PI)) *@ (2 * Math.PI)) // Limit to (0, 2*pi)
    posMat ~ encBase + float(encTheta)     // Add the offset angles to the base angles
    sin(posMat, posMat);                    // Compute the sine
    if (opts.posMagnitude != 1f) posMat ~ posMat * opts.posMagnitude; // Scale if needed
    val t2 = toc
    posEncTime += t2 - t1
    posMat
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
    var decay = 0.999f;
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
    var posDistance = 10000.0;
    var updateMasks = true;
    var boundaryWord = 2;
    var normInit = 1f;
    var resLinks:IMat = null;
    var resScale:Float = 1f;
  }

  var posEncTime = 0.0

  def posEncoding(startpos:Long, mat:FMat, outScale:Float=1f, posScale:Float=1f, maxr:Float=10000) = { 
    val t1 = toc
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
    val t2 = toc
    posEncTime += t2 - t1
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

  def boundaryMats(degree:Int, nheads:Int, seqlength:Int):(IMat, IMat, FMat) = { 
    val len = seqlength / degree;
    val colmask = izeros((2*degree) \ degree \ nheads \ len);
    val rowmask = izeros((2*degree) \ degree \ nheads \ len);
    val cmask = zeros((2*degree) \ degree \ nheads \ len);
    val colseq = izeros((2*degree) \ 1 \ 1 \ 1)
    val rowseq = izeros((2*degree) \ 1 \ 1 \ 1)
    colseq(?) = icol(0->(2*degree));
    rowseq(?) = degree;
    val colvec = icol(0->degree);
    for (i <- 0 until len) { 
      for (j <- 0 until nheads) { 
        for (k <- 0 until degree) { 
          colmask(?, k, j, i) = colseq + (i * degree)
          rowmask(?, k, j, i) = rowseq + (k + i * degree)
          cmask(k + colvec + 1, k, j, i) = 1f
        }
      }
    }
    (colmask, rowmask, cmask)
  }

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
