package BIDMach.networks

import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,Dict,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach._
import edu.berkeley.bid.CUMACH
import edu.berkeley.bid.CPUMACH
import jcuda.runtime.JCuda._
import scala.util.hashing.MurmurHash3
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import java.text.SimpleDateFormat
import java.util.Calendar
import java.io.DataOutputStream
import java.io.DataInputStream
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.InputStreamReader
import java.io.PrintWriter
import java.util.Scanner
import scala.concurrent.Future
import scala.concurrent.Await
import scala.concurrent.duration.Duration

/**
 * Fast Word2Vec implementation for CPU and GPU. Currently supports skip-gram models with negative sampling. 
 * 
 * The input is an IMat with 2 rows. Each column holds a word ID (top row) and the corresponding sentence ID (second row). 
 * Options are:
 - nskip(5) the size of the skip-gram window.
 - nneg(5) the number of negative samples. 
 - nreuse(5) the number of times to re-use negative samples.  
 - vocabSize(100000) the vocabulary size. The input matrix can contain larger word IDs, in which case those IDs are marked as OOV. 
 - wexpt(0.75f) the exponent for negative sample weighting.
 - wsample(1e-4f) frequent word sample factor.
 - headlen(10000) size of the smallest block of words for distributed model synchronization. 
 - iflip(false) true if word and sentence IDs flipped (sentence ID in first row, word ID in second). 
 - eqPosNeg(false) normalize positive and negative word weights in the likelihood. 
 - aopts:ADAGrad.Opts(null) set this to an ADAGRAD.Opts object to use integrated adagrad updates. 
 *
 * The code has the ability to build models larger than a single Java array, and bigger than a single node can store. 
 * These options control performance in the case of models that must be distributed across multiple arrays and/or multiple machines
 - maxArraySize(1024^3) the maximum size in words of a model array.
 - nHeadTerms(0) the size of the head of the model - these terms are not changed.
 - nSlices(1) Process (num) slices of the model on (num) nodes.
 - iSlice(0) which model slice are we processing on this node?
 */

class Word2Vec(override val opts:Word2Vec.Opts = new Word2Vec.Options) extends Model(opts) {
  
  var firstPos = -1L;
  var wordtab:Mat = null;
  var randpermute:Mat = null;
  var ubound:Mat = null;
  var minusone:Mat = null;
  var wordmask:Mat = null;
  var allones:Mat = null;
  var randwords:Mat = null;
  var randsamp:Mat = null;
  var retEvalPos:GMat = null;
  var retEvalNeg:GMat = null;
  var nfeats = 0;
  var ncols = 0;
  var expt = 0f;
  var vexp = 0f;
  var salpha = 0f;
  var maxCols = 0;
  var nmmats = 1;
  var fmm:Array[Array[Float]] = null;
  
  var ntimes = 12;
  var times:FMat = null;
  var delays:FMat = null;
  var log:ArrayBuffer[String] = null
  val dateFormat = new SimpleDateFormat("hh:mm:ss:SSS")

  
  def addTime(itime:Int, lasti:Int = -1) = {
    val t = toc
    times(itime) = t;
    if (itime > 0) {
    	delays(itime) += times(itime) - times(itime + lasti);
    } 
    val today = Calendar.getInstance().getTime()
    log += "Log: %s, GPU %d, event %d" format (dateFormat.format(today), if (useGPU) getGPU else 0, itime);
  }
  
  var test1:Mat = null;
  var test2:Mat = null;
  var test3:Mat = null;
  var test4:Mat = null;
  

  override def init() = {
    val mats = datasource.next;
	  nfeats = opts.vocabSize;
	  ncols = mats(0).ncols;
	  maxCols = opts.maxArraySize / opts.dim;
	  datasource.reset;
	  val actualFeats = opts.nHeadTerms + 1 + (nfeats - opts.nHeadTerms - 1) / opts.nSlices;   // Number of features on this node. 
	  nmmats = 1 + (actualFeats - 1)/maxCols;                                // number of model mats needed
	  println("nmmats = %d" format nmmats);
	  val offset = if (opts.dualMode) 1 else 0;
    if (refresh) {
      if (actualFeats <= maxCols) {
      	setmodelmats(new Array[Mat](2));
      	val mm0 = rand(opts.dim, actualFeats);
      	mm0 ~ mm0 - 0.5f;
      	mm0 ~ mm0 / opts.dim;
      	modelmats(0) = mm0;                                                    // syn0 - context model
      	modelmats(1) = zeros(opts.dim, actualFeats);                                // syn1neg - target word model
      } else {
        setmodelmats(new Array[Mat](2 * (nmmats + offset)));
        for (i <- 0 until nmmats) {
          val xfeats = if (i < nmmats - 1) maxCols else actualFeats - (nmmats - 1) * maxCols;
          val tmp = rand(opts.dim, xfeats);
          tmp ~ tmp - 0.5f;
          tmp ~ tmp / opts.dim;
        	modelmats(2 * (i + offset)) = tmp;             
        	modelmats(2 * (i + offset) + 1) = zeros(opts.dim, xfeats);
        }
        if (opts.dualMode) {
          modelmats(0) <-- modelmats(2).copy;
          modelmats(1) <-- modelmats(3).copy;
        }
      }
    }
    modelmats(0) = convertMat(modelmats(0));                                   // At most the first two will be GPU-based
    modelmats(1) = convertMat(modelmats(1)); 
    val nskip = opts.nskip;
    val nwindow = nskip * 2 + 1;
    val skipcol = icol((-nskip) to -1) on icol(1 to nskip)
    expt = 1f / (1f - opts.wexpt);
    wordtab = convertMat(max(0, min(ncols+1, iones(nwindow-1, 1) * irow(1 -> (ncols+1)) + skipcol)));  // Indices for convolution matrix
    wordmask = convertMat(skipcol * iones(1, ncols));                          // columns = distances from center word
    randpermute = convertMat(zeros(nwindow-1, ncols));                         // holds random values for permuting negative context words
    ubound = convertMat(zeros(1, ncols));                                      // upper bound random matrix
    minusone = convertMat(irow(-1));
    allones = convertMat(iones(1, ncols));
    randwords = convertMat(zeros(1, (1.01 * opts.nneg * nskip * ncols / opts.nreuse).toInt)); // generates random negative words
    randsamp = convertMat(zeros(1, ncols));                                    // For sub-sampling frequent words
    val gopts = opts.asInstanceOf[ADAGrad.Opts];
    vexp = gopts.vexp.v;
    salpha = opts.wsample * math.log(nfeats).toFloat;
    fmm = new Array[Array[Float]](modelmats.length);
    if (useGPU) {
      retEvalPos = GMat(1,1);
      retEvalNeg = GMat(1,1);
    } else {
      if (Mat.useMKL) {
        for (i <- 0 until modelmats.length) {
          fmm(i) = modelmats(i).asInstanceOf[FMat].data;
        }
      }
    }
    times = zeros(1, ntimes);
    delays = zeros(1, ntimes);
    log = ArrayBuffer();
  }
  
  def dobatch(gmats:Array[Mat], ipass:Int, pos:Long):Unit = {
    addTime(0);
    if (gmats(0).ncols == ncols) {
    	if (firstPos < 0) firstPos = pos;
    	val nsteps = 1f * pos / firstPos;
    	val gopts = opts.asInstanceOf[ADAGrad.Opts];
    	val lrate = gopts.lrate.dv.toFloat * math.pow(nsteps, - gopts.texp.dv).toFloat;
    	val (words, lb, ub, trandwords, goodwords) = wordMats(gmats, ipass, pos);

    	val lrpos = lrate.dv.toFloat;
    	val lrneg = if (opts.eqPosNeg) lrpos else lrpos/opts.nneg; 
    	if (opts.nSlices == 1 && nmmats == 1) {
    		procPositives(opts.nskip, words, lb, ub, modelmats(1), modelmats(0), lrpos, vexp);
    		addTime(8);   	
    		procNegatives(opts.nneg, opts.nreuse, trandwords, goodwords, modelmats(1), modelmats(0), lrneg, vexp);    	  	
    		addTime(9);
    	} else {
    		procPositivesSlice(opts.nskip, words, lb, ub, modelmats, lrpos, vexp, opts.iSlice);
    		addTime(8);   	
    		procNegativesSlice(opts.nneg, opts.nreuse, trandwords, goodwords, modelmats, lrneg, vexp, opts.iSlice);    	  	
    		addTime(9);
    	}
    }
  }
  
  def evalbatch(gmats:Array[Mat], ipass:Int, pos:Long):FMat = {
  	addTime(0);
  	if (gmats(0).ncols == ncols) {
  	val (words, lb, ub, trandwords, goodwords) = wordMats(gmats, ipass, pos);
  	val (epos, eneg) = if (opts.nSlices == 1 && nmmats == 1) {
  		val epos0 = evalPositives(opts.nskip, words, lb, ub, modelmats(1), modelmats(0));
  		addTime(10,-3);
  		val eneg0 = evalNegatives(opts.nneg, opts.nreuse, trandwords, goodwords, modelmats(1), modelmats(0));
  		addTime(11);
  		(epos0, eneg0)
  	} else {
  		val epos0 = evalPositivesSlice(opts.nskip, words, lb, ub, modelmats, opts.iSlice);
  		addTime(10,-3);
  		val eneg0 = evalNegativesSlice(opts.nneg, opts.nreuse, trandwords, goodwords, modelmats, opts.iSlice);
  		addTime(11);
  		(epos0, eneg0)
  	}
  	val score = ((epos + eneg / (if (opts.eqPosNeg) 1 else opts.nneg)) / goodwords.length);
  	row(score)
  	} else row(0);
  }
  
  def wordMats(mats:Array[Mat], ipass:Int, pos:Long):(Mat, Mat, Mat, Mat, Mat) = {
  
    val wordsens = mats(0);
    val words = if (opts.iflip) wordsens(1,?) else wordsens(0,?);
    val wgood = words < opts.vocabSize;                                        // Find OOV words 
    addTime(1);
    
    rand(randsamp);                                                            // Take a random sample
    val wrat = float(words+1) * salpha;
    wrat ~ sqrt(wrat) + wrat;
    wgood ~ wgood ∘ int(randsamp < wrat);
    words ~ (wgood ∘ (words + 1)) - 1;                                         // Set OOV or skipped samples to -1
    addTime(2);
        
    rand(ubound);                                                              // get random upper and lower bounds   
    val ubrand = min(opts.nskip, int(ubound * opts.nskip) + 1);
    val lbrand = - ubrand;
    addTime(3);
    
    val sentencenum = if (opts.iflip) wordsens(0,?) else wordsens(1,?);        // Get the nearest sentence boundaries
    val lbsentence = - cumsumByKey(allones, sentencenum) + 1;
    val ubsentence = reverse(cumsumByKey(allones, reverse(sentencenum))) - 1;
    val lb = max(lbrand, lbsentence);                                          // Combine the bounds
    val ub = min(ubrand, ubsentence);
    test3 = lb
    test4 = ub
    addTime(4);
    
    val (trandwords, contextwords) = (words, lb, ub) match {
      case (giwords:GIMat, gilb:GIMat, giub:GIMat) => {

      	val iwords = minusone \ words \ minusone;                              // Build a convolution matrix.
      	val cwords = iwords(wordtab);
      	val pgoodwords = (wordmask >= lb) ∘ (wordmask <= ub) ∘ (cwords >= 0) ∘ (words >= 0);  // Find context words satisfying the bound
      	                                                                           // and check that context and center word are good.
      	val fgoodwords = float(pgoodwords);
      	addTime(5);
      	
      	test1 = cwords;

      	rand(randpermute);                                                      // Prepare a random permutation of context words for negative sampling
      	randpermute ~ (fgoodwords ∘ (randpermute + 1f)) - 1f;                   // set the values for bad words to -1.
      	val (vv, ii) = sortdown2(randpermute.view(randpermute.length, 1));      // Permute the good words
      	val ngood = sum(vv >= 0f).dv.toInt;                                     // Count of the good words
      	val ngoodcols = ngood / opts.nreuse;                                    // Number of good columns
      	val cwi = cwords(ii);
      	
      	test2 = cwi
      	addTime(6);

      	rand(randwords);                                                        // Compute some random negatives
      	val irandwords = min(nfeats-1, int(nfeats * (randwords ^ expt))); 
      	val trandwords0 = irandwords.view(opts.nneg, ngoodcols);                // shrink the matrices to the available data
      	val contextwords0 = cwi.view(opts.nreuse, ngoodcols);
      	addTime(7);
      	(trandwords0, contextwords0)
      }
      case (iwords:IMat, ilb:IMat, iub:IMat) => {
        getnegs(iwords, ilb, iub, Mat.numThreads);
      }
    }
    
    (words, lb, ub, trandwords, contextwords);
  }
  
  def getnegs(words:IMat, lb:IMat, ub:IMat, nthreads:Int):(IMat, IMat) = {
    val ncols = words.ncols;
                                                                               // First count the good context words
    val cwcounts = irow((0 until nthreads).par.map((ithread:Int) => {          // work on blocks
      val istart = ((1L * ncols * ithread)/nthreads).toInt;
      val iend = ((1L * ncols * (ithread + 1))/nthreads).toInt;
      var i = istart;
      var icount = 0;
      while (i < iend) {                                                       // iterate over center words
        if (words.data(i) >= 0) {                                              // check center word is good
        	var j = lb.data(i);                                                  // get lower and upper bounds
        	var jend = ub.data(i);
        	while (j <= jend) {
        		if (j != 0 && words.data(i + j) >= 0) {                            // if not center word and context word is good, count it. 
        		  icount += 1;        		  
        		}
        		j += 1;
        	}
        }
        i += 1;
      }
      icount
    }).toArray)
                                                                               // Now we know how many good words in each block
    val ccc = cumsum(cwcounts);                                                // so size the context word and neg word matrices
    val ngroups = ccc(ccc.length - 1) / opts.nreuse;
    val contextwords0 = izeros(opts.nreuse, ngroups);
    val trandwords0 = izeros(opts.nneg, ngroups);
    
    (0 until nthreads).par.map((ithread:Int) => {                              // Copy the good words into a dense matrix (contextwords0)
      val istart = ((1L * ncols * ithread)/nthreads).toInt;
      val iend = ((1L * ncols * (ithread + 1))/nthreads).toInt;
      var i = istart;
      var icount = 0;
      val mptr = ccc(ithread) - ccc(0);
      while (i < iend) {
        if (words.data(i) >= 0) {
        	var j = lb.data(i);
        	var jend = ub.data(i);
        	while (j <= jend && mptr + icount < contextwords0.length) {
        		if (j != 0 && words.data(i + j) >= 0) {
        		  contextwords0.data(mptr + icount) = words.data(i + j)
        		  icount += 1;        		  
        		}
        		j += 1;
        	}
        }
        i += 1;
      }
      icount
    })
    
    addTime(5);
    
    val prand = drand(opts.nreuse, ngroups);                                   // Rands for permutation
    
    var i = 0;                                                                 // Permute the good context words randomly
    val n = prand.length;
    while (i < n) {
      val indx = math.min(n-1, i + math.floor(prand.data(i) * (n - i)).toInt);
      if (indx > i) {
        val tmp = contextwords0.data(i);
        contextwords0.data(i) = contextwords0.data(indx);
        contextwords0.data(indx) = tmp;
      }
      i += 1;
    }
    addTime(6);
    
    val randneg = rand(opts.nneg, ngroups);                                    // Compute some random negatives

    (0 until nthreads).par.map((ithread:Int) => {                              // Work in blocks over the negs
      val istart = ((1L * ngroups * opts.nneg * ithread)/nthreads).toInt;
      val iend = ((1L * ngroups * opts.nneg * (ithread + 1))/nthreads).toInt;
      var i = istart;
      while (i < iend) {
      	trandwords0.data(i) = math.min(nfeats-1, (nfeats * math.pow(randneg.data(i), expt)).toInt);
        i += 1;
      }
    })
//    println("mean=%f" format mean(FMat(trandwords0(?) < opts.nHeadTerms)).v);
    addTime(7);
    
    (trandwords0, contextwords0)    
  }
  
  def procPositives(nskip:Int, words:Mat, lbound:Mat, ubound:Mat, model1:Mat, model2:Mat, lrate:Float, vexp:Float) = {
    val nrows = model1.nrows;
    val ncols = model1.ncols;
    val nwords = words.ncols;
    Mat.nflops += 6L * nwords * nskip * nrows;
    (words, lbound, ubound, model1, model2) match {
      case (w:GIMat, lb:GIMat, ub:GIMat, m1:GMat, m2:GMat) => {
        val err = CUMACH.word2vecPos(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, lrate, vexp);
        if (err != 0)  throw new RuntimeException("CUMACH.word2vecPos error " + cudaGetErrorString(err));
      }
      case (w:IMat, lb:IMat, ub:IMat, m1:FMat, m2:FMat) => if (Mat.useMKL) {
        CPUMACH.word2vecPos(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, lrate, vexp, Mat.numThreads);
      } else {
        Word2Vec.procPosCPU(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, lrate, vexp, Mat.numThreads);
      }
    }
  }
  
  def procNegatives(nwa:Int, nwb:Int, wordsa:Mat, wordsb:Mat, modela:Mat, modelb:Mat, lrate:Float, vexp:Float) = {
    val nrows = modela.nrows;
    val ncols = modela.ncols;
    val nwords = wordsa.ncols;
    Mat.nflops += 6L * nwords * nwa * nwb * nrows;
    (wordsa, wordsb, modela, modelb) match {
      case (wa:GIMat, wb:GIMat, ma:GMat, mb:GMat) => {
        val err = CUMACH.word2vecNeg(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, lrate, vexp);
        if (err != 0) throw new RuntimeException("CUMACH.word2vecNeg error " + cudaGetErrorString(err));
      }
      case (wa:IMat, wb:IMat, ma:FMat, mb:FMat) => if (Mat.useMKL) {
        CPUMACH.word2vecNeg(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, lrate, vexp, Mat.numThreads);
      } else {
      	Word2Vec.procNegCPU(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, lrate, vexp, Mat.numThreads);
      }
    }
  }
  
  def procPositivesSlice(nskip:Int, words:Mat, lbound:Mat, ubound:Mat, modelmats:Array[Mat], lrate:Float, vexp:Float, islice:Int) = {
    import scala.concurrent.ExecutionContext.Implicits.global
    val nrows = modelmats(0).nrows;
    val nwords = words.ncols;
    Mat.nflops += 6L * nwords * nskip * nrows;
    (words, lbound, ubound) match {
      case (w:IMat, lb:IMat, ub:IMat) => if (Mat.useMKL) {
      	CPUMACH.word2vecPosSlice(nrows, nwords, nskip, w.data, lb.data, ub.data, fmm, lrate, vexp, Mat.numThreads, 
          islice, opts.nSlices, maxCols, opts.nHeadTerms, if (opts.dualMode) 1 else 0, opts.doHead);
      } else {
      	Word2Vec.procPosCPUslice(nrows, nwords, nskip, w.data, lb.data, ub.data, modelmats, lrate, vexp, Mat.numThreads, 
          islice, opts.nSlices, maxCols, opts.nHeadTerms, opts.dualMode, opts.doHead);
      }
      case (w:GIMat, lb:GIMat, ub:GIMat) => if (opts.dualMode) {
      	val m0 = modelmats(0).asInstanceOf[GMat];
      	val m1 = modelmats(1).asInstanceOf[GMat];
      	m0 <-- modelmats(2);
      	m1 <-- modelmats(3);
//      	val err = CUMACH.word2vecPos(nrows, m0.ncols, nskip, w.data, lb.data, ub.data, m0.data, m1.data, lrate, vexp);
//      	if (err != 0)  throw new RuntimeException("CUMACH.word2vecPos error " + cudaGetErrorString(err));   
      	modelmats(2) <-- m0;
      	modelmats(3) <-- m1;
      	Word2Vec.procPosCPUslice(nrows, nwords, nskip, IMat(w).data, IMat(lb).data, IMat(ub).data, modelmats, lrate, vexp, Mat.numThreads, 
      			islice, opts.nSlices, maxCols, opts.nHeadTerms, opts.dualMode, opts.doHead);
      } else {
        throw new RuntimeException("Use dualMode to use the GPU with multi-part models")
      }        
    }
  }
  
  def procNegativesSlice(nwa:Int, nwb:Int, wordsa:Mat, wordsb:Mat, modelmats:Array[Mat], lrate:Float, vexp:Float, islice:Int) = {
    import scala.concurrent.ExecutionContext.Implicits.global
    val nrows = modelmats(0).nrows;
    val nvocab = modelmats(0).ncols;
    val nwords = wordsa.ncols;
    Mat.nflops += 6L * nwords * nwa * nwb * nrows;
    (wordsa, wordsb) match {
    case (wa:IMat, wb:IMat) => if (Mat.useMKL) {
    	CPUMACH.word2vecNegSlice(nrows, nwords, nwa, nwb, wa.data, wb.data, fmm, lrate, vexp, Mat.numThreads, 
    			islice, opts.nSlices, maxCols, opts.nHeadTerms, if (opts.dualMode) 1 else 0, opts.doHead);
    } else {
    	Word2Vec.procNegCPUslice(nrows, nwords, nwa, nwb, wa.data, wb.data, modelmats, lrate, vexp, Mat.numThreads, 
    			islice, opts.nSlices, maxCols, opts.nHeadTerms, opts.dualMode, opts.doHead);
    }
    case (wa:GIMat, wb:GIMat) => {
    	if (opts.dualMode) {
    		val m0 = modelmats(0).asInstanceOf[GMat];
    		val m1 = modelmats(1).asInstanceOf[GMat];
    		m0 <-- modelmats(2);
    		m1 <-- modelmats(3);
    		val err = CUMACH.word2vecNegFilt(nrows, nwords, nvocab, nwa, nwb, wa.data, wb.data, m0.data, m1.data, lrate, vexp);
    		if (err != 0)  throw new RuntimeException("CUMACH.word2vecNegFilt error " + cudaGetErrorString(err));    
    		modelmats(2) <-- m0;
    		modelmats(3) <-- m1;
    		Word2Vec.procNegCPUslice(nrows, nwords, nwa, nwb, IMat(wa).data, IMat(wb).data, modelmats, lrate, vexp, Mat.numThreads, 
    				islice, opts.nSlices, maxCols, opts.nHeadTerms, opts.dualMode, opts.doHead);
    	} else {
    		throw new RuntimeException("Use dualMode to use the GPU with multi-part models")
    	}
    }
    }
  }
    
  def evalPositives(nskip:Int, words:Mat, lbound:Mat, ubound:Mat, model1:Mat, model2:Mat):Double = {
    val nrows = model1.nrows;
    val ncols = model1.ncols;
    val nwords = words.ncols;
    Mat.nflops += 2L * nwords * nskip * nrows;
    (words, lbound, ubound, model1, model2) match {
      case (w:GIMat, lb:GIMat, ub:GIMat, m1:GMat, m2:GMat) => {
        retEvalPos.clear
        val err = CUMACH.word2vecEvalPos(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, retEvalPos.data);
    		if (err != 0) throw new RuntimeException("CUMACH.word2vecEvalPos error " + cudaGetErrorString(err));
        retEvalPos.dv;
      }
      case (w:IMat, lb:IMat, ub:IMat, m1:FMat, m2:FMat) => 
      if (Mat.useMKL) {
      	CPUMACH.word2vecEvalPos(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, Mat.numThreads);
      } else {
        Word2Vec.evalPosCPU(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, Mat.numThreads);
      }
    }
  }
  
  def evalPositivesSlice(nskip:Int, words:Mat, lbound:Mat, ubound:Mat, modelmats:Array[Mat], islice:Int):Double = {
    val nrows = modelmats(0).nrows;
    val nwords = words.ncols;
    Mat.nflops += 2L * nwords * nskip * nrows;
    (words, lbound, ubound) match {
      case (w:IMat, lb:IMat, ub:IMat) => 
        Word2Vec.evalPosCPUslice(nrows, nwords, nskip, w.data, lb.data, ub.data, modelmats, Mat.numThreads,
        		islice, opts.nSlices, maxCols, opts.nHeadTerms, opts.dualMode);
    }
  }
  
  def evalNegatives(nwa:Int, nwb:Int, wordsa:Mat, wordsb:Mat, modela:Mat, modelb:Mat):Double = {
    val nrows = modela.nrows;
    val ncols = modela.ncols;
    val nwords = wordsa.ncols;
    Mat.nflops += 2L * nwords * nwa * nwb * nrows;
    (wordsa, wordsb, modela, modelb) match {
      case (wa:GIMat, wb:GIMat, ma:GMat, mb:GMat) => {
        retEvalNeg.clear
        val err = CUMACH.word2vecEvalNeg(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, retEvalNeg.data);
    		if (err != 0) throw new RuntimeException("CUMACH.word2vecEvalNeg error " + cudaGetErrorString(err));
        retEvalNeg.dv;      
      }
      case (wa:IMat, wb:IMat, ma:FMat, mb:FMat) => 
      if (Mat.useMKL) {
      	CPUMACH.word2vecEvalNeg(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, Mat.numThreads); 
      } else {
      	Word2Vec.evalNegCPU(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, Mat.numThreads);
      }
    }
  }
  
  def evalNegativesSlice(nwa:Int, nwb:Int, wordsa:Mat, wordsb:Mat, modelmats:Array[Mat], islice:Int):Double = {
    val nrows = modelmats(0).nrows;
    val nwords = wordsa.ncols;
    Mat.nflops += 2L * nwords * nwa * nwb * nrows;
    (wordsa, wordsb) match {
      case (wa:IMat, wb:IMat) => 
      	Word2Vec.evalNegCPUslice(nrows, nwords, nwa, nwb, wa.data, wb.data, modelmats, Mat.numThreads,
      	    islice, opts.nSlices, maxCols, opts.nHeadTerms, opts.dualMode);
    }
  }
  
  def trailingZeros(a:Long):Int = {
    var aa = a;
    var nz = 0;
    while ((aa & 1L) == 0) {
      aa = aa >> 1;
      nz += 1;
    }
    nz
  }
  
  override def mergeModelFn(models:Array[Model], mm:Array[Mat], um:Array[Mat], istep:Long):Unit = {
    val headlen = if (istep > 0) math.max(opts.headlen, opts.headlen << trailingZeros(istep)) else 0;
    val mlen = models(0).modelmats.length;
    val thisGPU = getGPU;
    val modj = new Array[Mat](models.length);
    for (j <- 0 until mlen) {
      val mmj = if (headlen > 0) mm(j).view(mm(j).nrows, math.min(mm(j).ncols, headlen)) else mm(j);
      mmj.clear
      for (i <- 0 until models.length) {
        if (useGPU && i < Mat.hasCUDA) setGPU(i);
        modj(i) = if (headlen > 0) models(i).modelmats(j).view(models(i).modelmats(j).nrows, math.min(models(i).modelmats(j).ncols, headlen)) else models(i).modelmats(j);
        val umj = if (headlen > 0) um(j).view(um(j).nrows, math.min(um(j).ncols, headlen)) else um(j);
        umj <-- modj(i)
        mmj ~ mmj + umj;
      }
      mmj ~ mmj * (1f/models.length);
      for (i <- 0 until models.length) {
        modj(i) <-- mmj;
      }
    }
    setGPU(thisGPU);
  }

}

object Word2Vec  {
  trait Opts extends Model.Opts {
    var aopts:ADAGrad.Opts = null;
    var nskip = 5;
    var nneg = 5;
    var nreuse = 5;  
    var vocabSize = 100000;
    var wexpt = 0.75f;
    var wsample = 1e-4f;
    var headlen = 10000;
    var iflip = false;
    var eqPosNeg = false;
    var maxArraySize = 2047*1024*1024;
    var nHeadTerms = 0; 
    var nSlices = 1;
    var iSlice = 0;
    var dualMode = false;
    var doHead = 1;
  }
  
  class Options extends Opts {}
  
  
  def procPosCPU(nrows:Int, ncols:Int, skip:Int, W:Array[Int], LB:Array[Int], UB:Array[Int],
      A:Array[Float], B:Array[Float], lrate:Float, vexp:Float, nthreads:Int):Int = {

    (0 until nthreads).par.map((ithread:Int) => {
    	val istart = ((1L * ithread * ncols)/nthreads).toInt;
    	val iend = ((1L * (ithread+1) * ncols)/nthreads).toInt;
    	val daa = new Array[Float](nrows);
    	var i = istart;
    	while (i < iend) {
    	  var j = 0;
    	  var k = 0;
    	  var c = 0;
    	  var cv = 0f;

    	  val iac = W(i);
    	  val ascale = math.pow(1+iac, vexp).toFloat;
    	  val ia = nrows * iac;                                        // Get the current word (as a model matrix offset). 
    	  if (ia >= 0) {                                               // Check for OOV words
    	  	c = 0;
    	  	while (c < nrows) {                                        // Current word
    	  		daa(c) = 0;                                              // delta for the A matrix (maps current and negative words). 
    	  		c += 1;
    	  	}
    	  	j = LB(i);
    	  	while (j <= UB(i)) {                                       // Iterate over neighbors in the skip window
    	  		if (j != 0 && i + j >= 0 && i + j < ncols) {             // context word index is in range (and not current word).
    	  		  val ibc = W(i + j);
    	  		  val bscale = math.pow(1+ibc, vexp).toFloat;
    	  			val ib = nrows * ibc;                                  // Get the context word and check it
    	  			if (ib >= 0) {
    	  				c = 0;
    	  				cv = 0f;
    	  				while (c < nrows) {                                  // Inner product between current and context words. 
    	  					cv += A(c + ia) * B(c + ib);
    	  					c += 1;
    	  				}

    	  				if (cv > 16.0f) {                                    // Apply logistic function with guards
    	  					cv = 1.0f;
    	  				} else if (cv < -16.0f) {
    	  					cv = 0.0f;
    	  				} else {
    	  					cv = math.exp(cv).toFloat;
    	  					cv = cv / (1.0f + cv);
    	  				}
    	  				cv = lrate * (1.0f - cv);                            // Subtract prediction from target (1.0), and scale by learning rate. 

    	  				c = 0;
    	  				while (c < nrows) {
    	  					daa(c) += ascale * cv * B(c + ib);                 // Compute backward derivatives for A and B with pseudo-ADAGrad scaling
    	  					B(c + ib) += bscale * cv * A(c + ia);
    	  					c += 1;
    	  				}
    	  			}
    	  		}
    	  		j += 1;
    	  	}
    	  	c = 0;
    	  	while (c < nrows) {                                        // Add derivative for A to A. 
    	  		A(c + ia) += daa(c);
    	  		c += 1;
    	  	}
    	  }
    	  i += 1;
    	}
    });
    0;
  }
  
  def mapIndx(indx:Int, islice:Int, nslices:Int, nHead:Int, maxCols:Int, nrows:Int, offset:Int):(Int, Int, Boolean, Boolean) = {
  	val newi = if (indx >= nHead) ((indx - nHead) / nslices + nHead) else indx;                     // new column index
  	val m = newi / maxCols + offset;                                 // which matrix are we in? 
  	val ismine = (indx >= nHead) && (indx % nslices == islice);
  	val ishead = (indx < nHead);
  	val i = nrows * (newi - m * maxCols);
  	(m, i, ismine, ishead)
  }

  def procPosCPUslice(nrows:Int, ncols:Int, skip:Int, W:Array[Int], LB:Array[Int], UB:Array[Int],
     modelmats:Array[Mat], lrate:Float, vexp:Float, nthreads:Int, 
     islice:Int, nslices:Int, maxCols:Int, nHead:Int, dualMode:Boolean, doHead:Int):Int = {

    val arrayOffset = if (dualMode) 1 else 0;
    (0 until nthreads).par.map((ithread:Int) => {
    	val istart = ((1L * ithread * ncols)/nthreads).toInt;
    	val iend = ((1L * (ithread+1) * ncols)/nthreads).toInt;
    	val daa = new Array[Float](nrows);
    	var i = istart;
    	while (i < iend) {
    	  var j = 0;
    	  var k = 0;
    	  var c = 0;
    	  var cv = 0f;

    	  val iac = W(i);
    	  val ascale = math.pow(1+iac, vexp).toFloat; 
    	  if (iac >= 0) {                                              // Check for OOV words
    	    val (ma, ia, aismine, aishead) = mapIndx(iac, islice, nslices, nHead, maxCols, nrows, arrayOffset);
    	    val A = modelmats(2*ma+1).asInstanceOf[FMat].data;
    	  	c = 0;
    	  	while (c < nrows) {                                        // Current word
    	  		daa(c) = 0;                                              // delta for the A matrix (maps current and negative words). 
    	  		c += 1;
    	  	}
    	  	j = LB(i);
    	  	var touched = false;
    	  	while (j <= UB(i)) {                                       // Iterate over neighbors in the skip window
    	  		if (j != 0 && i + j >= 0 && i + j < ncols) {             // context word index is in range (and not current word).
    	  		  val ibc = W(i + j);                                    // Get the context word 
    	  		  val bscale = math.pow(1+ibc, vexp).toFloat;  
    	  			if (ibc >= 0) {                                        // check if context word is OOV
    	  			  val (mb, ib, bismine, bishead) = mapIndx(ibc, islice, nslices, nHead, maxCols, nrows, arrayOffset);
    	  				val B = modelmats(2*mb).asInstanceOf[FMat].data;
    	  				if ((doHead > 1 && aishead && bishead) || (aismine && bishead) || (bismine && aishead) || (aismine && bismine)) {
    	  				  touched = true;
    	  					c = 0;
    	  					cv = 0f;
    	  					while (c < nrows) {                                // Inner product between current and context words. 
    	  						cv += A(c + ia) * B(c + ib);
    	  						c += 1;
    	  					}

    	  					if (cv > 16.0f) {                                  // Apply logistic function with guards
    	  						cv = 1.0f;
    	  					} else if (cv < -16.0f) {
    	  						cv = 0.0f;
    	  					} else {
    	  						cv = math.exp(cv).toFloat;
    	  						cv = cv / (1.0f + cv);
    	  					}
    	  					cv = lrate * (1.0f - cv);                          // Subtract prediction from target (1.0), and scale by learning rate. 

    	  					c = 0;
    	  					while (c < nrows) {
    	  						daa(c) += ascale * cv * B(c + ib);               // Compute backward derivatives for A and B with pseudo-ADAGrad scaling
    	  						c += 1;
    	  					}
    	  					if (bismine || (bishead && doHead > 0)) {
    	  						c = 0;
    	  						while (c < nrows) {
    	  							B(c + ib) += bscale * cv * A(c + ia);
    	  							c += 1;
    	  						}
    	  					}
    	  				}
    	  			}
    	  		}
    	  		j += 1;
    	  	}
    	  	if (touched && (aismine || (aishead && doHead > 0))) {
    	  		c = 0;
    	  		while (c < nrows) {                                        // Add derivative for A to A. 
    	  			A(c + ia) += daa(c);
    	  			c += 1;
    	  		}
    	  	}
    	  }
    	  i += 1;
    	}
    });
    0;
  }

  
  def procNegCPU(nrows:Int, nwords:Int, nwa:Int, nwb:Int, WA:Array[Int], WB:Array[Int], A:Array[Float], B:Array[Float], 
      lrate:Float, vexp:Float, nthreads:Int):Int = {

  	(0 until nthreads).par.map((ithread:Int) => {
  		val istart = ((1L * nwords * ithread) / nthreads).toInt;
  		val iend = ((1L * nwords * (ithread+1)) / nthreads).toInt;
  		val aa = new Array[Float](nwa * nrows);
  		val bb = new Array[Float](nrows);
  		var i = istart;
  		while (i < iend) {
  			var j = 0;
  			var k = 0;
  			var c = 0;

  			j = 0;	
  			while (j < nwa) {                                            // Clear tmp A matrix 
  			  val ja = j * nrows;
  			  c = 0; 
  			  while (c < nrows) {
  			    aa(c + ja) = 0;
  			    c += 1;
  			  }
  			  j+= 1;
  			}
  			
  			k = 0;
  			while (k < nwb) {                                            // Loop over B words
  				c = 0; 
  			  while (c < nrows) {                                        // Clear tmp B vector
  			    bb(c) = 0;
  			    c += 1;
  			  }
  			  val ibc = WB(k+i*nwb);
  			  val bscale = math.pow(1+ibc, vexp).toFloat;
  				val ib = nrows * ibc;                                      // Get the B word as an array offset. 
  				j = 0;
  				while (j < nwa) {                                          // Now iterate over A words. 
  				  val iac = WA(j+i*nwa);
  				  val ascale = math.pow(1+iac, vexp).toFloat;
  					val ia = nrows * iac; 		                               // Get an A word offset 
  					
  					var cv = 0f;
  					c = 0;
  					while (c < nrows) {                                      // Inner product between A and B columns
  						cv += A(c + ia) * B(c + ib);
  						c += 1;
  					}
  					
  					if (cv > 16.0f) {                                        // Guarded logistic function
  						cv = 1.0f;
  					} else if (cv < -16.0f) {
  						cv = 0.0f;
  					} else {
  						cv = math.exp(cv).toFloat;
  						cv = cv / (1.0f + cv);
  					}
  					cv = - cv * lrate;                                       // Scale derivative by learning rate. 

  					val ja = j * nrows;
  					c = 0;
  					while (c < nrows) {                                      // Update the derivatives
  						aa(c + ja) += ascale * cv * B(c + ib);
  						bb(c) +=  bscale * cv * A(c + ia);
  						c += 1;
  					}
  					j += 1;
  				}
  				c = 0;
  				while (c < nrows) {                                        // Add B's derivative to B
  					B(c + ib) += bb(c);
  					c += 1;
  				}
  				k += 1;
  			}
  			j = 0;
  			while (j < nwa) {                                            // Add A's derivatives to A
  			  val ja = j * nrows;
  			  val ia = nrows * WA(j+i*nwa);
  			  c = 0;
  			  while (c < nrows) {
  			    A(c + ia) += aa(c + ja);
  			    c += 1;
  			  }
  			  j += 1;
  			}
  			i += 1;
  		}
  	});
  	0;
  }
  
    
  def procNegCPUslice(nrows:Int, nwords:Int, nwa:Int, nwb:Int, WA:Array[Int], WB:Array[Int], modelmats:Array[Mat], 
      lrate:Float, vexp:Float, nthreads:Int, islice:Int, nslices:Int, maxCols:Int, nHead:Int, dualMode:Boolean, doHead:Int):Int = {

  	val arrayOffset = if (dualMode) 1 else 0;
    (0 until nthreads).par.map((ithread:Int) => {
  		val istart = ((1L * nwords * ithread) / nthreads).toInt;
  		val iend = ((1L * nwords * (ithread+1)) / nthreads).toInt;
  		val aa = new Array[Float](nwa * nrows);
  		val bb = new Array[Float](nrows);
  		var i = istart;
  		while (i < iend) {
  			var j = 0;
  			var k = 0;
  			var c = 0;

  			j = 0;	
  			while (j < nwa) {                                            // Clear tmp A matrix 
  			  val ja = j * nrows;
  			  c = 0; 
  			  while (c < nrows) {
  			    aa(c + ja) = 0;
  			    c += 1;
  			  }
  			  j+= 1;
  			}
  			
  			k = 0;
  			while (k < nwb) {                                            // Loop over B words
  				c = 0; 
  			  while (c < nrows) {                                        // Clear tmp B vector
  			    bb(c) = 0;
  			    c += 1;
  			  }
  			  val ibc = WB(k+i*nwb);
  			  val bscale = math.pow(1+ibc, vexp).toFloat;
  			  val (mb, ib, bismine, bishead) = mapIndx(ibc, islice, nslices, nHead, maxCols, nrows, arrayOffset);
  			  val B = modelmats(2*mb).asInstanceOf[FMat].data;
  				j = 0;
  				while (j < nwa) {                                          // Now iterate over A words. 
  				  val iac = WA(j+i*nwa);
  				  val ascale = math.pow(1+iac, vexp).toFloat;
  				  val (ma, ia, aismine, aishead) = mapIndx(iac, islice, nslices, nHead, maxCols, nrows, arrayOffset);
  				  val A = modelmats(2*ma+1).asInstanceOf[FMat].data;	
  					var cv = 0f;
  					if ((doHead > 1 && aishead && bishead) || (aismine && bishead) || (bismine && aishead) || (aismine && bismine)) {
  						c = 0;
  						while (c < nrows) {                                      // Inner product between A and B columns
  							cv += A(c + ia) * B(c + ib);
  							c += 1;
  						}

  						if (cv > 16.0f) {                                        // Guarded logistic function
  							cv = 1.0f;
  						} else if (cv < -16.0f) {
  							cv = 0.0f;
  						} else {
  							cv = math.exp(cv).toFloat;
  							cv = cv / (1.0f + cv);
  						}
  						cv = - cv * lrate;                                       // Scale derivative by learning rate. 

  						val ja = j * nrows;
  						c = 0;
  						while (c < nrows) {                                      // Update the derivatives
  							aa(c + ja) += ascale * cv * B(c + ib);
  							bb(c) +=  bscale * cv * A(c + ia);
  							c += 1;
  						}
  					}
  					j += 1;
  				}
  				if (bismine || (bishead && doHead > 0)) {
  					c = 0;
  					while (c < nrows) {                                        // Add B's derivative to B
  						B(c + ib) += bb(c);
  						c += 1;
  					}
  				}
  				k += 1;
  			}
  			j = 0;
  			while (j < nwa) {                                            // Add A's derivatives to A
  			  val ja = j * nrows;
  			  val iac = WA(j+i*nwa);
  			  val (ma, ia, aismine, aishead) = mapIndx(iac, islice, nslices, nHead, maxCols, nrows, arrayOffset);
  			  val A = modelmats(2*ma+1).asInstanceOf[FMat].data;
  			  if (aismine || (aishead && doHead > 0)) {
  			  	c = 0;
  			  	while (c < nrows) {
  			  		A(c + ia) += aa(c + ja);
  			  		c += 1;
  			  	}
  			  }
  			  j += 1;
  			}
  			i += 1;
  		}
  	});
  	0;
  }

  def evalPosCPU(nrows:Int, ncols:Int, skip:Int, W:Array[Int], LB:Array[Int], UB:Array[Int],
      A:Array[Float], B:Array[Float], nthreads:Int):Double = {

    (0 until nthreads).par.map((ithread:Int) => {
    	val istart = ((1L * ithread * ncols)/nthreads).toInt;
    	val iend = ((1L * (ithread+1) * ncols)/nthreads).toInt;
    	val daa = new Array[Float](nrows);
    	var i = istart;
    	var sum = 0.0;
    	while (i < iend) {
    	  var j = 0;
    	  var k = 0;
    	  var c = 0;
    	  var cv = 0f;

    	  val ia = nrows * W(i);                                       // Get the current word (as a model matrix offset). 
    	  if (ia >= 0) {                                               // Check for OOV words
    	  	c = 0;
    	  	while (c < nrows) {                                        // Current word
    	  		daa(c) = 0;                                              // delta for the A matrix (maps current and negative words). 
    	  		c += 1;
    	  	}
    	  	j = LB(i);
    	  	while (j <= UB(i)) {                                       // Iterate over neighbors in the skip window
    	  		if (j != 0 && i + j >= 0 && i + j < ncols) {             // context word index is in range (and not current word).
    	  			val ib = nrows * W(i + j);                             // Get the context word and check it. 
    	  			if (ib >= 0) {
    	  				c = 0;
    	  				cv = 0f;
    	  				while (c < nrows) {                                  // Inner product between current and context words. 
    	  					cv += A(c + ia) * B(c + ib);
    	  					c += 1;
    	  				}

    	  				if (cv > 16.0f) {                                    // Apply logistic function with guards
    	  					cv = 1.0f;
    	  				} else if (cv < -16.0f) {
    	  					cv = 0.0f;
    	  				} else {
    	  					cv = math.exp(cv).toFloat;
    	  					cv = cv / (1.0f + cv);
    	  				}
    	  				sum += math.log(math.max(cv, 1e-20));                            
    	  			}
    	  		}
    	  		j += 1;
    	  	}
    	  }
    	  i += 1;
    	}
    	sum;
    }).reduce(_+_);
  }
  
  def evalPosCPUslice(nrows:Int, ncols:Int, skip:Int, W:Array[Int], LB:Array[Int], UB:Array[Int],
     modelmats:Array[Mat], nthreads:Int, islice:Int, nslices:Int, maxCols:Int, nHead:Int, dualMode:Boolean):Double = {

    val arrayOffset = if (dualMode) 1 else 0;
    (0 until nthreads).par.map((ithread:Int) => {
    	val istart = ((1L * ithread * ncols)/nthreads).toInt;
    	val iend = ((1L * (ithread+1) * ncols)/nthreads).toInt;
    	val daa = new Array[Float](nrows);
    	var i = istart;
    	var sum = 0.0;
    	while (i < iend) {
    	  var j = 0;
    	  var k = 0;
    	  var c = 0;
    	  var cv = 0f;

    	  val iac =  W(i);                                       // Get the current word (as a model matrix offset). 
    	  if (iac >= 0) {  
    	  	val (ma, ia, aismine, aishead) = mapIndx(iac, islice, nslices, nHead, maxCols, nrows, arrayOffset);
    	  	if (aismine || aishead) {
    	  		val A = modelmats(2*ma+1).asInstanceOf[FMat].data;
    	  		c = 0;
    	  		while (c < nrows) {                                        // Current word
    	  			daa(c) = 0;                                              // delta for the A matrix (maps current and negative words). 
    	  			c += 1;
    	  		}
    	  		j = LB(i);
    	  		while (j <= UB(i)) {                                       // Iterate over neighbors in the skip window
    	  			if (j != 0 && i + j >= 0 && i + j < ncols) {             // context word index is in range (and not current word).
    	  				val ibc = W(i + j);                             // Get the context word and check it. 
    	  				if (ibc >= 0) {
    	  					val (mb, ib, bismine, bishead) = mapIndx(ibc, islice, nslices, nHead, maxCols, nrows, arrayOffset);
    	  					if (bismine || bishead) {
    	  						val B = modelmats(2*mb).asInstanceOf[FMat].data;
    	  						c = 0;
    	  						cv = 0f;
    	  						while (c < nrows) {                                  // Inner product between current and context words. 
    	  							cv += A(c + ia) * B(c + ib);
    	  							c += 1;
    	  						}

    	  						if (cv > 16.0f) {                                    // Apply logistic function with guards
    	  							cv = 1.0f;
    	  						} else if (cv < -16.0f) {
    	  							cv = 0.0f;
    	  						} else {
    	  							cv = math.exp(cv).toFloat;
    	  							cv = cv / (1.0f + cv);
    	  						}
    	  						sum += math.log(math.max(cv, 1e-20));                            
    	  					}
    	  				}
    	  			}
    	  			j += 1;
    	  		}
    	  	}
    	  }
    	  i += 1;
    	}
    	sum;
    }).reduce(_+_);
  }

  
  def evalNegCPU(nrows:Int, nwords:Int, nwa:Int, nwb:Int, WA:Array[Int], WB:Array[Int], A:Array[Float], B:Array[Float], nthreads:Int):Double = {

  	(0 until nthreads).par.map((ithread:Int) => {
  		val istart = ((1L * nwords * ithread) / nthreads).toInt;
  		val iend = ((1L * nwords * (ithread+1)) / nthreads).toInt;
  		val aa = new Array[Float](nwa * nrows);
  		val bb = new Array[Float](nrows);
  		var sum = 0.0;
  		var i = istart;
  		while (i < iend) {
  			var j = 0;
  			var k = 0;
  			var c = 0;

  			j = 0;	
  			while (j < nwa) {                                            // Clear tmp A matrix 
  			  val ja = j * nrows;
  			  c = 0; 
  			  while (c < nrows) {
  			    aa(c + ja) = 0;
  			    c += 1;
  			  }
  			  j+= 1;
  			}
  			
  			k = 0;
  			while (k < nwb) {                                            // Loop over B words
  				c = 0; 
  			  while (c < nrows) {                                        // Clear tmp B vector
  			    bb(c) = 0;
  			    c += 1;
  			  }
  				val ib = nrows * WB(k+i*nwb);                              // Get the B word as an array offset. 
  				j = 0;
  				while (j < nwa) {                                          // Now iterate over A words. 
  					val ia = nrows * WA(j+i*nwa); 		                       // Get an A word offset 
  					
  					var cv = 0f;
  					c = 0;
  					while (c < nrows) {                                      // Inner product between A and B columns
  						cv += A(c + ia) * B(c + ib);
  						c += 1;
  					}
  					
  					if (cv > 16.0f) {                                        // Guarded logistic function
  						cv = 1.0f;
  					} else if (cv < -16.0f) {
  						cv = 0.0f;
  					} else {
  						cv = math.exp(cv).toFloat;
  						cv = cv / (1.0f + cv);
  					}
  					sum += math.log(math.max(1-cv, 1e-20));                                            
  					j += 1;
  				}
  				k += 1;
  			}
  			i += 1;
  		}
  		sum;
  	}).reduce(_+_);
  }
  
  def evalNegCPUslice(nrows:Int, nwords:Int, nwa:Int, nwb:Int, WA:Array[Int], WB:Array[Int], modelmats:Array[Mat], nthreads:Int,
     islice:Int, nslices:Int, maxCols:Int, nHead:Int, dualMode:Boolean):Double = {

    val arrayOffset = if (dualMode) 1 else 0;
  	(0 until nthreads).par.map((ithread:Int) => {
  		val istart = ((1L * nwords * ithread) / nthreads).toInt;
  		val iend = ((1L * nwords * (ithread+1)) / nthreads).toInt;
  		val aa = new Array[Float](nwa * nrows);
  		val bb = new Array[Float](nrows);
  		var sum = 0.0;
  		var i = istart;
  		while (i < iend) {
  			var j = 0;
  			var k = 0;
  			var c = 0;

  			j = 0;	
  			while (j < nwa) {                                            // Clear tmp A matrix 
  			  val ja = j * nrows;
  			  c = 0; 
  			  while (c < nrows) {
  			    aa(c + ja) = 0;
  			    c += 1;
  			  }
  			  j+= 1;
  			}
  			
  			k = 0;
  			while (k < nwb) {                                            // Loop over B words
  				c = 0; 
  			  while (c < nrows) {                                        // Clear tmp B vector
  			    bb(c) = 0;
  			    c += 1;
  			  }
  				val ibc = WB(k+i*nwb);                              // Get the B word as an array offset. 
  				val (mb, ib, bismine, bishead) = mapIndx(ibc, islice, nslices, nHead, maxCols, nrows, arrayOffset);
  				if (bismine || bishead) {
  					val B = modelmats(2*mb).asInstanceOf[FMat].data;
  					j = 0;
  					while (j < nwa) {                                          // Now iterate over A words. 
  						val iac = WA(j+i*nwa); 		                       // Get an A word offset 
  						val (ma, ia, aismine, aishead) = mapIndx(iac, islice, nslices, nHead, maxCols, nrows, arrayOffset);
  						if (aismine || aishead) {
  							val A = modelmats(2*ma+1).asInstanceOf[FMat].data;
  							var cv = 0f;
  							c = 0;
  							while (c < nrows) {                                      // Inner product between A and B columns
  								cv += A(c + ia) * B(c + ib);
  								c += 1;
  							}
  							if (cv > 16.0f) {                                        // Guarded logistic function
  								cv = 1.0f;
  							} else if (cv < -16.0f) {
  								cv = 0.0f;
  							} else {
  								cv = math.exp(cv).toFloat;
  								cv = cv / (1.0f + cv);
  							}
  							sum += math.log(math.max(1-cv, 1e-20)); 
  						}
  						j += 1;
  					}
  				}
  				k += 1;
  			}
  			i += 1;
  		}
  		sum;
  	}).reduce(_+_);
  }
  
  
  def mkModel(fopts:Model.Opts) = {
    new Word2Vec(fopts.asInstanceOf[Word2Vec.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
    new ADAGrad(nopts.asInstanceOf[ADAGrad.Opts])
  } 
  
  def mkRegularizer(nopts:Mixin.Opts):Array[Mixin] = {
    Array(new L1Regularizer(nopts.asInstanceOf[L1Regularizer.Opts]))
  }
    
  class LearnOptions extends Learner.Options with Word2Vec.Opts with MatDS.Opts with ADAGrad.Opts with L1Regularizer.Opts;
  
  def learner(mat0:Mat, targ:Mat) = {
    val opts = new LearnOptions;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1);
  	val nn = new Learner(
  	    new MatDS(Array(mat0, targ), opts), 
  	    new Word2Vec(opts), 
  	    null,
  	    null, 
  	    opts)
    (nn, opts)
  }
  
  class FDSopts extends Learner.Options with Word2Vec.Opts with FilesDS.Opts with ADAGrad.Opts with L1Regularizer.Opts
  
  def learner(fn1:String):(Learner, FDSopts) = learner(List(FilesDS.simpleEnum(fn1,1,0)));
  
  def learner(fnames:List[(Int)=>String]):(Learner, FDSopts) = {   
    val opts = new FDSopts
    opts.fnames = fnames;
    opts.batchSize = 100000;
    opts.eltsPerSample = 500;
    implicit val threads = threadPool(4);
    val ds = new FilesDS(opts);
  	val nn = new Learner(
  			ds, 
  	    new Word2Vec(opts), 
  	    null,
  	    null, 
  	    opts)
    (nn, opts)
  }
  
  def predictor(model0:Model, mat0:Mat, preds:Mat):(Learner, LearnOptions) = {
    val model = model0.asInstanceOf[Word2Vec];
    val opts = new LearnOptions;
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    if (mat0.asInstanceOf[AnyRef] != null) opts.putBack = 1;
    
    val newmod = new Word2Vec(opts);
    newmod.refresh = false;
    newmod.copyFrom(model);
    val mopts = model.opts.asInstanceOf[Word2Vec.Opts];
    opts.dim = mopts.dim;
    opts.vocabSize = mopts.vocabSize;
    opts.nskip = mopts.nskip;
    opts.nneg = mopts.nneg;
    opts.nreuse = mopts.nreuse;
    val nn = new Learner(
        new MatDS(Array(mat0, preds), opts), 
        newmod, 
        null,
        null, 
        opts);
    (nn, opts)
  }
  
   def predictor(model0:Model, mat0:Mat):(Learner, LearnOptions) = {
    val model = model0.asInstanceOf[Word2Vec];
    val opts = new LearnOptions;
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)    
    val newmod = new Word2Vec(opts);
    newmod.refresh = false;
    newmod.copyFrom(model);
    val mopts = model.opts.asInstanceOf[Word2Vec.Opts];
    opts.dim = mopts.dim;
    opts.vocabSize = mopts.vocabSize;
    opts.nskip = mopts.nskip;
    opts.nneg = mopts.nneg;
    opts.nreuse = mopts.nreuse;
    opts.maxArraySize = mopts.maxArraySize;
    opts.iSlice = mopts.iSlice;
    opts.nSlices = mopts.nSlices;
    opts.nHeadTerms = mopts.nHeadTerms;
    val nn = new Learner(
        new MatDS(Array(mat0), opts), 
        newmod, 
        null,
        null, 
        opts);
    (nn, opts)
  }
  
  class LearnParOptions extends ParLearner.Options with Word2Vec.Opts with FilesDS.Opts with ADAGrad.Opts;
  
  def learnPar(fn1:String):(ParLearnerF, LearnParOptions) = {learnPar(List(FilesDS.simpleEnum(fn1,1,0)))}
  
  def learnPar(fnames:List[(Int) => String]):(ParLearnerF, LearnParOptions) = {
    val opts = new LearnParOptions;
    opts.batchSize = 10000;
    opts.lrate = 1f;
    opts.fnames = fnames;
    implicit val threads = threadPool(4)
    val nn = new ParLearnerF(
        new FilesDS(opts), 
        opts, mkModel _,
        null, null,
        null, null, 
        opts)
    (nn, opts)
  }
  
  // Read a Google Word2Vec model file in binary or text format. 
  
  def readGoogleW2V(fname:String, dict:Dict, n:Int, binary:Boolean = false):FMat = {
    val ins = HMat.getInputStream(fname, 0);
    val din = new DataInputStream(ins);
    val sin = new Scanner(din);
    val header = sin.nextLine
    val dims = header.split(" ");
    val nr = dims(0).toInt;
    val dim = dims(1).toInt;
    val model = FMat(dim, n);

    var i = 0;
    while (i < nr) {
    	val word = sin.next;
    	val icol = dict(word);
    	val saveIt = (icol >= 0 && icol < n);
    	var j = 0;
    	while (j < dim) {
    		val v = if (binary) {
    			din.readFloat;
    		} else {
    			sin.nextFloat;
    		}
    		if (saveIt) model(j, icol) = v;
    		j += 1;
    	}
    	sin.nextLine;
    	i += 1;
    	if (i % 1000 == 0) println("i=%d %s" format (i, word))
    }
    model;
  }
  
  // Write a Google Word2Vec model file in binary or text format. 
  
  def saveGoogleW2V(dict:CSMat, mod:FMat, fname:String, binary:Boolean = false) = {
  	val outs = HMat.getOutputStream(fname, 0);
  	val dout = new DataOutputStream(outs);
  	val fout = new PrintWriter(dout);
  	val cr = String.format("\n");
  	fout.print(mod.ncols.toString + " " + mod.nrows.toString + cr);
  	fout.flush;
  	var i = 0;
  	while (i < mod.ncols) {
  		fout.print(dict(i)+ " ");
  		fout.flush;
  		var nwritten = 0;
  		var j = 0;
  		while (j < mod.nrows) {
  			if (binary) {
  			  dout.writeFloat(mod(j,i));
  			} else {
  			  dout.writeBytes("%g " format mod(j,i));
  			}
  			j += 1;
  		}
  		i += 1;
  		dout.writeBytes(cr);
  	}
  	dout.close;
};

}


