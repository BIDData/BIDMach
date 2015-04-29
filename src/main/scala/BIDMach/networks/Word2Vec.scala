package BIDMach.networks

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
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

/**
 * 
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

  
  var test1:Mat = null;
  var test2:Mat = null;
  var test3:Mat = null;
  

  override def init() = {
    val mats = datasource.next;
	  nfeats = opts.vocabSize;
	  ncols = mats(0).ncols;
	  datasource.reset;
    if (refresh) {
    	setmodelmats(new Array[Mat](2));
    	modelmats(0) = convertMat((rand(opts.dim, nfeats) - 0.5f)/opts.dim);              // syn0 - context model
    	modelmats(1) = convertMat(zeros(opts.dim, nfeats));                               // syn1neg - target word model
    }
    val nskip = opts.nskip;
    val nwindow = nskip * 2 + 1;
    val skipcol = icol((- nskip) to nskip);
    expt = 1f / (1f - opts.wexpt);
    wordtab = convertMat(max(0, min(ncols+1, iones(nwindow, 1) * irow(1 -> (ncols+1)) + skipcol)));  // Indices for convolution matrix
    wordmask = convertMat(skipcol * iones(1, ncols));                                   // columns = distances from center word
    randpermute = convertMat(zeros(nwindow, ncols));                                    // holds random values for permuting negative context words
    ubound = convertMat(zeros(1, ncols));                                               // upper bound random matrix
    minusone = convertMat(irow(-1));
    allones = convertMat(iones(1, ncols));
    randwords = convertMat(zeros(1, (1.01 * opts.nneg * nskip * ncols / opts.nreuse).toInt)); // generates random negative words
    randsamp = convertMat(zeros(1, ncols));                                             // For sub-sampling frequent words
    val gopts = opts.asInstanceOf[ADAGrad.Opts];
    vexp = gopts.vexp.v;
    salpha = opts.wsample * math.log(nfeats).toFloat;
    if (useGPU) {
      retEvalPos = GMat(1,1);
      retEvalNeg = GMat(1,1);
    }
  }
  
  def dobatch(gmats:Array[Mat], ipass:Int, pos:Long):Unit = {
    if (firstPos < 0) firstPos = pos;
    val nsteps = 1f * pos / firstPos;
    val gopts = opts.asInstanceOf[ADAGrad.Opts];
    val lrate = gopts.lrate.dv.toFloat * math.pow(nsteps, - gopts.texp.dv).toFloat;
    val (words, lb, ub, trandwords, goodwords) = wordMats(gmats, ipass, pos);
    
    val lrpos = lrate.dv.toFloat;
//    val lrneg = lrpos/opts.nneg;  
    val lrneg = lrpos;
    procPositives(opts.nskip, words, lb, ub, modelmats(1), modelmats(0), lrpos);
    procNegatives(opts.nneg, opts.nreuse, trandwords, goodwords, modelmats(1), modelmats(0), lrneg); 
  }
  
  def evalbatch(gmats:Array[Mat], ipass:Int, pos:Long):FMat = {
  	val (words, lb, ub, trandwords, goodwords) = wordMats(gmats, ipass, pos);
  	val epos = evalPositives(opts.nskip, words, lb, ub, modelmats(1), modelmats(0));
    val eneg = evalNegatives(opts.nneg, opts.nreuse, trandwords, goodwords, modelmats(1), modelmats(0));
//  	val score = ((epos + eneg / opts.nneg) /opts.nskip / words.ncols);
  	val score = ((epos + eneg) / goodwords.length);
  	row(score)
  }
  
  def wordMats(mats:Array[Mat], ipass:Int, pos:Long):(Mat, Mat, Mat, Mat, Mat) = {
  
    val wordsens = mats(0);
    val words = wordsens(0,?);
    val wgood = words < opts.vocabSize;                                        // Find OOV words 
                                       
    rand(randsamp);                                                            // Take a random sample
    val wrat = float(words+1) * salpha;
    wrat ~ sqrt(wrat) + wrat;
    wgood ~ wgood ∘ int(randsamp < wrat);
    words ~ wgood + (wgood ∘ words - 1);                                       // Set OOV or skipped samples to -1
       
    rand(ubound);                                                              // get random upper and lower bounds   
    val ubrand = int(ubound * opts.nskip);
    val lbrand = - ubrand;
    
    val sentencenum = wordsens(1,?);                                           // Get the nearest sentence boundaries
    val lbsentence = - cumsumByKey(allones, sentencenum) + 1;
    val ubsentence = reverse(cumsumByKey(allones, reverse(sentencenum))) - 1;
    val lb = max(lbrand, lbsentence);                                          // Combine the bounds
    val ub = min(ubrand, ubsentence);
       
    val iwords = minusone \ words \ minusone;                                  // Build a convolution matrix.
    val cwords = iwords(wordtab);
    val pgoodwords = (wordmask >= lb) ∘ (wordmask <= ub) ∘ (cwords >= 0);      // Find words satisfying the bound
    val fgoodwords = float(pgoodwords);
    
    rand(randpermute);                                                         // Prepare a random permutation of context words for negative sampling
    randpermute ~ fgoodwords + (fgoodwords ∘ randpermute - 1);                 // set the values for bad words to -1. 
    val (vv, ii) = sortdown2(randpermute(?));                                  // Permute the good words
    val ngood = sum(vv > 0f).dv.toInt;                                         // Count of the good words
    val ngoodcols = ngood / opts.nreuse;                                       // Number of good columns
    
    rand(randwords);                                                           // Compute some random negatives
    val irandwords = min(nfeats-1, int(nfeats * (randwords ^ expt)));    
    val trandwords = irandwords.view(opts.nneg, ngoodcols);                    // shrink the matrices to the available data
    val cwi = cwords(ii);
    val goodwords = cwi.view(opts.nreuse, ngoodcols);
    
    test1 = words;
    test2 = trandwords;
    test3 = cwords;
    
    (words, lb, ub, trandwords, goodwords);
  }
  
  def procPositives(nskip:Int, words:Mat, lbound:Mat, ubound:Mat, model1:Mat, model2:Mat, lrate:Float) = {
    val nrows = model1.nrows;
    val ncols = model1.ncols;
    val nwords = words.ncols;
    Mat.nflops += 6L * nwords * nskip * nrows;
    (words, lbound, ubound, model1, model2) match {
      case (w:GIMat, lb:GIMat, ub:GIMat, m1:GMat, m2:GMat) => {
        val err = CUMACH.word2vecPos(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, lrate);
        if (err != 0)  throw new RuntimeException("CUMACH.word2vecPos error " + cudaGetErrorString(err));
      }
      case (w:IMat, lb:IMat, ub:IMat, m1:FMat, m2:FMat) => if (Mat.useMKL) {
        CPUMACH.word2vecPos(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, lrate, vexp, Mat.numThreads);
      } else {
        procPosCPU(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, lrate, vexp, Mat.numThreads);
      }
    }
  }
  
  def procNegatives(nwa:Int, nwb:Int, wordsa:Mat, wordsb:Mat, modela:Mat, modelb:Mat, lrate:Float) = {
    val nrows = modela.nrows;
    val ncols = modela.ncols;
    val nwords = wordsa.ncols;
    Mat.nflops += 6L * nwords * nwa * nwb * nrows;
    (wordsa, wordsb, modela, modelb) match {
      case (wa:GIMat, wb:GIMat, ma:GMat, mb:GMat) => {
        val err = CUMACH.word2vecNeg(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, lrate);
        if (err != 0) throw new RuntimeException("CUMACH.word2vecNeg error " + cudaGetErrorString(err));
      }
      case (wa:IMat, wb:IMat, ma:FMat, mb:FMat) => if (Mat.useMKL) {
        CPUMACH.word2vecNeg(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, lrate, vexp, Mat.numThreads);
      } else {
      	procNegCPU(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, lrate, vexp, Mat.numThreads);
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
        evalPosCPU(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, Mat.numThreads);
      }
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
      	evalNegCPU(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, Mat.numThreads);
      }
    }
  }

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
    	  val ascale = math.pow(iac, vexp).toFloat;
    	  val ia = nrows * iac;                                        // Get the current word (as a model matrix offset). 
    	  if (ia >= 0) {                                               // Check for OOV words
    	  	c = 0;
    	  	while (c < nrows) {                                        // Current word
    	  		daa(c) = 0;                                              // delta for the A matrix (maps current and negative words). 
    	  		c += 1;
    	  	}
    	  	j = LB(i);
    	  	while (j <= UB(i)) {                                       // Iterate over neighbors in the skip window
    	  		cv = 0f;
    	  		if (j != 0 && i + j >= 0 && i + j < ncols) {             // context word index is in range (and not current word).
    	  		  val ibc = W(i + j);
    	  		  val bscale = math.pow(ibc, vexp).toFloat;
    	  			val ib = nrows * ibc;                                  // Get the context word and check it
    	  			if (ib >= 0) {
    	  				c = 0;
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
    	  					daa(c) += (1+ascale) * cv * B(c + ib);                    // Compute backward derivatives for A and B with pseudo-ADAGrad scaling
    	  					B(c + ib) += (1+bscale) * cv * A(c + ia);
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
  			  val bscale = math.pow(ibc, vexp).toFloat;
  				val ib = nrows * ibc;                                      // Get the B word as an array offset. 
  				j = 0;
  				while (j < nwa) {                                          // Now iterate over A words. 
  				  val iac = WA(j+i*nwa);
  				  val ascale = math.pow(iac, vexp).toFloat;
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
  						aa(c + ja) += (1+ascale) * cv * B(c + ib);
  						bb(c) +=  (1+bscale) * cv * A(c + ia);
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
    	  		cv = 0f;
    	  		if (j != 0 && i + j >= 0 && i + j < ncols) {             // context word index is in range (and not current word).
    	  			val ib = nrows * W(i + j);                             // Get the context word and check it. 
    	  			if (ib >= 0) {
    	  				c = 0;
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
  }
  
  class Options extends Opts {}
  
  
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
  
  def learner(fn1:String, fn2:String):(Learner, FDSopts) = learner(List(FilesDS.simpleEnum(fn1,1,0),
  		                                                                  FilesDS.simpleEnum(fn2,1,0)));
  
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
    val model = model0.asInstanceOf[DNN];
    val opts = new LearnOptions;
    opts.batchSize = math.min(10000, mat0.ncols/30 + 1)
    opts.addConstFeat = model.opts.asInstanceOf[DataSource.Opts].addConstFeat;
    opts.putBack = 1;
    
    val newmod = new Word2Vec(opts);
    newmod.refresh = false;
    newmod.copyFrom(model)
    val nn = new Learner(
        new MatDS(Array(mat0, preds), opts), 
        newmod, 
        null,
        null, 
        opts);
    (nn, opts)
  }
  
  class LearnParOptions extends ParLearner.Options with DNN.Opts with FilesDS.Opts with ADAGrad.Opts with L1Regularizer.Opts;
  
  def learnPar(fn1:String, fn2:String):(ParLearnerF, LearnParOptions) = {learnPar(List(FilesDS.simpleEnum(fn1,1,0), FilesDS.simpleEnum(fn2,1,0)))}
  
  def learnPar(fnames:List[(Int) => String]):(ParLearnerF, LearnParOptions) = {
    val opts = new LearnParOptions;
    opts.batchSize = 10000;
    opts.lrate = 1f;
    opts.fnames = fnames;
    implicit val threads = threadPool(4)
    val nn = new ParLearnerF(
        new FilesDS(opts), 
        opts, mkModel _,
        opts, mkRegularizer _,
        opts, mkUpdater _, 
        opts)
    (nn, opts)
  }
}


