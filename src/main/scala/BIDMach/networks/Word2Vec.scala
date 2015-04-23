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
  var nfeats = 0;
  var ncols = 0;
  var expt = 0f;

  override def init() = {
    mats = datasource.next;
	  nfeats = mats(0).nrows;
	  ncols = mats(0).ncols;
	  datasource.reset;
    if (refresh) {
    	setmodelmats(new Array[Mat](2));
    	modelmats(0) = convertMat(zeros(opts.dim, nfeats));
    	modelmats(1) = convertMat(rand(opts.dim, nfeats));
    }
    val nskip = opts.nskip;
    val nwindow = nskip * 2 + 1;
    val skipcol = icol((- nskip) -> nskip);
    expt = 1f / (1f - opts.wexpt);
    wordtab = convertMat(max(0, min(ncols+1, iones(nwindow, 1) * irow(1 -> (ncols+1)) + skipcol)));
    wordmask = convertMat(skipcol * iones(1, ncols));
    randpermute = convertMat(zeros(nwindow, ncols));
    ubound = convertMat(zeros(1, ncols));
    minusone = convertMat(irow(-1));
    allones = convertMat(iones(1, ncols));
    randwords = convertMat(zeros(opts.nneg, ncols));
  }
  
  def dobatch(gmats:Array[Mat], ipass:Int, pos:Long):Unit = {
    if (firstPos < 0) firstPos = pos;
    val nsteps = 1f * pos / firstPos;
    val lrate = opts.lrate * math.pow(nsteps, opts.texp).toFloat;
    val (words, lb, ub, trandwords, goodwords) = wordMats(mats, ipass, pos);
  
    Word2Vec.procPositives(opts.nskip, words, lb, ub, modelmats(0), modelmats(1), lrate);
    Word2Vec.procNegatives(opts.nneg, opts.nreuse, trandwords, goodwords, modelmats(0), modelmats(1), lrate); 
  }
  
  def evalbatch(mats:Array[Mat], ipass:Int, pos:Long):FMat = {
  	val (words, lb, ub, trandwords, goodwords) = wordMats(mats, ipass, pos);
  	val epos = Word2Vec.evalPositives(opts.nskip, words, lb, ub, modelmats(0), modelmats(1));
    val eneg = Word2Vec.evalNegatives(opts.nneg, opts.nreuse, trandwords, goodwords, modelmats(0), modelmats(1));
  	val score = ((epos + eneg / opts.nneg) / words.ncols);
  	row(score)
  }
  
  def wordMats(mats:Array[Mat], ipass:Int, pos:Long):(Mat, Mat, Mat, Mat, Mat) = {
  
    val wordsens = mats(0);
    val words = wordsens(0,?);
    
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
    rand(randpermute);
    randpermute ~ pgoodwords + (pgoodwords ∘ randpermute - 1);                 // set the values for bad words to -1. 
    val (vv, ii) = sortdown2(randpermute(?));                                  // Permute the good words
    val ngood = sum(vv > 0f).dv.toInt;                                         // Count of the good words
    val ngoodcols = ngood / opts.nreuse;                                       // Number of good columns
    
    rand(randwords);                                                           // Compute some random negatives
    val irandwords = min(nfeats-1, int(nfeats * (randwords ^ expt)));
    
    val trandwords = irandwords.view(opts.nneg, ngoodcols);                    // shrink the matrices to the available data
    val goodwords = cwords(ii).view(opts.nreuse, ngoodcols);
    
    (words, lb, ub, trandwords, goodwords);
  }
}

object Word2Vec  {
  trait Opts extends Model.Opts {
    var aopts:ADAGrad.Opts = null;
    var nskip = 5;
    var lrate = 0.1f;
    var texp = 0.5f;
    var nneg = 5;
    var nreuse = 5;    
    var wexpt = 0.75f;
  }
  
  class Options extends Opts {}
  
  def procPositives(nskip:Int, words:Mat, lbound:Mat, ubound:Mat, model1:Mat, model2:Mat, lrate:Float):Int = {
    val nrows = model1.nrows;
    val ncols = model1.ncols;
    val nwords = words.ncols;
    (words, lbound, ubound, model1, model2) match {
      case (w:GIMat, lb:GIMat, ub:GIMat, m1:GMat, m2:GMat) => CUMACH.word2vecPos(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, lrate);
      case (w:IMat, lb:IMat, ub:IMat, m1:FMat, m2:FMat) => if (Mat.useMKL) {
        CPUMACH.word2vecPos(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, lrate, Mat.numThreads); 0;
      } else {
        procPosCPU(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, lrate, Mat.numThreads);
      }
    }
  }
  
  def procNegatives(nwa:Int, nwb:Int, wordsa:Mat, wordsb:Mat, modela:Mat, modelb:Mat, lrate:Float):Int = {
    val nrows = modela.nrows;
    val ncols = modela.ncols;
    val nwords = wordsa.ncols;
    (wordsa, wordsb, modela, modelb) match {
      case (wa:GIMat, wb:GIMat, ma:GMat, mb:GMat) => CUMACH.word2vecNeg(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, lrate);
      case (wa:IMat, wb:IMat, ma:FMat, mb:FMat) => if (Mat.useMKL) {
        CPUMACH.word2vecNeg(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, lrate, Mat.numThreads); 0;
      } else {
      	procNegCPU(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, lrate, Mat.numThreads);
      }
    }
  }
  

  def procPosCPU(nrows:Int, ncols:Int, skip:Int, W:Array[Int], LB:Array[Int], UB:Array[Int],
      A:Array[Float], B:Array[Float], lrate:Float, nthreads:Int):Int = {

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
    	  				cv = lrate * (1.0f - cv);                            // Subtract prediction from target (1.0), and scale by learning rate. 

    	  				c = 0;
    	  				while (c < nrows) {
    	  					daa(c) += cv * B(c + ib);                          // Compute backward derivatives for A and B. 
    	  					B(c + ib) += cv * A(c + ia);
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

  
  def procNegCPU(nrows:Int, nwords:Int, nwa:Int, nwb:Int, WA:Array[Int], WB:Array[Int], A:Array[Float], B:Array[Float], lrate:Float, nthreads:Int):Int = {

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
  					cv = - cv * lrate;                                       // Scale derivative by learning rate. 

  					val ja = j * nrows;
  					c = 0;
  					while (c < nrows) {                                      // Update the derivatives
  						aa(c + ja) += cv * B(c + ib);
  						bb(c) += cv * A(c + ia);
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
  
   def evalPositives(nskip:Int, words:Mat, lbound:Mat, ubound:Mat, model1:Mat, model2:Mat):Double = {
    val nrows = model1.nrows;
    val ncols = model1.ncols;
    val nwords = words.ncols;
    (words, lbound, ubound, model1, model2) match {
//      case (w:GIMat, lb:GIMat, ub:GIMat, m1:GMat, m2:GMat) => CUMACH.word2vecPos(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, lrate);
      case (w:IMat, lb:IMat, ub:IMat, m1:FMat, m2:FMat) => 
//        if (Mat.useMKL) {
//        CPUMACH.word2vecPos(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, lrate, Mat.numThreads); 0;
//      } else {
        evalPosCPU(nrows, nwords, nskip, w.data, lb.data, ub.data, m1.data, m2.data, Mat.numThreads);
//      }
    }
  }
  
  def evalNegatives(nwa:Int, nwb:Int, wordsa:Mat, wordsb:Mat, modela:Mat, modelb:Mat):Double = {
    val nrows = modela.nrows;
    val ncols = modela.ncols;
    val nwords = wordsa.ncols;
    (wordsa, wordsb, modela, modelb) match {
//      case (wa:GIMat, wb:GIMat, ma:GMat, mb:GMat) => CUMACH.word2vecNeg(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, lrate);
      case (wa:IMat, wb:IMat, ma:FMat, mb:FMat) => 
//        if (Mat.useMKL) {
//        CPUMACH.word2vecNeg(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, lrate, Mat.numThreads); 0;
//      } else {
      	evalNegCPU(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, Mat.numThreads);
//      }
    }
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
    	  				sum += math.log(cv);                            
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
  					sum += math.log(1-cv);                                            
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


