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
  var minusone:Mat = null;

  override def init() = {
    mats = datasource.next;
	  var nfeats = mats(0).nrows;
	  var ncols = mats(0).ncols;
	  datasource.reset;
    if (refresh) {
    	setmodelmats(new Array[Mat](2));
    	modelmats(0) = convertMat(zeros(opts.dim, nfeats));
    	modelmats(1) = convertMat(rand(opts.dim, nfeats));
    }
    val nskip = opts.nskip;
    val nwindow = nskip * 2 + 1;
    wordtab = convertMat(max(0, min(ncols, iones(nwindow, 1) * irow(1 -> (ncols+1)) + icol((- nskip) -> nskip))));
    randpermute = convertMat(zeros(nwindow, ncols));
    minusone = convertMat(irow(-1));
  }
  
  
  def dobatch(gmats:Array[Mat], ipass:Int, pos:Long):Unit = {
    val words = gmats(0);
    if (firstPos < 0) firstPos = pos;
    val nsteps = 1f * pos / firstPos;
    val lrate = opts.lrate * math.pow(nsteps, opts.texp).toFloat;
    
    Word2Vec.procPositives(opts.nskip, gmats(0), modelmats(0), modelmats(1), lrate);
    
    val iwords = minusone \ words;
    val cwords = iwords(wordtab);
    rand(randpermute);
//    kvSort(randpermute.contents, cwords.contents);
    
   
  }
  
  def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat = {  
  	zeros(1,1)
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
    
  }
  
  class Options extends Opts {}
  
  def procPositives(nskip:Int, words:Mat, model1:Mat, model2:Mat, lrate:Float):Int = {
    val nrows = model1.nrows;
    val ncols = model1.ncols;
    val nwords = words.ncols;
    (words, model1, model2) match {
      case (w:GIMat, m1:GMat, m2:GMat) => CUMACH.word2vecPos(nrows, nwords, nskip, w.data, m1.data, m2.data, lrate);
      case (w:IMat, m1:FMat, m2:FMat) => if (Mat.useMKL) {
        CPUMACH.word2vecPos(nrows, nwords, nskip, w.data, m1.data, m2.data, lrate, Mat.numThreads); 0;
      } else {
        procPosCPU(nrows, nwords, nskip, w.data, m1.data, m2.data, lrate, Mat.numThreads);
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
  

  def procPosCPU(nrows:Int, ncols:Int, skip:Int, W:Array[Int], A:Array[Float], B:Array[Float], lrate:Float, nthreads:Int):Int = {

    (0 until nthreads).par.map((ithread:Int) => {
    	val istart = ((1L * ithread * ncols)/nthreads).toInt;
    	val iend = ((1L * (ithread+1) * ncols)/nthreads).toInt;
    	val Atmp = new Array[Float](nrows);
    	var i = istart;
    	while (i < iend) {
    	  var j = 0;
    	  var k = 0;
    	  var c = 0;
    	  var cv = 0f;

    	  val ia = nrows * W(i);                                       // Get the current word (as a model offset). 
    	  if (ia >= 0) {                                               // Check for OOV words
    	  	c = 0;
    	  	while (c < nrows) {                                        // Current word
    	  		Atmp(c) = 0;                                             // delta for the A matrix (maps current and random words). 
    	  		c += 1;
    	  	}
    	  	j = -skip;
    	  	while (j <= skip) {                                        // Iterate over neighbors in the skip window
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
    	  					cv = 1.0f / (1.0f + cv);
    	  				}
    	  				cv = lrate * (1.0f - cv);                            // Subtract prediction from target (1.0), and scale by learning rate. 

    	  				c = 0;
    	  				while (c < nrows) {
    	  					Atmp(c) += cv * B(c + ib);                         // Compute backward derivatives for A and B. 
    	  					B(c + ib) += cv * A(c + ia);
    	  					c += 1;
    	  				}
    	  			}
    	  		}
    	  		j += 1;
    	  	}
    	  	c = 0;
    	  	while (c < nrows) {                                        // Add derivative for A to A. 
    	  		A(c + ia) += Atmp(c);
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
  		val ia = new Array[Int](nwa);
  		val ib = new Array[Int](nwb);
  		val bb = new Array[Float](nwb * nrows);
  		val aa = new Array[Float](nrows);
  		val prods = new Array[Float](nwa*nwb);
  		var i = istart;
  		while (i < iend) {
  			var j = 0;
  			var k = 0;
  			var c = 0;

  			k = 0;
  			while (k < nwb) {                                            // Iterate over "B" (context) words.  
  				ib(k) = nrows * WB(k+i*nwb);                               // Get the word and check it. 
  				if (ib(k) >= 0) {
  					j = 0;
  					while (j < nwa) {                                        // Now iterate over "A" (random) words. 
  						ia(j) = nrows * WA(j+i*nwb); 		                       // Get an A word and check it. 
  						var cv = 0f;
  						if (ia(j) >= 0) {
  							c = 0;
  							while (c < nrows) {                                  // Inner product between A and B columns
  								cv += A(c + ia(j)) * B(c + ib(k));
  								c += 1;
  							}
  							if (cv > 16.0f) {                                    // Guarded logistic function
  								cv = 1.0f;
  							} else if (cv < -16.0f) {
  								cv = 0.0f;
  							} else {
  								cv = math.exp(cv).toFloat;
  								cv = 1.0f / (1.0f + cv);
  							}
  							cv = - cv * lrate;                                   // Scale derivative by learning rate. 
  						}
  						prods(j + k * nwa) = cv;
  						j += 1;
  					}
  				} 
  				k += 1;
  			}
  			
  			k = 0;
  			while (k < nwb) {
  			  c = 0;
  			  while (c < nrows) {
  			  	bb(c + k * nrows) = 0; 
  			  	c += 1;
  			  }
  			  k += 1;
  			}
  			j = 0;
  			while (j < nwa) {
  				if (ia(j) >= 0) {
  				  c = 0; while (c < nrows) {aa(c) = 0; c += 1;}
  					k = 0;
  					while (k < nwb) {
  						if (ib(k) >= 0) {
  							val v = prods(j + k * nwa);
  							c = 0;
  							while (c < nrows) {
  							  bb(c + k * nrows) += v * A(c + ia(j));
  							  aa(c) += v * B(c + ib(k));
  								c += 1;
  							}
  						}
  						k += 1;
  					}
  					c = 0;
  					while (c < nrows) {
  						A(c + ia(j)) += aa(c);
  						c += 1;
  					}
  				}
  				j += 1;
  			}
  			k = 0;
  			while (k < nwb) {
  			  c = 0;
  			  while (c < nrows) {
  			    B(c + ib(k)) += bb(c + k * nrows);
  			    c += 1;
  			  }
  			  k += 1;
  			}
  			i += 1;
  		}
  	});
  	0;
  }
}


