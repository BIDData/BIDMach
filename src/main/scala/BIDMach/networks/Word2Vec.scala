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
  
  def procPositives(nskip:Int, words:Mat, model1:Mat, model2:Mat, lrate:Float) = {
    val nrows = model1.nrows;
    val ncols = model1.ncols;
    val nwords = words.ncols;
    (words, model1, model2) match {
      case (w:GIMat, m1:GMat, m2:GMat) => CUMACH.word2vecPos(nrows, nwords, nskip, w.data, m1.data, m2.data, lrate);
      case (w:IMat, m1:FMat, m2:FMat) => if (Mat.useMKL) {
        CPUMACH.word2vecPos(nrows, nwords, nskip, w.data, m1.data, m2.data, lrate, Mat.numThreads);
      } else {
        procPosCPU(nrows, nwords, nskip, w.data, m1.data, m2.data, lrate, Mat.numThreads);
      }
    }
  }
  
  def procNegatives(nwa:Int, nwb:Int, wordsa:Mat, wordsb:Mat, modela:Mat, modelb:Mat, lrate:Float) = {
    val nrows = modela.nrows;
    val ncols = modela.ncols;
    val nwords = wordsa.ncols;
    (wordsa, wordsb, modela, modelb) match {
      case (wa:GIMat, wb:GIMat, ma:GMat, mb:GMat) => CUMACH.word2vecNeg(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, lrate);
      case (wa:IMat, wb:IMat, ma:FMat, mb:FMat) => if (Mat.useMKL) {
        CPUMACH.word2vecNeg(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, lrate, Mat.numThreads);
      } else {
      	procNegCPU(nrows, nwords, nwa, nwb, wa.data, wb.data, ma.data, mb.data, lrate, Mat.numThreads);
      }
    }
  }
  

  def procPosCPU(nrows:Int, ncols:Int, skip:Int, W:Array[Int], A:Array[Float], B:Array[Float], lrate:Float, nthreads:Int) = {

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

    	  val ia = W(i);
    	  c = 0;
    	  while (c < nrows) {                                    // Current word
    	  	Atmp(c) = 0;                                         // delta for the A matrix (maps current and random words). 
    	  	c += 1;
    	  }
    	  j = -skip;
    	  while (j <= skip) {
    	  	cv = 0f;
    	  	if (j != 0 && i + j >= 0 && i + j < ncols) {
    	  		val ib = W(i + j);
    	  		c = 0;
    	  		while (c < nrows) {
    	  			cv += A(c + ia) * B(c + ib);
    	  			c += 1;
    	  		}

    	  		if (cv > 16.0f) {
    	  			cv = 1.0f;
    	  		} else if (cv < -16.0f) {
    	  			cv = 0.0f;
    	  		} else {
    	  			cv = math.exp(cv).toFloat;
    	  			cv = 1.0f / (1.0f + cv);
    	  		}
    	  		cv = lrate * (1.0f - cv);

    	  		c = 0;
    	  		while (c < nrows) {
    	  			Atmp(c) += cv * B(c + ib);
    	  			B(c + ia) += cv * A(c + ia);
    	  			c += 1;
    	  		}
    	  	}
    	  	j += 1;
    	  }
    	  c = 0;
    	  while (c < nrows) {
    	  	A(c + ia) += Atmp(c);
    	  	c += 1;
    	  }
    	  i += 1;
    	}
    });
  }

  
  def procNegCPU(nrows:Int, nwords:Int, nwa:Int, nwb:Int, WA:Array[Int], WB:Array[Int], A:Array[Float], B:Array[Float], lrate:Float, nthreads:Int) = {

  	(0 until nthreads).par.map((ithread:Int) => {
  		val istart = ((1L * nwords * ithread) / nthreads).toInt;
  		val iend = ((1L * nwords * (ithread+1)) / nthreads).toInt;
  		val Btmp = new Array[Float](nrows);
  		var i = istart;
  		while (i < iend) {
  			var j = 0;
  			var k = 0;
  			var c = 0;

  			k = 0;
  			while (k < nwb) {
  				val ib = nrows * WB(k+i*nwb);
  				c = 0;
  				while (c < nrows) {
  				  Btmp(c) = 0;
  				  c += 1;
  				}
  				j = 0;
  				while (j < nwa) {
  					val ia = nrows * WA(j+i*nwb); 					
  					var cv = 0f;
  					c = 0;
  					while (c < nrows) {
  						cv += A(c + ia) * B(c + ib);
  						c += 1;
  					}
  					if (cv > 16.0f) {
  						cv = 1.0f;
  					} else if (cv < -16.0f) {
  						cv = 0.0f;
  					} else {
  						cv = math.exp(cv).toFloat;
  						cv = 1.0f / (1.0f + cv);
  					}
  					cv = - cv * lrate;
  					c = 0;
  					while (c < nrows) {
  						Btmp(c) += cv * A(c + ia);
  						A(c + ia) += cv * B(c + ib);
  						c += 1;
  					}
  					j += 1;
  				}
  				c = 0;
  				while (c < nrows) {
  					B(c + ib) = Btmp(c)
  					c += 1;
  				}
  				k += 1;
  			}
  			i += 1;
  		}
  	});
  }
}


