package BIDMach.allreduce

import scala.util.Random
import BIDMat.{Mat,FMat,IMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.VSL
import edu.berkeley.bid.VSL._
import edu.berkeley.bid.RAND
import edu.berkeley.bid.RAND._

class RandPerm(seed:Int) { 
  val BRNG:Int = if (!Mat.useMKLRand) 0 else BRNG_MCG31
  val METHOD:Int = 0
  val stream = if (Mat.useMKLRand) new VSL() else null;
  val errcode = if (Mat.useMKLRand) vslNewStream(stream, BRNG, seed) else 0;
  val engine = if (Mat.useSTLRand) new RAND() else null;
  val errcode2 = if (Mat.useSTLRand) newEngine(engine, 0, seed) else 0;
  val myrand = new java.util.Random(seed);

  def rand(mat:FMat) = { 
    val minv = 0f;
    val maxv = 1.0f;
    if (Mat.useMKLRand) {
      vsRngUniform( METHOD, stream, mat.length, mat.data, minv, maxv );
    } else if (Mat.useSTLRand) {
      SUniform(0, engine, mat.length, mat.data, minv, maxv);
    } else {
      var i = 0; val len = mat.length; val odata = mat.data; 
      while (i < len) {odata(i) = myrand.nextFloat; i += 1}     
    }
    Mat.nflops += 10L*mat.length;
    mat;
  }

  def permcols1(data1:FMat, rr:FMat, len:Int) = { 
    val nr = rr.nrows;
    val nc = rr.ncols;
    var i = 0;
    while (i < nc) { 
      var j = 0;
      val joff = i * nr;
      val nrr = if (i < nc-1) nr else nr - (data1.length - len);
      while (j < (nrr - 1)) { 
	    val indx = joff + math.min(nrr - 1, j + (rr.data(j+joff) * (nrr - j)).toInt);
	    val tmpv = data1.data(j + joff);
	    data1.data(j + joff) = data1.data(indx);
	    data1.data(indx) = tmpv;
	    j += 1;
      }
      i += 1;
    }
  }

  def ipermcols1(data1:FMat, rr:FMat, len:Int) = { 
    val nr = rr.nrows;
    val nc = rr.ncols;
    var i = nc - 1
    while (i >= 0) { 
      val joff = i * nr;
      val nrr = if (i < nc-1) nr else nr - (data1.length - len);
      var j = nrr - 2;
      while (j >= 0) { 
	    val indx = joff + math.min(nrr - 1, j + (rr.data(j+joff) * (nrr - j)).toInt);
	    val tmpv = data1.data(j + joff);
	    data1.data(j + joff) = data1.data(indx);
	    data1.data(indx) = tmpv;
	    j -= 1;
      }
      i -= 1;
    }
  }

  def permcols2(data1:FMat, rr:FMat, len:Int) = { 
    val nr = rr.nrows;
    val nc = rr.ncols;
    var i = 0;
    while (i < nc) { 
      var j = 0;
      val joff = i * nr;
      val nrr = if (i < nc - (data1.length - len)) nr else (nr - 1);
      while (j < (nrr - 1)) { 
	    val indx = joff + math.min(nrr - 1, j + (rr.data(j+joff) * (nrr - j)).toInt);
	    val tmpv = data1.data(j + joff);
	    data1.data(j + joff) = data1.data(indx);
	    data1.data(indx) = tmpv;
	    j += 1;
      }
      i += 1;
    }
  }

  def ipermcols2(data1:FMat, rr:FMat, len:Int) = { 
    val nr = rr.nrows;
    val nc = rr.ncols;
    var i = nc - 1;
    while (i >= 0) { 
      val joff = i * nr;
      val nrr = if (i < nc - (data1.length - len)) nr else (nr - 1);
      var j = nrr - 2;
      while (j >= 0) { 
	    val indx = joff + math.min(nrr - 1, j + (rr.data(j+joff) * (nrr - j)).toInt);
	    val tmpv = data1.data(j + joff);
	    data1.data(j + joff) = data1.data(indx);
	    data1.data(indx) = tmpv;
	    j -= 1;
      }
      i -= 1;
    }
  }

  def randperm(mat:FMat) = { 
    tic
    val dd = math.ceil(math.sqrt(mat.length)).toInt;
    val data1 = zeros(dd, dd);
    val rand1 = zeros(dd, dd);
    val rand2 = zeros(dd, dd);
    val out = zeros(mat.nrows, mat.ncols);
    val t0 = toc;
    rand(rand1);
    rand(rand2);
    val t1 = toc;
    var i = 0;
    while (i < mat.length) { 
      data1.data(i) = mat.data(i);
      i += 1;
    }
    val t2 = toc;
    permcols1(data1, rand1, mat.length)
    val t3 = toc;
    val datat = data1.t;
    val t4 = toc;
    permcols2(datat, rand2, mat.length)
    val t5 = toc;

    i = 0;
	var iout = 0;
    while (i < dd) { 
      val nr = if (i < dd - (data1.length - mat.length)) dd else (dd - 1);
      var j = 0;
      while (j < nr) { 
	    out.data(iout) = datat.data(j + i * dd);
        iout += 1;
        j += 1;
      }
      i += 1;
    }
    val t6 = toc;
    (out, row(t0, t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t6));
  }

  def irandperm(mat:FMat) = { 
    tic
    val dd = math.ceil(math.sqrt(mat.length)).toInt;
    val data1 = zeros(dd, dd);
    val rand1 = zeros(dd, dd);
    val rand2 = zeros(dd, dd);
    val out = zeros(mat.nrows, mat.ncols);
    val t0 = toc;
    rand(rand1);
    rand(rand2);
    val t1 = toc;
    var i = 0;
	var iin = 0;
    while (i < dd) { 
      val nr = if (i < dd - (data1.length - mat.length)) dd else (dd - 1);
      var j = 0;
      while (j < nr) { 
        data1.data(j + i * dd) = mat.data(iin);
        iin += 1;
        j += 1;
      }
      i += 1;
    }

    val t2 = toc;
    ipermcols2(data1, rand2, mat.length)
    val t3 = toc;
    val datat = data1.t;
    val t4 = toc;
    ipermcols1(datat, rand1, mat.length)
    val t5 = toc;
    i = 0;
    while (i < mat.length) { 
      out.data(i) = datat.data(i);
      i += 1;
    }

    val t6 = toc;
    (out, row(t0, t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t6));
  }

  def randperm2(mat:FMat) = { 
    tic
    val out = zeros(mat.nrows, mat.ncols);
    val rand1 = zeros(mat.length, 1);
    val t0 = toc;
    rand(rand1);
    val t1 = toc;
    var i = 0;
    while (i < mat.length) { 
      out.data(i) = mat.data(i);
      i += 1;
    }
    val t2 = toc;
    permcols1(out, rand1, mat.length)
    val t3 = toc;
    (out, row(t0, t1-t0, t2-t1, t3-t2, t3));
  }

  def irandperm2(mat:FMat) = { 
    tic
    val out = zeros(mat.nrows, mat.ncols);
    val rand1 = zeros(mat.length, 1);
    val t0 = toc;
    rand(rand1);
    val t1 = toc;
    var i = 0;
    while (i < mat.length) { 
      out.data(i) = mat.data(i);
      i += 1;
    }
    val t2 = toc;
    ipermcols1(out, rand1, mat.length)
    val t3 = toc;
    (out, row(t0, t1-t0, t2-t1, t3-t2, t3));
  }
}

object RandPerm { 
}
