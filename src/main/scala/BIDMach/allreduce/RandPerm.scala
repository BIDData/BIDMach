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


  def randmove(mat:Array[Float], ranges:IMat) = { 
    val aranges = ranges \ mat.length;
    val ir = ranges.copy
    val or = ranges.copy
    var nr = ranges.length;
    val ncols = math.min(1+mat.length/nr, 123451);
    val rr = zeros(nr, ncols);
//    rand(rr);
    var icol = ncols;
    while (nr > 0) { 
      if (icol >= ncols) { 
	rand(rr);
	icol = 0;
      }
      // Shuffle
      var i = 0;
      while (i < nr - 1) { 
	val rv = rr.data(icol * rr.nrows + i);
	val indx = math.min(nr - 1, (rv * (nr - i)).toInt);
	val vv = mat(or.data(i));
	mat(or.data(i)) = mat(or.data(indx));
	mat(or.data(indx)) = vv;
	i += 1;
      }
      icol += 1;
      nr = 0;
      i = 0;
      while (i < ir.length) { 
	ir.data(i) = ir.data(i) + 1;
	if (ir.data(i) < aranges(i+1)) {
	  or.data(nr) = ir.data(i)
	  nr += 1;
	} 
	i += 1;
      }
    }
    mat;
  }

  def irandmove(mat:Array[Float], ranges:IMat) = { 
    val aranges = ranges \ mat.length;
    val ir = ranges.copy
    val or = ranges.copy
    var nr = ranges.length;
    val ncols = math.min(1+mat.length/nr, 123451);
    val rr = zeros(nr, ncols);
//    rand(rr);
    var icol = ncols;
    while (nr > 0) { 
      if (icol >= ncols) { 
	rand(rr);
	icol = 0;
      }
      // UnShuffle
      var i = nr - 2;
      while (i >= 0) { 
	val rv = rr.data(icol * rr.nrows + i);
	val indx = math.min(nr - 1, (rv * (nr - i)).toInt);
	val vv = mat(or.data(i));
	mat(or.data(i)) = mat(or.data(indx));
	mat(or.data(indx)) = vv;
	i -= 1;
      }
      icol += 1;
      nr = 0;
      i = 0;
      while (i < ir.length) { 
	ir.data(i) = ir.data(i) + 1;
	if (ir.data(i) < aranges(i+1)) {
	  or.data(nr) = ir.data(i)
	  nr += 1;
	} 
	i += 1;
      }
    }
    mat
  }

}

