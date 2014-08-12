package BIDMach.models

// Random Forest model.
// Computes a matrix representing the Forest.
// The matrix encodes a 2 x nnodes x ntrees array.
// nnodes is the number of nodes in each tree, which is 2^(depth + 1) (-1). 
// The (0,X,Y) slice holds the index of the feature to test
// the (1,X,Y) slice holds the threshold to compare it with. 
// At leaf nodes, the (0,X,Y)-value is negative and the 
// (1,X,Y) value holds the majority category id for that node. 

import BIDMat.{SBMat,CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,LMat,Mat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.CUMAT
import BIDMach.models.RandForest._
import scala.util.hashing.MurmurHash3
import jcuda._

class RandomForest(override val opts:RandomForest.Opts) extends Model(opts) {
  
  val ITree = 0; val INode = 1; val JFeat = 2; val IFeat = 3; val IVFeat = 4; val ICat = 5
  
  var nnodes:Int = 0;
  var tmp1:Mat = null;
  var tmp2:Mat = null;
  var tc1:Mat = null;
  var tc2:Mat = null;
  var totalinds:LMat = null;
  var totalcounts:FMat = null;
  var ntrees = 0;
  var nsamps = 0;
  var nfeats = 0;
  var nvals = 0;
  var ncats = 0;
  val fieldlengths = izeros(1,6);
  
  def init() = {
    nnodes = math.pow(2, opts.depth + 1).toInt;  // Num nodes per tree. Keep as a power of two (dont subtract one), so blocks align better. 
    ntrees = opts.ntrees;
    nsamps = opts.nsamps;
    nvals = opts.nvals;
    val modelmat = if (opts.useGPU && Mat.hasCUDA > 0) gizeros(2 * nnodes, opts.ntrees) else izeros(2 * nnodes, opts.ntrees);
    modelmats = Array{modelmat};
    mats = datasource.next;
    nfeats = mats(0).nrows;
    ncats = mats(1).nrows;
    val nc = mats(0).ncols;
    val nnz = mats(1).nnz;
    tmp1 = lzeros(1, (opts.margin * nnz * ntrees * nsamps).toInt);
    tmp2 = lzeros(1, (opts.margin * nnz * ntrees * nsamps).toInt);
    tc1 = zeros(1, (opts.margin * nnz * ntrees * nsamps).toInt);
    tc2 = zeros(1, (opts.margin * nnz * ntrees * nsamps).toInt);
    datasource.reset;
    fieldlengths(ITree) = countbits(ntrees);
    fieldlengths(INode) = countbits(nnodes);
    fieldlengths(JFeat) = countbits(nsamps);
    fieldlengths(IFeat) = countbits(nfeats);
    fieldlengths(IVFeat) = countbits(nvals);
    fieldlengths(ICat) = countbits(ncats);
  } 
  
  def countbits(n:Int):Int = {
    var i = 0;
    var j = 1;
    while (j < n) {
      j *= 2;
      i += 1;
    }
    i
  }
  
  def doblock(gmats:Array[Mat], ipass:Int, i:Long) = {
    val sdata = gmats(0);
    val cats = gmats(1);
    val nnodes = gmats(2);
    val (inds, counts):(LMat, FMat) = (sdata, cats, nnodes, tmp1, tmp2, tc1) match {
        case (idata:IMat, scats:SMat, inodes:IMat, lt1:LMat, lt2:LMat, tc:FMat) => {
          val lt = treePack(idata, inodes, scats, lt1, nsamps, fieldlengths);
          sort(lt);    
          makeV(lt, lt2, tc);
        }
    }
    if (totalinds.asInstanceOf[AnyRef] != null) {
      val (inds1, counts1) = addV(inds, counts, totalinds, totalcounts);
      totalinds = inds1;
      totalcounts = counts1;
    } else {
      totalinds = inds;
      totalcounts = counts;
    }        
  }
  
  def evalblock(mats:Array[Mat], ipass:Int):FMat = {
    row(0)
  } 
  
  def rhash(v1:Int, v2:Int, v3:Int, nb:Int):Int = {
    math.abs(MurmurHash3.mix(MurmurHash3.mix(v1, v2), v3) % nb)
  }
  
  def packFields(itree:Int, inode:Int, jfeat:Int, ifeat:Int, ivfeat:Int, icat:Int, fieldlengths:IMat):Long = {
    icat.toLong + 
    ((ivfeat.toLong + 
        ((ifeat.toLong + 
            ((jfeat.toLong + 
                ((inode.toLong + 
                    (itree.toLong << fieldlengths(INode))
                    ) << fieldlengths(JFeat))
                ) << fieldlengths(IFeat))
          ) << fieldlengths(IVFeat))
      ) << fieldlengths(ICat))
  }
  
  def extractAbove(fieldNum : Int, packedFields : Long, FieldShifts : IMat) : Int = {
    (packedFields >>> FieldShifts(fieldNum)).toInt
  }

  def extractField(fieldNum : Int, packedFields : Long, FieldShifts : IMat, FieldMasks : IMat) : Int = {
    (packedFields >>> FieldShifts(fieldNum)).toInt & FieldMasks(fieldNum) 
  }

  def getFieldShifts(fL : IMat) : IMat = {
    val out = izeros(1, fL.length)
    var i = fL.length - 2
    while (i >= 0) {
      out(i) = out(i+1) + fL(i+1)
      i -= 1
    }
    out
  }

  def getFieldMasks(fL : IMat) : IMat = {
    val out = izeros(1, fL.length)
    var i = 0
    while (i < fL.length) {
      out(i) = (1 << fL(i)) - 1
      i += 1
    }
    out
  }
  
  def treePack(idata:IMat, treenodes:IMat, cats:SMat, out:LMat, nsamps:Int, fieldlengths:IMat):LMat = {
    val nfeats = idata.nrows
    val nitems = idata.ncols
    val ntrees = treenodes.nrows
    val ncats = cats.nrows
    val nnzcats = cats.nnz
    val ionebased = Mat.ioneBased
    var icol = 0
    while (icol < nitems) {
      var jci = cats.jc(icol) - ionebased
      val jcn = cats.jc(icol+1) - ionebased
      var itree = 0
      while (itree < ntrees) {
        val inode = treenodes(itree, icol)
        var jfeat = 0
        while (jfeat < nsamps) {
          val ifeat = rhash(itree, inode, jfeat, nfeats)
          val ivfeat = idata(ifeat, icol)
          var j = jci
          while (j < jcn) {
            out.data(jfeat + nsamps * (itree + ntrees * j)) = packFields(itree, inode, jfeat, ifeat, ivfeat, cats.ir(j) - ionebased, fieldlengths)
            j += 1
          }
          jfeat += 1
        }
        itree += 1
      }
      icol += 1
    }
    new LMat(nsamps * nnzcats * ntrees, 1, out.data);
  }
  
  def treeStep(idata:IMat, tnodes:IMat,  trees:IMat, nifeats:Int, nnodes:Int)  {
    val nfeats = idata.nrows;
    val nitems = idata.ncols;
    val ntrees = tnodes.nrows;
    var icol = 0;
    while (icol < nitems) {
      var itree = 0;
      while (itree < ntrees) {
        val inode = tnodes(itree, icol);
        val ifeat = trees(2 * inode, itree);
        val ithresh = trees(1 + 2 * inode, itree);
        if (ifeat < 0) {
          tnodes(itree, icol) = ithresh;
        } else {
          val ivfeat = idata(ifeat, icol);
          if (ivfeat > ithresh) {
            tnodes(itree, icol) = 2 * inode + 2;
          } else {
            tnodes(itree, icol) = 2 * inode + 1;
          }
        }
        itree += 1;
      }
      icol += 1;
    }
  }
  
  def treeWalk(idata:IMat, tnodes:IMat,  trees:IMat, nifeats:Int, nnodes:Int, depth:Int)  {
    val nfeats = idata.nrows;
    val nitems = idata.ncols;
    val ntrees = tnodes.nrows;
    var icol = 0;
    while (icol < nitems) {
      var itree = 0;
      while (itree < ntrees) {
        var inode = 0;
        var id = 0;
        while (id <= depth) {
          val ifeat = trees(2 * inode, itree);
          val ithresh = trees(1 + 2 * inode, itree);
          if (ifeat < 0) {
            inode = ithresh;
            id = depth;
          } else {
            val ivfeat = idata(ifeat, icol);
            if (ivfeat > ithresh) {
              inode = 2 * inode + 2;
            } else {
              inode = 2 * inode + 1;
            }
          }
          id += 1;
        }
        tnodes(itree, icol) = inode;
        itree += 1;
      }
      icol += 1;
    }
  }
    
  def makeV(ind:LMat, out:LMat, counts:FMat):(LMat, FMat) = {
    val n = ind.length;
    var cc = 0;
    var ngroups = 0;
    var i = 1;
    while (i <= n) {
      cc += 1;
      if (i == n || ind(i) != ind(i-1)) {
        out(ngroups) = ind(i-1);
        counts(ngroups) = cc;
        ngroups += 1;
        cc = 0;
      }
      i += 1;
    }
    (new LMat(ngroups, 1, out.data), new FMat(ngroups, 1, counts.data))
  }
  
  def countV(ind1:LMat, counts1:FMat, ind2:LMat, counts2:FMat):Int = {
    var count = 0
    val n1 = counts1.length
    val n2 = counts2.length
    var i1 = 0
    var i2 = 0
    while (i1 < n1 || i2 < n2) {
      if (i1 >= n1 || ind2(i2) < ind1(i1)) {
        count += 1
        i2 += 1
      } else if (i2 >= n2 || ind1(i1) < ind2(i2)) {
        count += 1
        i1 += 1
      } else {
        count += 1
        i1 += 1
        i2 += 1
      }
    }
    return count
  }
  
  def addV(ind1:LMat, counts1:FMat, ind2:LMat, counts2:FMat):(LMat, FMat) = {
    val size = countV(ind1, counts1, ind2, counts2);
    val ind3 = LMat(size, 1)
    val counts3 = FMat(size, 1)
    var count = 0
    val n1 = counts1.length
    val n2 = counts2.length
    var i1 = 0
    var i2 = 0
    while (i1 < n1 || i2 < n2) {
      if (i1 >= n1 || ind2(i2) < ind1(i1)) {
        ind3(count) = ind2(i2)
        counts3(count) = counts2(i2)
        count += 1
        i2 += 1
      } else if (i2 >= n2 || ind1(i1) < ind2(i2)) {
        ind3(count) = ind1(i1)
        counts3(count) = counts1(i1)
        count += 1
        i1 += 1
      } else {
        ind3(count) = ind1(i1)
        counts3(count) = counts1(i1) + counts2(i2)
        count += 1
        i1 += 1
        i2 += 1
      }
    }
    (ind3, counts3)
  }
}

object RandomForest {
    
  trait Opts extends Model.Opts { 
     var depth = 1;
     var ntrees = 10;
     var nsamps = 10;
     var nvals = 1000;
     var margin = 1.5f;
  }
  
  class Options extends Opts {}
}