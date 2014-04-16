package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.Learner

object RForest {
  import jcuda.runtime._
  import jcuda.runtime.JCuda._
  import jcuda.runtime.cudaError._
  import jcuda.runtime.cudaMemcpyKind._
  import scala.util.hashing.MurmurHash3
  import edu.berkeley.bid.CUMACH
  
  def rhash(v1:Int, v2:Int, v3:Int, nb:Int):Int = {
    MurmurHash3.mix(MurmurHash3.mix(v1, v2), v3) % nb
  }
  
  def packFields(itree:Int, inode:Int, irfeat:Int, ivfeat:Int, icat:Int, fieldlengths:IMat):Long = {
    icat.toLong + 
    ((ivfeat.toLong + 
        ((irfeat.toLong + 
            ((inode.toLong + 
                (itree.toLong << fieldlengths(1))
             ) << fieldlengths(2))
          ) << fieldlengths(3))
      ) << fieldlengths(4))
  }
   
  def treePack(fdata:FMat, fbounds:FMat, treenodes:IMat, cats:SMat, nsamps:Int, fieldlengths:IMat) {
    val nfeats = fdata.nrows
    val nitems = fdata.ncols
    val ntrees = treenodes.nrows
    val ncats = cats.nrows
    val nnzcats = cats.nnz
    val out = new Array[Long](ntrees * nsamps * nnzcats)
    var ioff = Mat.ioneBased
    val nifeats = 2 ^ fieldlengths(3)
    var icol = 0
    while (icol < nitems) {
      var jci = cats.jc(icol) - ioff
      val jcn = cats.jc(icol+1) - ioff
      var itree = 0
      while (itree < ntrees) {
        val inode = treenodes(itree, icol)
        var jfeat = 0
        while (jfeat < nsamps) {
          val ifeat = rhash(itree, inode, jfeat, nfeats)
          val vfeat = fdata(ifeat, icol)
          val ivfeat = math.min(nifeats-1, math.floor((vfeat - fbounds(ifeat,0))/(fbounds(ifeat,1) - fbounds(ifeat,0))*nifeats).toInt)
          var jc = jci
          while (jc < jcn) {
            out(jfeat + nsamps * (itree + ntrees * jc)) = packFields(itree, inode, jfeat, ivfeat, cats.ir(jc), fieldlengths)
            jc += 1
          }
          jfeat += 1
        }
        itree += 1
      }
      icol += 1
    }
  }
  
  def countC(ind:Array[Long]):Int = {
    val n = ind.length
    var count = math.min(1, n)
    var i = 1
    while (i < n) {
      if (ind(i) != ind(i-1)) {
        count += 1
      }
    }
    return count
  }
  
  def makeC(ind:Array[Long], out:Array[Long], counts:Array[Float]) {
    val n = ind.length
    var cc = 0
    var group = 0
    var i = 1
    while (i <= n) {
      cc += 1
      if (i == n || ind(i) != ind(i-1)) {
        out(group) = ind(i-1)
        counts(group) = cc
        group += 1
        cc = 0
      }
      i += 1
    }
  }
  
  def mergeC(ind1:Array[Long], counts1:Array[Float], ind2:Array[Long], counts2:Array[Float]):Int = {
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
  
  def mergeV(ind1:Array[Long], counts1:Array[Float], ind2:Array[Long], counts2:Array[Float], ind3:Array[Long], counts3:Array[Float]):Int = {
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
    return count
  }
  
}