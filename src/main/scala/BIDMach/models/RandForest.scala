package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.Learner
import BIDMat.Sorting._
import scala.util.control.Breaks
import jcuda._
import jcuda.runtime._
import jcuda.runtime.JCuda._
import jcuda.runtime.cudaMemcpyKind._
import edu.berkeley.bid.CUMAT

object RandForest {
  import jcuda.runtime._
  import jcuda.runtime.JCuda._
  import jcuda.runtime.cudaError._
  import jcuda.runtime.cudaMemcpyKind._
  import scala.util.hashing.MurmurHash3
  import edu.berkeley.bid.CUMACH

  val NegativeInfinityF = 0xff800000.toFloat
  val NegativeInfinityI = 0xff800000.toInt
  val ITree = 0; val INode = 1; val JFeat = 2; val IFeat = 3; val IVFeat = 4; val ICat = 5
  
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
      out(i) = (fL(i) << 1) - 1
      i += 1
    }
    out
  }

  def sortLongs(a:Array[Long], useGPU : Boolean) {

    if (useGPU) {
      val memorySize = Sizeof.LONG*a.length
      val deviceData : Pointer = new Pointer();
      cudaMalloc(deviceData, memorySize);
      cudaMemcpy(deviceData, Pointer.to(a), memorySize, cudaMemcpyKind.cudaMemcpyHostToDevice);
      val err =  CUMAT.lsort(deviceData, a.length, 1)
      if (err != 0) {throw new RuntimeException("lsort: CUDA kernel error in lsort " + cudaGetErrorString(err))}
      cudaMemcpy(Pointer.to(a), deviceData, memorySize, cudaMemcpyKind.cudaMemcpyDeviceToHost);
      cudaFree(deviceData);
    } else {

      def comp(i1 : Int, i2 : Int) : Int = {
        val a1 = a(i1)
        val a2 = a(i2)
        return compareLong(a1, a2)
      }

      def swap(i1 : Int, i2 : Int) = {
        val tempA = a(i2)
        a(i2) = a(i1)
        a(i1) = tempA
      }

      quickSort(comp, swap, 0, a.length)
    }

  }

  def compareLong(i : Long, j : Long) : Int = {
    if (i < j) {
      return -1
    } else if (i == j) {
      return 0
    } else {
      return 1
    }
  }

  def treePack(fdata:IMat, treenodes:IMat, cats:IMat, out:Array[Long], jc:IMat, nsamps:Int, fieldlengths:IMat) = {
    val nfeats = fdata.nrows
    val nitems = fdata.ncols
    val ntrees = treenodes.nrows
    val ncats = cats.nrows
    val nnzcats = cats.length
    var icol = 0
    while (icol < nitems) {
      var jci = jc(icol)
      val jcn = jc(icol+1)
      var itree = 0
      while (itree < ntrees) {
        val inode = treenodes(itree, icol)
        var jfeat = 0
        while (jfeat < nsamps) {
          val ifeat = rhash(itree, inode, jfeat, nfeats)
          val ivfeat = fdata(ifeat, icol)
          var jc = jci
          while (jc < jcn) {
            out(jfeat + nsamps * (itree + ntrees * jc)) = packFields(itree, inode, jfeat, ifeat, ivfeat, cats(jc), fieldlengths)
            jc += 1
          }
          jfeat += 1
        }
        itree += 1
      }
      icol += 1
    }
    out
  }
  
  trait imptyType {
    val update: (Int)=>Float;
    val result: (Float, Int)=>Float;
  }
  
  object entImpurity extends imptyType {
    def updatefn(a:Int):Float = { val v = math.max(a,1).toFloat; v * math.log(v).toFloat }
    def resultfn(acc:Float, tot:Int) = { val v = math.max(tot,1).toFloat; math.log(v) - acc / v }
    val update = updatefn _ ;
    val result = resultfn _ ;
  }
  
  object giniImpurity extends imptyType {
    def updatefn(a:Int):Float = { val v = a.toFloat; v * v }
    def resultfn(acc:Float, tot:Int) = { val v = math.max(tot,1).toFloat; 1f - acc / (v * v) }
    val update = updatefn _ ;
    val result = resultfn _ ;
  }
  
  // Pass in one of the two object above as the last argument (imptyFns) to control the impurity
  // outv should be an nsamps * nnodes array to hold the feature threshold value
  // outf should be an nsamps * nnodes array to hold the feature index
  // outg should be an nsamps * nnodes array holding the impurity gain (use maxi2 to get the best)
  // jc should be a zero-based array that points to the start and end of each group of fixed node,jfeat

  def minImpurity(keys:Array[Long], cnts:IMat, outv:IMat, outf:IMat, outg:FMat, jc:IMat, fieldlens:IMat, 
      ncats:Int, imptyFns:imptyType) = {

    val totcounts = izeros(1,ncats);
    val counts = izeros(1,ncats);
    val fieldshifts = getFieldShifts(fieldlens);
    val fieldmasks = getFieldMasks(fieldlens);


    var j = 0;
    var tot = 0;
    var tott = 0;
    var acc = 0f;
    var acct = 0f;
    var i = 1;
    while (i < jc.length) {
      val jci = jc(i-1);
      val jcn = jc(i);

      totcounts.clear;
      counts.clear;
      tott = 0;
      j = jci;
      while (j < jcn) {                     // First get the total counts for each
        val key = keys(i)
        val cnt = cnts(i)
        val icat = extractField(ICat, key, fieldshifts, fieldmasks); 
        totcounts(icat) += cnt;
        tott += cnt;
        j += 1;
      }
      acct = 0; 
      j = 0;
      while (j < ncats) {                  // Get the impurity for the node
        acct += imptyFns.update(totcounts(j))
        j += 1
      }
      val nodeImpty = imptyFns.result(acct, tott);
      
      var oldvfeat = -1
      var minImpty = Float.MaxValue
      var partv = -1
      var besti = -1
      acc = 0;
      tot = 0;
      j = jci;
      while (j < jcn) {                   // Then incrementally update top and bottom impurities and find min total 
        val key = keys(i)
        val cnt = cnts(i)
        val vfeat = extractField(IVFeat, key, fieldshifts, fieldmasks);
        val icat = extractField(ICat, key, fieldshifts, fieldmasks);
        val oldcnt = counts(icat);
        val newcnt = oldcnt + cnt;
        counts(icat) = newcnt;
        val oldcntt = totcounts(icat) - oldcnt;
        val newcntt = totcounts(icat) - newcnt;
        tot += cnt;
        acc += imptyFns.update(newcnt) - imptyFns.update(oldcnt);
        acct += imptyFns.update(newcntt) - imptyFns.update(oldcntt);
        if (vfeat != oldvfeat) {
          val impty = imptyFns.result(acc, tot) + imptyFns.result(acct, tott - tot)
          if (impty < minImpty) {
            val ifeat = extractField(IFeat, key, fieldshifts, fieldmasks);
            minImpty = impty;
            partv = vfeat;
            besti = ifeat;
          }
        }
        j += 1;
      }
      outv(i) = partv;
      outg(i) = nodeImpty - minImpty;
      outf(i) = besti;
      i += 1;
    }
  }

  def treeSteps(tn : IMat, fd : FMat, fb : FMat, fL : IMat, tMI : IMat, depth : Int, ncats : Int, isLastIteration : Boolean)  {
    val dnodes = (math.pow(2, depth).toInt - 1)
    val nfeats = fd.nrows
    val nitems = fd.ncols
    val ntrees = tn.nrows
    val nifeats = math.pow(2, fL(3)) - 1
    var icol = 0
    while (icol < nitems) {
      var itree = 0
      while (itree < ntrees) {
        val inode = tn(itree, icol)
        if (isLastIteration) {
          val cat = tMI(2, itree * dnodes + inode)
          tn(itree, icol) = math.min(cat, ncats)
        } else if (tMI(3, itree * dnodes + inode) > 0) {
          // Do nothing
        } else {
          val ifeat : Int = tMI(0, itree * dnodes + inode)
          val vfeat : Float = fd(ifeat, icol)
          val ivfeat = math.min(nifeats, math.floor((vfeat - fb(ifeat,0))/(fb(ifeat,1) - fb(ifeat,0))*nifeats).toInt)
          val ithresh = tMI(1, itree * dnodes + inode)
          if (ivfeat > ithresh) {
            tn(itree, icol) = 2 * tn(itree, icol) + 2
          } else {
            tn(itree, icol) = 2 * tn(itree, icol) + 1
          }
        }
        itree += 1
      }
      icol += 1
    }
  }

  def treeSearch(ntn : IMat, fd : FMat, fb : FMat, fL : IMat, tMI : IMat, depth : Int, ncats : Int) {
    var d = 0
    while (d < depth) {
      treeSteps(ntn, fd, fb, fL, tMI, depth, ncats, d==(depth - 1)) 
      d += 1
    }
  }

  // expects a to be n * t
  // returns out (n * numCats)
  def accumG(a : Mat, dim : Int, numBuckets : Int)  : Mat = {
    (dim, a) match {
      case (1, aMat : IMat) => {
        // col by col
        null
      }
      case (2, aMat : IMat) => {
        val iData = (icol(0->aMat.nrows) * iones(1, aMat.ncols)).data
        val i = irow(iData)
        val j = irow(aMat.data)
        val ij = i on j
        val out = accum(ij.t, 1, null, a.nrows, scala.math.max(a.ncols, numBuckets))
        out
      }
    }
  }

  def voteForBestCategoriesAcrossTrees(treeCatsT : Mat, numCats : Int) : Mat = {
    val accumedTreeCatsT = accumG(treeCatsT, 2, numCats + 1)
    var bundle : (Mat, Mat) = null
    (accumedTreeCatsT) match {
      case (acTCT : IMat) => {
        bundle = maxi2(acTCT, 2)
      }
    }
    val majorityVal = bundle._1
    val majorityIndicies = bundle._2
    majorityIndicies.t
  }
  
  def countC(ind:Array[Long]):Int = {
    val n = ind.length
    var count = math.min(1, n)
    var i = 1
    while (i < n) {
      if (ind(i) != ind(i-1)) {
        count += 1
      }
      i += 1
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