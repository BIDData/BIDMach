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

object RForest {
  import jcuda.runtime._
  import jcuda.runtime.JCuda._
  import jcuda.runtime.cudaError._
  import jcuda.runtime.cudaMemcpyKind._
  import scala.util.hashing.MurmurHash3
  import edu.berkeley.bid.CUMACH

  val NegativeInfinityF = 0xff800000.toFloat
  val NegativeInfinityI = 0xff800000.toInt
  val ITree = 0; val INode = 1; val IRFeat = 2; val IVFeat = 3; val ICat = 4
  
  def rhash(v1:Int, v2:Int, v3:Int, nb:Int):Int = {
    math.abs(MurmurHash3.mix(MurmurHash3.mix(v1, v2), v3) % nb)
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
  
  def extractAbove(fieldNum : Int, packedFields : Long, fieldlengths : IMat, FieldMaskRShifts : IMat) : Int = {
    (packedFields >>> FieldMaskRShifts(fieldNum)).toInt
  }

  def extractField(fieldNum : Int, packedFields : Long, fieldlengths : IMat, FieldMaskRShifts : IMat, FieldMasks : IMat) : Int = {
    (packedFields >>> FieldMaskRShifts(fieldNum)).toInt & FieldMasks(fieldNum) 
  }

  def getFieldMaskRShifts(fL : IMat) : Mat = {
    val out = fL.izeros(1, fL.length)
    var i : Int = 0
    while (i < fL.length) {
      out(i) = getFieldMaskRShiftFor(i, fL)
      i += 1
    }
    out
  }

  def getFieldMaskRShiftFor(fieldNum : Int, fL : IMat) : Int = {
    var bitShift : Int = 0
    var i : Int = fieldNum + 1
    while (i < fL.length) {
      bitShift += fL(i)
      i += 1
    }
    bitShift
  }

  def getFieldMasks(fL : IMat) : Mat = {
    val out = fL.izeros(1, fL.length)
    var i = 0
    while (i < fL.length) {
      out(i) = getFieldMaskFor(i, fL)
      i += 1
    }
    out
  }

  def getFieldMaskFor(fieldNum : Int, fL : IMat) : Int = {
    (math.pow(2, fL(fieldNum))).toInt - 1
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

  def treePack(fdata:FMat, fbounds:FMat, treenodes:IMat, cats:SMat, nsamps:Int, fieldlengths:IMat) : Array[Long] = {
    val nfeats = fdata.nrows
    val nitems = fdata.ncols
    val ntrees = treenodes.nrows
    val ncats = cats.nrows
    val nnzcats = cats.nnz
    val out = new Array[Long](ntrees * nsamps * nnzcats)
    var ioff = Mat.ioneBased
    val nifeats =  math.pow(2, fieldlengths(3)).toInt - 1
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
          val ivfeat = math.min(nifeats, math.floor((vfeat - fbounds(ifeat,0))/(fbounds(ifeat,1) - fbounds(ifeat,0))*nifeats).toInt)
          var jc = jci
          while (jc < jcn) {
            out(jfeat + nsamps * (itree + ntrees * jc)) = packFields(itree, inode, ifeat, ivfeat, cats.ir(jc) - ioff, fieldlengths)
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

  def updateTreeData(packedVals : Array[Long], fL : IMat, ncats : Int, tMI : IMat, d : Int, isLastIteration : Boolean, 
      FieldMaskRShifts : IMat, FieldMasks : IMat) = {
    val n = packedVals.length
    val dnodes = (math.pow(2, d).toInt - 1)
    var bgain = NegativeInfinityF; var bthreshold = NegativeInfinityI; var brfeat = -1; 
    val catCounts = fL.zeros(1, ncats);
    val catCountsSoFar = fL.zeros(1, ncats);

    var i = 0
    while (i < n) {
        val itree = extractField(ITree, packedVals(i), fL, FieldMaskRShifts, FieldMasks)
        val inode = extractField(INode, packedVals(i), fL, FieldMaskRShifts, FieldMasks)
        val ivfeat = extractField(IVFeat, packedVals(i), fL, FieldMaskRShifts, FieldMasks)
        val uirfeat = extractAbove(IRFeat, packedVals(i), fL, FieldMaskRShifts) // u for unique
        val uivfeat = extractAbove(IVFeat, packedVals(i), fL, FieldMaskRShifts)
        val uinode = extractAbove(INode, packedVals(i), fL, FieldMaskRShifts)

        var j = i
        catCounts.clear
        val mybreaks = new Breaks
        import mybreaks.{break, breakable}
        breakable {
          while (j < n) {
            val jcat = extractField(ICat, packedVals(j), fL, FieldMaskRShifts, FieldMasks)
            val ujrfeat = extractAbove(IRFeat, packedVals(j), fL, FieldMaskRShifts)
            if (ujrfeat != uirfeat) {
              break()
            }
            catCounts(jcat) += 1f
            j+=1
          }
        }
        val (bCatCount, bCat) = maxi2(catCounts, 2)
        val inext = j // beginning of next feat
        j = i
        catCountsSoFar.clear
        var ujvlastFeat = uivfeat
        while (j < inext && (inext - i)> 10) {
          val ujvfeat = extractAbove(IVFeat, packedVals(j), fL, FieldMaskRShifts)
          val jvfeat = extractField(IVFeat, packedVals(j), fL, FieldMaskRShifts, FieldMasks)
          val jrfeat = extractField(IRFeat, packedVals(j), fL, FieldMaskRShifts, FieldMasks)
          val jcat = extractField(ICat, packedVals(j), fL, FieldMaskRShifts, FieldMasks)
          if (ujvlastFeat != ujvfeat || (j == (inext - 1)))  {
            val gain = getGain(catCountsSoFar, catCounts)
            if (gain > bgain && gain > 0f) {
              bgain = gain
              bthreshold = jvfeat
              brfeat = jrfeat
            }
            ujvlastFeat = ujvfeat
          }
          catCountsSoFar(jcat) += 1f

          j+=1
        }

        i = inext

        if (i == n || extractAbove(INode, packedVals(i), fL, FieldMaskRShifts) != uinode) {
          tMI(0, (itree  * dnodes) + inode) = brfeat
          tMI(1, (itree  * dnodes) + inode) = bthreshold
          tMI(2, (itree  * dnodes) + inode) = bCat(0)
          if (isLastIteration || bgain <= 0f) {
            tMI(3, (itree  * dnodes) + inode) = 1
          }
          bgain = NegativeInfinityF; bthreshold = NegativeInfinityI; brfeat = -1;
        }
    }

  }

  def getGain(catCountsSoFar : FMat, catCounts : FMat) : Float = {
    val topCountsSoFar = catCounts - catCountsSoFar
    val topTots = sum(topCountsSoFar, 2)(0)
    val botTots = sum(catCountsSoFar, 2)(0)
    var totTots = sum(catCounts, 2)(0)
    var topFract = 0f
    var botFract = 0f 
    if (totTots > 0f) {
      botFract = botTots/ totTots
      topFract = 1f - botFract
    }
    val totalImpurity = getImpurity(catCounts, totTots)
    val topImpurity = getImpurity(topCountsSoFar, topTots)
    val botImpurity = getImpurity(catCountsSoFar, botTots)
    val infoGain = totalImpurity - (topFract * topImpurity) - (botFract * botImpurity)
    infoGain
  }

  def getImpurity(countsSoFar : FMat, totCounts : Float) : Float = {
    var impurity = 0f
    var i = 0
    while (i < countsSoFar.length)  {
      var p = 0f 
      if (totCounts > 0f) {
        p = countsSoFar(i)/ totCounts
      }
      var plog : Float = 0f
      if (p != 0) {
        plog = scala.math.log(p).toFloat
      }
      impurity = impurity + (-1f * p * plog)
      i += 1
    }
    impurity
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