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
import BIDMach.Learner
import BIDMach.datasources.MatDS
import BIDMach.updaters.Batch
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.CUMAT
import scala.util.hashing.MurmurHash3
import java.util.Arrays

class RandomForest(override val opts:RandomForest.RFopts) extends Model(opts) {
  
  val ITree = 0; val INode = 1; val JFeat = 2; val IFeat = 3; val IVFeat = 4; val ICat = 5
  
  var nnodes:Int = 0;
  var tmp1:Mat = null;
  var tmp2:Mat = null;
  var tc1:Mat = null;
  var totalinds:LMat = null;
  var totalcounts:IMat = null;
  var trees:Mat = null;
  var outv:IMat = null;
  var outf:IMat = null;
  var outn:IMat = null;
  var outg:FMat = null;
  var outc:IMat = null;
  var jc:IMat = null;
  var ntrees = 0;
  var nsamps = 0;
  var nfeats = 0;
  var nvals = 0;
  var ncats = 0;
  val fieldlengths = izeros(1,6);
  
  def init() = {
    opts.npasses = opts.depth;                   // Make sure we make the correct number of passes
    nnodes = math.pow(2, opts.depth + 1).toInt;  // Num nodes per tree. Keep as a power of two (dont subtract one), so blocks align better. 
    ntrees = opts.ntrees;
    nsamps = opts.nsamps;
    nvals = opts.nvals;
    trees = if (opts.useGPU && Mat.hasCUDA > 0) gizeros(2 * nnodes, opts.ntrees) else izeros(2 * nnodes, opts.ntrees);
    trees.set(-1);
    modelmats = Array(trees);
    mats = datasource.next;
    nfeats = mats(0).nrows;
    ncats = mats(1).nrows;
    val nc = mats(0).ncols;
    val nnz = mats(1).nnz;
    datasource.reset;
    tmp1 = lzeros(1, (opts.margin * nnz * ntrees * nsamps).toInt);
    tmp2 = lzeros(1, (opts.margin * nnz * ntrees * nsamps).toInt);
    tc1 = izeros(1, (opts.margin * nnz * ntrees * nsamps).toInt);
    outv = IMat(nsamps, ntrees * nnodes);
    outf = IMat(nsamps, ntrees * nnodes);
    outn = IMat(nsamps, ntrees * nnodes);
    outg = FMat(nsamps, ntrees * nnodes);
    outc = IMat(nsamps, ntrees * nnodes);
    jc = IMat(nsamps, ntrees * nnodes);
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
    val nnodes:Mat = if (gmats.length > 2) gmats(2) else null
    val (inds, counts):(LMat, IMat) = (sdata, cats, trees, tmp1, tmp2, tc1) match {
        case (idata:IMat, scats:SMat, itrees:IMat, lt1:LMat, lt2:LMat, tc:IMat) => {
          val xnodes = if (nnodes.asInstanceOf[AnyRef] != null) nnodes.asInstanceOf[IMat] else treeWalk(idata, itrees, ipass);
          print("xnodes="+xnodes.toString)
          val lt = treePack(idata, xnodes, scats, lt1, nsamps, fieldlengths);
          Arrays.sort(lt.data, 0, lt.length);    
          makeV(lt, lt2, tc);
        }
        case _ => {
          throw new RuntimeException("RandomForest doblock types dont match")
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
    // measure impurity here
    row(0)
  } 
  
  override def updatePass(ipass:Int) = {
    println("Update pass %d" format ipass)
//    throw new RuntimeException("unexceptional")
    val gg = minImpurity(totalinds, totalcounts, outv, outf, outn, outg, outc, jc, opts.impurity);
    println("gg %d %d" format (gg.nrows, gg.ncols))
    print("gg data "+gg.toString)
    print("outv data "+outv.toString)
    val (vm, im) = mini2(gg);
    val inds = im + irow(0->im.length) * gg.nrows;
    val inodes = outn(inds);
    print("inodes "+inodes.toString+"\n")
    val reqgain = if (ipass < opts.depth) opts.gain else Float.MaxValue;
    val i1 = find(vm < -reqgain);
    val i2 = find(vm >= -reqgain);
    println("i1 %d i2 %d" format (i1.length, i2.length))
    if (i1.length > 0) {
      trees(inodes(i1)*2) = outf(inds(i1));
      trees(inodes(i1)*2+1) = outv(inds(i1));
    }
    if (i2.length > 0) {
      trees(inodes(i2)*2) = iones(i2.length,1) * -1;
      trees(inodes(i2)*2+1) = outc(inds(i2));
    }
    totalinds = null;
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
  
  def treeWalk(idata:IMat, trees:IMat, depth:Int):IMat = {
    val nfeats = idata.nrows;
    val nitems = idata.ncols;
    val ntrees = trees.ncols;
    val tnodes:IMat = IMat(ntrees, nitems); 
    var icol = 0;
    while (icol < nitems) {
      var itree = 0;
      while (itree < ntrees) {
        var inode = 0;
        var id = 0;
        while (id < depth) {
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
    tnodes
  }
    
  def makeV(ind:LMat, out:LMat, counts:IMat):(LMat, IMat) = {
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
    (new LMat(ngroups, 1, out.data), new IMat(ngroups, 1, counts.data))
  }
  
  def countV(ind1:LMat, counts1:IMat, ind2:LMat, counts2:IMat):Int = {
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
  
  def addV(ind1:LMat, counts1:IMat, ind2:LMat, counts2:IMat):(LMat, IMat) = {
    val size = countV(ind1, counts1, ind2, counts2);
    val ind3 = LMat(size, 1)
    val counts3 = IMat(size, 1)
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
  
  // Find boundaries where (key >> shift) changes
  
  def findBoundaries(keys:LMat, jc:IMat, shift:Int):IMat = { 
    var oldv = -1L;
    var v = -1L;
    var i = 0;
    var n = 0;
    while (i < keys.length) {
      v = keys(i) >>> shift;
      if (oldv != v) {
        jc(n) = i;
        n += 1;
        oldv = v;
      }
      i += 1
    }
    jc(n) = i;
    n += 1;
    new IMat(n, 1, jc.data)
  }
  
  trait imptyType {
    val update: (Int)=>Float;
    val result: (Float, Int)=>Float;
  }
  
  object entImpurity extends imptyType {
    def updatefn(a:Int):Float = { val v = math.max(a,1).toFloat; v * math.log(v).toFloat }
    def resultfn(acc:Float, tot:Int):Float = { val v = math.max(tot,1).toFloat; math.log(v).toFloat - acc / v }
    val update = updatefn _ ;
    val result = resultfn _ ;
  }
  
  object giniImpurity extends imptyType {
    def updatefn(a:Int):Float = { val v = a.toFloat; v * v }
    def resultfn(acc:Float, tot:Int) = { val v = math.max(tot,1).toFloat; 1f - acc / (v * v) }
    val update = updatefn _ ;
    val result = resultfn _ ;
  }
  
  val imptyFunArray = Array[imptyType](entImpurity,giniImpurity)
  
  // Pass in one of the two object above as the last argument (imptyFns) to control the impurity
  // outv should be an nsamps * nnodes array to hold the feature threshold value
  // outf should be an nsamps * nnodes array to hold the feature index
  // outg should be an nsamps * nnodes array holding the impurity gain (use maxi2 to get the best)
  // jc should be a zero-based array that points to the start and end of each group of fixed node, jfeat

  def minImpurity(keys:LMat, cnts:IMat, outv:IMat, outf:IMat, outn:IMat, outg:FMat, outc:IMat, jcc:IMat, fnum:Int):FMat = {
    
    val update = imptyFunArray(fnum).update
    val result = imptyFunArray(fnum).result

    val totcounts = izeros(1,ncats);
    val counts = izeros(1,ncats);
    val fieldshifts = getFieldShifts(fieldlengths);
    val fieldmasks = getFieldMasks(fieldlengths);

    val jc = findBoundaries(keys, jcc, fieldshifts(JFeat));
    
    var j = 0;
    var tot = 0;
    var tott = 0;
    var acc = 0f;
    var acct = 0f;
    var i = 0;
    while (i < jc.length - 1) {
      val jci = jc(i);
      val jcn = jc(i+1);

      totcounts.clear;
      counts.clear;
      tott = 0;
      j = jci;
      var maxcnt = -1;
      var imaxcnt = -1;
      while (j < jcn) {                     // First get the total counts for each
        val key = keys(j)
        val cnt = cnts(j)
        val icat = extractField(ICat, key, fieldshifts, fieldmasks);
        val newcnt = totcounts(icat) + cnt;
        totcounts(icat) = newcnt;
        tott += cnt;
        if (newcnt > maxcnt) {
          maxcnt = newcnt;
          imaxcnt = icat;
        }
        j += 1;
      }
      acct = 0; 
      j = 0;
      while (j < ncats) {                  // Get the impurity for the node
        acct += update(totcounts(j));
        j += 1
      }
//      if (i < 32)  println("scala %d %d %f" format (i, tott, acct))
      val nodeImpty = result(acct, tott);
      
      var lastival = -1
      var minImpty = Float.MaxValue
      var lastImpty = Float.MaxValue
      var partv = -1
      var besti = -1
      acc = 0;
      tot = 0;
      j = jci;
      val inode = (keys(j) >>> fieldshifts(INode)).toInt;
      while (j < jcn) {                   // Then incrementally update top and bottom impurities and find min total 
        val key = keys(j)
        val cnt = cnts(j)
        val ival = extractField(IVFeat, key, fieldshifts, fieldmasks);
        val icat = extractField(ICat, key, fieldshifts, fieldmasks);
        val oldcnt = counts(icat);
        val newcnt = oldcnt + cnt;
        counts(icat) = newcnt;
        val oldcntt = totcounts(icat) - oldcnt;
        val newcntt = totcounts(icat) - newcnt;
        tot += cnt;
        acc += update(newcnt) - update(oldcnt);
        acct += update(newcntt) - update(oldcntt);
        val impty = result(acc, tot) + result(acct, tott - tot)
//        if (i==0) println("scala pos %d impty %f icat %d cnts %d %d cacc %f %d" format (j, impty,  icat, oldcnt, newcnt, acc, tot))
        if (ival != lastival) {
          if (lastImpty < minImpty) { 
            minImpty = lastImpty;
            partv = ival;
            besti = extractField(IFeat, key, fieldshifts, fieldmasks);
          }
        }
        lastival = ival;
        lastImpty = impty;
        j += 1;
      }
      outv(i) = partv;
      outg(i) = nodeImpty - minImpty;
      outf(i) = besti;
      outc(i) = imaxcnt;
      outn(i) = inode;
      i += 1;
    }
    new FMat(nsamps, (jc.length - 1)/nsamps, outg.data);
  }

}

object RandomForest {
    
  trait Opts extends Model.Opts { 
     var depth = 8;
     var ntrees = 32;
     var nsamps = 32;
     var nvals = 1000;
     var gain = 0.1f; 
     var margin = 1.5f;
     var impurity = 0;  // zero for entropy, one for Gini impurity
  }
  
  class Options extends Opts {}
  
  class RFopts extends Learner.Options with RandomForest.Opts with MatDS.Opts with Batch.Opts;
  
  def learner(data:IMat, labels:SMat) = {
    val opts = new RFopts;
    opts.useGPU = false;
    opts.nvals = maxi(maxi(data)).dv.toInt;
    opts.batchSize = math.min(100000000/data.nrows, data.ncols);
    val nn = new Learner(
        new MatDS(Array(data, labels), opts), 
        new RandomForest(opts), 
        null, 
        new Batch(opts),
        opts)
    (nn, opts)
  }
}