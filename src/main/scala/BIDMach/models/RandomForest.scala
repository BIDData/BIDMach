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
import BIDMach.datasources.{DataSource,MatDS}
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
  var ftrees:IMat = null;
  var vtrees:IMat = null
  var ctrees:IMat = null;
  var gtrees:GIMat = null;
  var outv:IMat = null;
  var outf:IMat = null;
  var outn:IMat = null;
  var outg:FMat = null;
  var outc:IMat = null;
  var outlr:IMat = null;
  var jc:IMat = null;
  var xnodes:IMat = null;
  var ynodes:IMat = null;
  var ntrees = 0;
  var nsamps = 0;
  var nfeats = 0;
  var nvals = 0;
  var ncats = 0;
  val fieldlengths = izeros(1,6);
  var fieldmasks:IMat = null;
  var t0 = 0f;
  var t1 = 0f;
  var t2 = 0f;
  var t3 = 0f; 
  var t4 = 0f;
  var t5 = 0f;
  val runtimes = zeros(7,1);
  
  def init() = {
    opts.npasses = opts.depth;                   // Make sure we make the correct number of passes
    nnodes = math.pow(2, opts.depth + 1).toInt;  // Num nodes per tree. Keep as a power of two (dont subtract one), so blocks align better. 
    ntrees = opts.ntrees;
    nsamps = opts.nsamps;
    nvals = opts.nvals;
    ftrees = izeros(nnodes, opts.ntrees);
    vtrees = izeros(nnodes, opts.ntrees);
    ctrees = izeros(nnodes, opts.ntrees);
    ctrees.set(-1);
    ctrees(0,?) = 0;
    ftrees.set(-1)
    if (opts.useGPU && Mat.hasCUDA > 0) gtrees = GIMat(ftrees);
    modelmats = Array(ftrees, vtrees, ctrees);
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
    outlr = IMat(nsamps, ntrees * nnodes);
    jc = IMat(nsamps, ntrees * nnodes);
    fieldlengths(ITree) = countbits(ntrees);
    fieldlengths(INode) = countbits(nnodes);
    fieldlengths(JFeat) = countbits(nsamps);
    fieldlengths(IFeat) = countbits(nfeats);
    fieldlengths(IVFeat) = countbits(nvals);
    fieldlengths(ICat) = countbits(ncats);
    fieldmasks = getFieldMasks(fieldlengths);
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
    val t0 = toc
    val (inds, counts):(LMat, IMat) = (sdata, cats, tmp1, tmp2, tc1) match {
        case (idata:IMat, scats:SMat, lt1:LMat, lt2:LMat, tc:IMat) => {
          xnodes = if (nnodes.asInstanceOf[AnyRef] != null) {
            val nn = nnodes.asInstanceOf[IMat];
            treeStep(idata, nn, ftrees, vtrees, ctrees);
            Mat.nflops += idata.ncols;
            nn;
          } else {
            Mat.nflops += idata.ncols * ipass;
            treeWalk(idata, ftrees, vtrees, ctrees, ipass, false);
          }
 //         print("xnodes="+xnodes.toString)
          t1 = toc;
          runtimes(0) += t1 - t0;
          val lt = treePack(idata, xnodes, scats, lt1, nsamps, fieldlengths);
          Mat.nflops += scats.nnz * nsamps * ntrees * 6;
          t2 = toc;
          runtimes(1) += t2 - t1;
          Arrays.sort(lt.data, 0, lt.length);  
          Mat.nflops += lt.length * math.log(lt.length).toLong
          t3 = toc;
          runtimes(2) += t3 - t2;
          makeV(lt, lt2, tc);
        }
        case _ => {
          throw new RuntimeException("RandomForest doblock types dont match")
        }
    }
    t4 = toc;
    runtimes(3) += t4 - t3;
    if (totalinds.asInstanceOf[AnyRef] != null) {
      val (inds1, counts1) = addV(inds, counts, totalinds, totalcounts);
      totalinds = inds1;
      totalcounts = counts1;
    } else {
      totalinds = inds;
      totalcounts = counts;
    }        
    t5 = toc;
    runtimes(4) += t5 - t4;
  }
  
  def evalblock(mats:Array[Mat], ipass:Int, here:Long):FMat = {
    val sdata = gmats(0);
    val cats = gmats(1);
    val nnodes:Mat = if (gmats.length > 2) gmats(2) else null;
    val nodes:IMat = (sdata, cats, tmp1, tmp2, tc1) match {
    case (idata:IMat, scats:SMat, lt1:LMat, lt2:LMat, tc:IMat) => {
      ynodes = if (nnodes.asInstanceOf[AnyRef] != null) {
        val nn = nnodes.asInstanceOf[IMat];
        treeStep(idata, nn, ftrees, vtrees, ctrees);
        nn;
      } else {
        treeWalk(idata, ftrees, vtrees, ctrees, ipass, true);
      }
      ynodes
    }
    }
    val cinds = icol(0->nodes.ncols) ⊗  iones(ntrees,1);
    val (ii, jj, vv) = find3(SMat(cats));
//    val matches = FMat((-nodes) != ii.t);
//    -mean(mean(matches))
    
    val vc = accum((-nodes(?) - 1)\ cinds, 1, ncats, nodes.ncols);
    val (uu, mm) = maxi2(vc);
    mean(FMat(mm != ii.t));
  } 
  
  override def updatePass(ipass:Int) = {
    t0 = toc;
    val gg = minImpurity(totalinds, totalcounts, outv, outf, outn, outg, outc, outlr, jc, opts.impurity);
    t1 = toc;
    runtimes(5) += t1 - t0;
    println("gg %d %d" format (gg.nrows, gg.ncols))
    val (vm, im) = maxi2(gg);
    val inds = im + irow(0->im.length) * gg.nrows;
    val inodes = outn(inds);
    val reqgain = if (ipass < opts.depth - 1) opts.gain else Float.MaxValue;
    val i1 = find(vm > reqgain);
    println("reqgain "+mean(vm).dv.toString+"\n");
    ctrees(inodes) = outc(inds);                             // Save the node class for this node
    println("i1 %d" format i1.length)
    tochildren(inodes, outlr(inds)); // Save class id to children in class we don't visit them later
    if (i1.length > 0) {
      ftrees(inodes(i1)) = outf(inds(i1));
      vtrees(inodes(i1)) = outv(inds(i1));
    }
    totalinds = null;
    t2 = toc;
    runtimes(6) += t2 - t1;
  }
  
  def tochildren(inodes:IMat, icats:IMat) {
    var i = 0;
    val nmask = fieldmasks(INode);
    while (i < inodes.length) {
      val inode = inodes(i) & nmask;
      val itree = (inodes(i) >> fieldlengths(INode));
      ctrees(2*inode + 1, itree) = icats(i) & 0xffff;
      ctrees(2*inode + 2, itree) = (icats(i) >> 16) & 0xffff;
      i += 1;
    }

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
    val nfeats = idata.nrows;
    val nitems = idata.ncols;
    val ntrees = treenodes.nrows;
    val ncats = cats.nrows;
    val nnzcats = cats.nnz;
    val ionebased = Mat.ioneBased;
    var icol = 0;
    var nvals = 0;
    while (icol < nitems) {
      var jci = cats.jc(icol) - ionebased;
      val jcn = cats.jc(icol+1) - ionebased;
      var itree = 0;
      while (itree < ntrees) {
        val inode = treenodes(itree, icol);
        if (inode >= 0) {
          var jfeat = 0;
          while (jfeat < nsamps) {
            val ifeat = rhash(itree, inode, jfeat, nfeats);
            val ivfeat = idata(ifeat, icol);
            var j = jci;
            while (j < jcn) {
              out.data(nvals) = packFields(itree, inode, jfeat, ifeat, ivfeat, cats.ir(j) - ionebased, fieldlengths);
              j += 1;
              nvals += 1;
            }
            jfeat += 1;
          }
        }
        itree += 1;
      }
      icol += 1;
    }
    new LMat(nvals, 1, out.data);
  }
  
  def treeStep(idata:IMat, tnodes:IMat, ftrees:IMat, vtrees:IMat, ctrees:IMat)  {
    val nfeats = idata.nrows;
    val nitems = idata.ncols;
    val ntrees = tnodes.nrows;
    var icol = 0;
    while (icol < nitems) {
      var itree = 0;
      while (itree < ntrees) {
        val inode = tnodes(itree, icol);
        if (inode >= 0) {                                 // Not already classified. Otherwise inode should be a negative class id
          val ifeat = ftrees(inode, itree);
          val ithresh = vtrees(inode, itree);
          if (ifeat < 0) {                                // At a leaf in the tree so save class id
            tnodes(itree, icol) = -ctrees(inode, itree) -1;
          } else {                                        // Walk down one level in the tree
            val ivfeat = idata(ifeat, icol);
            if (ivfeat > ithresh) {
              tnodes(itree, icol) = 2 * inode + 2;
            } else {
              tnodes(itree, icol) = 2 * inode + 1;
            }
          }
        }
        itree += 1;
      }
      icol += 1;
    }
  }
  
  def treeWalk(idata:IMat, ftrees:IMat, vtrees:IMat, ctrees:IMat, depth:Int,  getcat:Boolean):IMat = {
    val nfeats = idata.nrows;
    val nitems = idata.ncols;
    val ntrees = ftrees.ncols;
    val tnodes:IMat = IMat(ntrees, nitems); 
    var icol = 0;
    while (icol < nitems) {
      var itree = 0;
      while (itree < ntrees) {
        var inode = 0;
        var id = 0;
        while (id < depth) {
          val ifeat = ftrees(inode, itree);
          val ithresh = vtrees(inode, itree);
          if (ifeat < 0) {
            inode = -ctrees(inode, itree) -1;                    
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
        if (getcat && inode >= 0) {
          inode = -ctrees(inode, itree) -1;
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
      if (i1 >= n1 || (i2 < n2 && ind2(i2) < ind1(i1))) {
        count += 1
        i2 += 1
      } else if (i2 >= n2 || (i1 < n1 && ind1(i1) < ind2(i2))) {
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
      if (i1 >= n1 || (i2 < n2 && ind2(i2) < ind1(i1))) {
        ind3(count) = ind2(i2)
        counts3(count) = counts2(i2)
        count += 1
        i2 += 1
      } else if (i2 >= n2 || (i1 < n1 && ind1(i1) < ind2(i2))) {
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
    val update: (Int)=>Double;
    val result: (Double, Int)=>Double;
    val combine: (Double, Double, Int, Int) => Double;
  }
  
  object entImpurity extends imptyType {
    def updatefn(a:Int):Double = { val v = math.max(a,1); v * math.log(v) }
    def resultfn(acc:Double, tot:Int):Double = { val v = math.max(tot,1); math.log(v) - acc / v }
    def combinefn(ent1:Double, ent2:Double, tot1:Int, tot2:Int):Double = { (ent1 * tot1 + ent2 * tot2)/math.max(1, tot1 + tot2) } 
    val update = updatefn _ ;
    val result = resultfn _ ;
    val combine = combinefn _ ;
  }
  
  object giniImpurity extends imptyType {
    def updatefn(a:Int):Double = { val v = a; v * v }
    def resultfn(acc:Double, tot:Int) = { val v = math.max(tot,1); 1f - acc / (v * v) }
    def combinefn(ent1:Double, ent2:Double, tot1:Int, tot2:Int):Double = { (ent1 * tot1 + ent2 * tot2)/math.max(1, tot1 + tot2) }
    val update = updatefn _ ;
    val result = resultfn _ ;
    val combine = combinefn _ ;
  }
  
  val imptyFunArray = Array[imptyType](entImpurity,giniImpurity)
  
  // Pass in one of the two object above as the last argument (imptyFns) to control the impurity
  // outv should be an nsamps * nnodes array to hold the feature threshold value
  // outf should be an nsamps * nnodes array to hold the feature index
  // outg should be an nsamps * nnodes array holding the impurity gain (use maxi2 to get the best)
  // jc should be a zero-based array that points to the start and end of each group of fixed node, jfeat

  def minImpurity(keys:LMat, cnts:IMat, outv:IMat, outf:IMat, outn:IMat, outg:FMat, outc:IMat, outlr:IMat, jcc:IMat, fnum:Int):FMat = {
    
    val update = imptyFunArray(fnum).update
    val result = imptyFunArray(fnum).result
    val combine = imptyFunArray(fnum).combine

    val totcounts = izeros(1,ncats);
    val counts = izeros(1,ncats);
    val fieldshifts = getFieldShifts(fieldlengths);
    val fieldmasks = getFieldMasks(fieldlengths);

    val jc = findBoundaries(keys, jcc, fieldshifts(JFeat));
    
    var j = 0;
    var tot = 0;
    var tott = 0;
    var acc = 0.0;
    var acct = 0.0;
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
      while (j < jcn) {                     // First get the total counts for each group, and the most frequent cat
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
//      println("totcounts "+totcounts.toString);
      j = 0;
      while (j < ncats) {                  // Get the impurity for the node
        acct += update(totcounts(j));
        j += 1
      }
      val nodeImpty = result(acct, tott);
      
      var lastival = -1
      var minImpty = nodeImpty
      var lastImpty = Double.MaxValue
      var partv = -1
      var lastkey = -1L
      acc = 0;
      tot = 0;
      j = jci;
      maxcnt = -1;
      var jmaxcnt = -1;
      var upcat = -1;
      var jmax = j;
      val inode = (keys(j) >>> fieldshifts(INode)).toInt;
      val ifeat = extractField(IFeat, keys(j), fieldshifts, fieldmasks);
      while (j < jcn) {     
        val key = keys(j)
        val cnt = cnts(j)
        val ival = extractField(IVFeat, key, fieldshifts, fieldmasks);
        val icat = extractField(ICat, key, fieldshifts, fieldmasks);  
        
        if (j > jci && ival != lastival) {
          lastImpty = combine(result(acc, tot), result(acct, tott - tot), tot, tott - tot);   // Dont compute every time!
          if (lastImpty < minImpty) { 
            minImpty = lastImpty;
            partv = lastival;
            upcat = jmaxcnt;
            jmax = j;
          }
        }       
        val oldcnt = counts(icat);
        val newcnt = oldcnt + cnt;
        counts(icat) = newcnt;
        if (newcnt > maxcnt) {
          maxcnt = newcnt;
          jmaxcnt = icat;
        }
        val oldcntt = totcounts(icat) - oldcnt;
        val newcntt = totcounts(icat) - newcnt;
        tot += cnt;
        acc += update(newcnt) - update(oldcnt);
        acct += update(newcntt) - update(oldcntt);
        lastkey = key;
        lastival = ival;
        j += 1;
      }
      counts.clear;
      maxcnt = -1
      var kmaxcnt = -1;
      while (j > jmax) {
        j -= 1;
        val key = keys(j)
        val cnt = cnts(j)
        val ival = extractField(IVFeat, key, fieldshifts, fieldmasks);
        val icat = extractField(ICat, key, fieldshifts, fieldmasks);
        val oldcnt = counts(icat);
        val newcnt = oldcnt + cnt;
        counts(icat) = newcnt;
        if (newcnt > maxcnt) {
          maxcnt = newcnt;
          kmaxcnt = icat;
        }        
      } 
      lastImpty = combine(result(acc, tot), result(acct, tott - tot), tot, tott - tot);   // For checking
//      println("Impurity %f, %f, min %f, %d, %d" format (nodeImpty, lastImpty, minImpty, partv, ifeat))
      outv(i) = partv;
      outg(i) = (nodeImpty - minImpty).toFloat;
      outf(i) = ifeat;
      outc(i) = imaxcnt;
      outlr(i) = jmaxcnt + (kmaxcnt << 16);
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
     var nvals = 2;
     var gain = 0.01f;
     var margin = 1.5f;
     var impurity = 0;  // zero for entropy, one for Gini impurity
  }
  
  class Options extends Opts {}
  
  class RFopts extends Learner.Options with RandomForest.Opts with DataSource.Opts with Batch.Opts;
  
  class RFSopts extends RFopts with MatDS.Opts;
  
  def learner(data:IMat, labels:SMat) = {
    val opts = new RFSopts;
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
  
  def learner(ds:DataSource) = {
    val opts = new RFopts;
    opts.useGPU = false;
    opts.nvals = 256;
    val nn = new Learner(
        ds, 
        new RandomForest(opts), 
        null, 
        new Batch(opts),
        opts)
    (nn, opts)
  }
  
  def entropy(a:DMat):Double = {
    val sa = sum(a).dv;
    (a ddot ln(max(drow(1.0), a))) / sa - math.log(sa)
  }
  
  def entropy(a:DMat, b:DMat):Double = {
    val ea = entropy(a);
    val eb = entropy(b);
    val sa = sum(a).dv;
    val sb = sum(b).dv;
    if (sa > 0 && sb > 0) {
      (sa * ea + sb * eb)/(sa + sb)
    } else if (sa > 0) {
      ea
    } else {
      eb
    }
  }
  
  def entropy(a:IMat):Double = entropy(DMat(a));
  
  def entropy(a:IMat, b:IMat):Double = entropy(DMat(a), DMat(b));
  
  def checktree(tree:IMat, ncats:Int) {
    val ntrees = tree.ncols;
    val nnodes = tree.nrows >> 1;
    def checknode(inode:Int, itree:Int) {
      if (tree(inode * 2, itree) < 0) {
        if (tree(inode * 2 + 1, itree) <  0 ||  tree(inode * 2 + 1, itree) > ncats) {
          throw new RuntimeException("Bad node %d in tree %d" format (inode, itree));
        }
      } else {
        checknode(inode*2+1, itree);
        checknode(inode*2+2, itree);
      }
    }
    var i = 0
    while (i < ntrees) {
      checknode(0, i);
      i += 1;
    }
    println("OK");
  }
}