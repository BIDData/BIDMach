package BIDMach.models

// Random Forest model (new version).
// Computes a matrix representing the Forest.


import BIDMat.{SBMat,CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GLMat,GSMat,HMat,IMat,LMat,Mat,SMat,SDMat}
import BIDMach.Learner
import BIDMach.datasources.{DataSource,MatDS}
import BIDMach.updaters.Batch
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.CUMAT
import edu.berkeley.bid.CUMACH
import edu.berkeley.bid.CUMACH
import jcuda._
import jcuda.runtime.JCuda._
import jcuda.runtime.cudaMemcpyKind._
import scala.util.hashing.MurmurHash3
import java.util.Arrays

class RandomForest(override val opts:RandomForest.Opts = new RandomForest.Options) extends Model(opts) {
  
  val ITree = 0; val INode = 1; val JFeat = 2; val IFeat = 3; val IVFeat = 4; val ICat = 5
  
  var nnodes = 0;
  var ntrees = 0;
  var nsamps = 0;
  var nfeats = 0;
  var nbits = 0;
  var ncats = 0;
  var iblock = 0;
  var seed = 0;
  var batchSize = 0;
  var blockv:SVec = null;
  var gtmpinds:GLMat = null;
  var gpiones:GIMat = null;
  var gtmpcounts:GIMat = null;
  var totals:SVTree = null;
  var nodecounts:IMat = null;
  var itrees:IMat = null;                   // Index of left child (right child is at this value + 1)
  var ftrees:IMat = null;                   // The feature index for this node
  var vtrees:IMat = null;                   // The value to compare with for this node
  var ctrees:FMat = null;                   // Majority class for this node
  var gitrees:GIMat = null;                   // Index of left child (right child is at this value + 1)
  var gftrees:GIMat = null;                   // The feature index for this node
  var gvtrees:GIMat = null;                   // The value to compare with for this node
  var gctrees:GMat = null; 
  var gftree:GIMat = null;
  var gitree:GIMat = null;
  var lout:LMat = null;
  var gout:GLMat = null;
  var gtnodes:GIMat = null;
  var gfnodes:GMat = null;
  var outv:IMat = null;                     // Threshold values returned by minImpurity
  var outf:IMat = null;                     // Features returned by minImpurity
  var outn:IMat = null;                     // Node numbers returned by minImpurity
  var outg:FMat = null;                     // Node impurity gain returned by minImpurity
  var outc:FMat = null;                     // Category label (or avg) returned by minImpurity
  var outleft:FMat = null;                  // child categories returned by minImpurity     
  var outright:FMat = null;
  var jc:IMat = null;
  var xnodes:IMat = null;
  var ynodes:FMat = null;
  var gains:FMat = null;
  var igains:FMat = null; 
  val fieldlengths = izeros(1,6);
  var gfieldlengths:GIMat = null;
  var fieldmasks:IMat = null;
  var fieldshifts:IMat = null;
  var t0 = 0f;
  var t1 = 0f;
  var t2 = 0f;
  var t3 = 0f; 
  var t4 = 0f;
  var t5 = 0f;
  var t6 = 0f;
  val runtimes = zeros(8,1);
  var x:Mat = null;
  var y:Mat = null;
  var onGPU = false;
  var useIfeats = false;
  
  def init() = {
    mats = datasource.next;
    nfeats = mats(0).nrows;
    val nc = mats(0).ncols;
    batchSize = nc;
    val nnz = mats(1).nnz;
    datasource.reset;   
    nnodes = opts.nnodes; 
    ntrees = opts.ntrees;
    nsamps = opts.nsamps;
    nbits = opts.nbits;
    seed = opts.seed;
    useIfeats = opts.useIfeats;
    ncats = if (opts.ncats > 0) opts.ncats else (maxi(mats(1)).dv.toInt + 1);
    fieldlengths(ITree) = RandomForest.countbits(ntrees);
    fieldlengths(INode) = RandomForest.countbits(nnodes);
    fieldlengths(JFeat) = RandomForest.countbits(nsamps);
    fieldlengths(IFeat) = if (useIfeats) RandomForest.countbits(nfeats) else 0;
    fieldlengths(IVFeat) = nbits;
    fieldlengths(ICat) = RandomForest.countbits(ncats);
    fieldmasks = getFieldMasks(fieldlengths);
    fieldshifts = getFieldShifts(fieldlengths);
    if (opts.GPU && Mat.hasCUDA > 0) onGPU = true;
    if (refresh) {
    	if (sum(fieldlengths).v > 64) {
    		throw new RuntimeException("RandomForest: Too many bits in treepack! "+ sum(fieldlengths).v);
    	}
    	opts.asInstanceOf[Learner.Options].npasses = opts.depth;                   // Make sure we make the correct number of passes
    	itrees = izeros(nnodes, ntrees);
    	ftrees = izeros(nnodes, ntrees);
    	vtrees = izeros(nnodes, ntrees);
    	ctrees = zeros(nnodes, ntrees);
    	gains = zeros(ntrees,1);
    	igains = zeros(ntrees,1);
    	nodecounts = iones(opts.ntrees, 1);
    	ctrees.set(-1);
    	ctrees(0,?) = 0;
    	ftrees.set(-1)
    	modelmats = Array(itrees, ftrees, vtrees, ctrees);
    	// Small buffers hold results of batch treepack and sort
    	val bsize = (opts.catsPerSample * batchSize * ntrees * nsamps).toInt;
      totals = new SVTree(20);
    	outv = IMat(nsamps, nnodes);
    	outf = IMat(nsamps, nnodes);
    	outn = IMat(nsamps, nnodes);
    	outg = FMat(nsamps, nnodes);
    	outc = FMat(nsamps, nnodes);
    	outleft = FMat(nsamps, nnodes);
    	outright = FMat(nsamps, nnodes);
    	jc = IMat(1, ntrees * nnodes * nsamps);
    	lout = LMat(1, batchSize * nsamps * ntrees);
    	
    	if (onGPU) {
    		gfieldlengths = GIMat(fieldlengths);
    		gpiones = giones(1, bsize);
    		gtmpinds = glzeros(1, bsize);
    		gtmpcounts = gizeros(1, bsize);
    		gout = GLMat(1, batchSize * nsamps * ntrees);
    		gtnodes = GIMat(ntrees, batchSize);
    		gfnodes = GMat(ntrees, batchSize);
        gftree = GIMat(nnodes, 1);
        gitree = GIMat(nnodes, 1);
        gitrees = GIMat(itrees);
        gftrees = GIMat(ftrees);
        gvtrees = GIMat(vtrees);
        gctrees = GMat(ctrees);
    	}
    }       
  } 
  
  def doblock(gmats:Array[Mat], ipass:Int, i:Long) = {
    val data = gmats(0);
    val cats = gmats(1);
    val nnodes:Mat = if (gmats.length > 2) gmats(2) else null
    val t0 = toc
    (data, cats) match {
    case (fdata:FMat, icats:IMat) => {
    	xnodes = if (nnodes.asInstanceOf[AnyRef] != null) {
    	  val nn = nnodes.asInstanceOf[IMat];
    	  treeStep(fdata, nn, null, itrees, ftrees, vtrees, ctrees, false);
    	  nn;
    	} else {
    	  val nn = izeros(ntrees, fdata.ncols);
    	  treeWalk(fdata, nn, null, itrees, ftrees, vtrees, ctrees, ipass, false);
    	  nn;
    	}
    	//         print("xnodes="+xnodes.toString)
    	t1 = toc;
    	runtimes(0) += t1 - t0;
        if (onGPU) {
    	  gout = gtreePack(fdata, xnodes, icats, gout, seed);
    	  t2 = toc; runtimes(1) += t2 - t1;
    	  gpsort(gout);  
    	  t3 = toc;  runtimes(2) += t3 - t2;
    	  blockv = gmakeV(gout, gpiones, gtmpinds, gtmpcounts);
    	} else {
    	  lout = treePack(fdata, xnodes, icats, lout, seed);
    	  t2 = toc; runtimes(1) += t2 - t1;
    	  java.util.Arrays.sort(lout.data, 0, lout.length);
    	  Mat.nflops += lout.length * math.log(lout.length).toLong;
    	  t3 = toc; runtimes(2) += t3 - t2;
    	  blockv = makeV(lout);
    	}
    }
    case (gdata:GMat, gicats:GIMat) => {
    	gtreeWalk(gdata, gtnodes, gfnodes, gitrees, gftrees, gvtrees, gctrees, ipass, false); 
    	t1 = toc; runtimes(0) += t1 - t0;
    	gout = gtreePack(gdata, gtnodes, gicats, gout, seed);
    	t2 = toc; runtimes(1) += t2 - t1;
    	gpsort(gout);  
    	t3 = toc;	runtimes(2) += t3 - t2;
    	blockv = gmakeV(gout, gpiones, gtmpinds, gtmpcounts);
    }
    case _ => {
    	throw new RuntimeException("RandomForest doblock types dont match %s %s" format (data.mytype, cats.mytype))
    }
    }
    t4 = toc;
    runtimes(3) += t4 - t3;
    if (opts.trace > 1) println("collect/add %d %d" format (gout.length, blockv.length))
    totals.addSVec(blockv);
    iblock += 1;
    t5 = toc;
    runtimes(4) += t5 - t4;
  }
  
  def evalblock(mats:Array[Mat], ipass:Int, here:Long):FMat = {
    val ipass0 = if (opts.training) ipass else opts.depth
    val data = gmats(0);
    val cats = gmats(1);
    val nnodes:Mat = if (gmats.length > 2) gmats(2) else null;
    val fnodes:FMat = zeros(ntrees, data.ncols);
    (data, cats) match {
      case (fdata:FMat, icats:IMat) => {
//      println("h=%d, hash=%f" format (here, sum(sum(fdata,2)).v))
        if (nnodes.asInstanceOf[AnyRef] != null) {
          val nn = nnodes.asInstanceOf[IMat];
          treeStep(fdata, nn, fnodes, itrees, ftrees, vtrees, ctrees, true);
        } else {
        	treeWalk(fdata, null, fnodes, itrees, ftrees, vtrees, ctrees, ipass0, true);
        }
      }
      case (fdata:FMat, fcats:FMat) => {
//      println("h=%d, hash=%f" format (here, sum(sum(fdata,2)).v))
        if (nnodes.asInstanceOf[AnyRef] != null) {
          val nn = nnodes.asInstanceOf[IMat];
          treeStep(fdata, nn, fnodes, itrees, ftrees, vtrees, ctrees, true);
        } else {
          treeWalk(fdata, null, fnodes, itrees, ftrees, vtrees, ctrees, ipass0, true);
        }
        if (datasource.opts.putBack == 1) {
        	val pcats = mean(fnodes);
          fcats <-- pcats;
        }
      }
    }
    ynodes = fnodes;
    if (opts.regression) {
      var mm = mean(fnodes);
      val diff = mm - FMat(cats)
      -mean(abs(diff)) //on -(diff dotr diff)/diff.length
    } else {
    	val mm = tally(fnodes);
    	mean(FMat(mm != IMat(cats)));
    }
  } 
  
  def tally(nodes:FMat):IMat = {
    val tallys = izeros(ncats, 1);
    val best = izeros(1, nodes.ncols);
    var i = 0;
    while (i < nodes.ncols) {
      var j = 0;
      var maxind = -1;
      var maxv = -1;
      tallys.clear
      while (j < nodes.nrows) {
         val ct = nodes.data(j + i * nodes.nrows).toInt;
         tallys.data(ct) += 1;
         if (tallys.data(ct) > maxv) {
           maxv = tallys.data(ct);
           maxind = ct;
         }
         j += 1;
      }
      best.data(i) = maxind;
      i += 1;
    }
    best
  }
  
  def tallyv(nodes:FMat):FMat = {
	  mean(nodes)
  }
  
  override def updatePass(ipass:Int) = { 
//    midmerge
    val tt = totals.getSum;
    tt.checkInds;
    t6 = toc;
    runtimes(5) += t6 - t5;
    val totalinds = tt.inds;
    val totalcounts = tt.counts;
    val (jc0, jtree) = findBoundaries(totalinds, jc);
    var itree = 0;
//    println("jc0="+jc0.t.toString);
    while (itree < ntrees) {
      t0 = toc;
      val gg = minImpurity(totalinds, totalcounts, outv, outf, outn, outg, outc, outleft, outright, jc0, jtree, itree, opts.impurity, opts.regression);
      t1 = toc;
      runtimes(6) += t1 - t0;
//      println("jc1 %d gg %d %d" format (jc1.length, gg.nrows, gg.ncols))
//      println("gg %s" format (gg.t.toString))
      val (vm, im) = maxi2(gg);                                         // Find feats with maximum -impurity gain
      val inds = im.t + icol(0->im.length) * gg.nrows;                  // Turn into an index for the "out" matrices
      val inodes = outn(inds);                                          // get the node indices
      ctrees(inodes, itree) = outc(inds);                               // Save the node class for these nodes
//      val reqgain = if (ipass < opts.depth - 1) opts.gain else Float.MaxValue;
      val reqgain = opts.gain
      val igain = find(vm > reqgain);                                   // find nodes above the impurity gain threshold
      gains(itree) = mean(vm).v
      igains(itree) = igain.length
      if (igain.length > 0) {
        val inn = inodes(igain);
        val igg = inds(igain);
        val ifff = outf(igg);
        if (! useIfeats) jfeatsToIfeats(itree, inn, ifff, seed, gitree, gftree);
        ftrees(inn, itree) = ifff;                    // Set the threshold features
        vtrees(inn, itree) = outv(igg);               // Threshold values
        val ibase = nodecounts(itree);
        itrees(inn, itree) = icol(ibase until (ibase + 2 * igain.length) by 2); // Create indices for new child nodes
        nodecounts(itree) += 2 * igain.length;                          // Update node counts for this tree
        tochildren(itree, inn, outleft(igg), outright(igg)); // Save class ids to children in class we don't visit them later
      }
      itree += 1;
      t2 = toc;
      runtimes(7) += t2 - t1;
    } 
    gitrees <-- itrees;
    gftrees <-- ftrees;
    gvtrees <-- vtrees;
    gctrees <-- ctrees;
    seed = opts.seed + 341211*(ipass+1);
    println("purity gain %5.4f, nnew %2.1f, nnodes %2.1f" format (mean(gains).v, 2*mean(igains).v, mean(FMat(nodecounts)).v));
//    if (ipass < opts.depth-1) { 
//      totalinds = new LMat(1,0,totalinds.data)
//    } else { 
//      totalinds = null;
//     totalcounts = null;
//    }
  }
  
  def tochildren(itree:Int, inodes:IMat, left:FMat, right:FMat) {
    var i = 0;
    while (i < inodes.length) {
      val inode = inodes(i);
      ctrees(itrees(inode, itree), itree) = left(i) ;
      ctrees(itrees(inode, itree)+1, itree) = right(i);
      i += 1;
    }

  }
  
  def rhash(v1:Int, v2:Int, v3:Int, nb:Int):Int = {
    math.abs(MurmurHash3.mix(MurmurHash3.mix(v1, v2), v3) % nb)
  }
  
  def rhash(v1:Int, v2:Int, v3:Int, v4:Int, nb:Int):Int = {
    math.abs(MurmurHash3.mix(MurmurHash3.mix(MurmurHash3.mix(v1, v2), v3), v4) % nb)
  }
  
  def packFields(itree:Int, inode:Int, jfeat:Int, ifeat:Int, ivfeat:Int, icat:Int):Long = {
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
  
  def unpackFields(im:Long):(Int, Int, Int, Int, Int, Int) = {
    var v = im;
    val icat = (v & ((1 << fieldlengths(ICat))-1)).toInt;
    v = v >>> fieldlengths(ICat);
    val ivfeat = (v & ((1 << fieldlengths(IVFeat))-1)).toInt;
    v = v >>> fieldlengths(IVFeat);
    val ifeat = (v & ((1 << fieldlengths(IFeat))-1)).toInt;
    v = v >>> fieldlengths(IFeat);
    val jfeat = (v & ((1 << fieldlengths(JFeat))-1)).toInt;
    v = v >>> fieldlengths(JFeat);
    val inode = (v & ((1 << fieldlengths(INode))-1)).toInt;
    v = v >>> fieldlengths(INode);
    val itree = v.toInt;
    (itree, inode, jfeat, ifeat, ivfeat, icat)
  }
  
  def extractAbove(fieldNum : Int, packedFields : Long) : Int = {
    (packedFields >>> fieldshifts(fieldNum)).toInt
  }

  def extractField(fieldNum : Int, packedFields : Long) : Int = {
    (packedFields >>> fieldshifts(fieldNum)).toInt & fieldmasks(fieldNum) 
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
  
  final val signbit:Int = 0x80000000;
  final val mag:Int = 0x7fffffff;

  
  @inline def floatConvert(a:Float):Int = {    
  	val vmask = fieldmasks(4);
  	val fshift = 32 - fieldlengths(4);
    var ai = java.lang.Float.floatToRawIntBits(a);
    if ((ai & signbit) > 0) {
      ai = -(ai & mag);
    }
    ai += signbit;
    (ai >> fshift) & vmask;
  }
  
  @inline def floatConvert2(a:Float):Int = {
    a.toInt
  }
  
  def treePack(fdata:FMat, treenodes:IMat, cats:IMat, out:LMat, seed:Int):LMat = {
    val nfeats = fdata.nrows;
    val nitems = fdata.ncols;
    val ntrees = treenodes.nrows;
    val ionebased = Mat.ioneBased;
    var icolx = 0;
    var nxvals = 0;
    while (icolx < nitems) {
      var itree = 0;
      while (itree < ntrees) {
        val inode = treenodes(itree, icolx);
        if (inode >= 0) {
          var jfeat = 0;
          while (jfeat < nsamps) {
            val ifeat = rhash(seed, itree, inode, jfeat, nfeats);
            val ivfeat = floatConvert(fdata(ifeat, icolx));
            val ic = cats(icolx);
            out.data(nxvals) = packFields(itree, inode, jfeat, if (useIfeats) ifeat else 0, ivfeat, ic);
            nxvals += 1;
            jfeat += 1;
          }
        }
        itree += 1;
      }
      icolx += 1;
    }
    Mat.nflops += 50L * nxvals 
    new LMat(nxvals, 1, out.data);
  }
  
  def treeStep(fdata:FMat, tnodes:IMat, fnodes:FMat, itrees:IMat, ftrees:IMat, vtrees:IMat, ctrees:FMat, getcat:Boolean)  {
    val nfeats = fdata.nrows;
    val nitems = fdata.ncols;
    val ntrees = tnodes.nrows;
    var icol = 0;
    while (icol < nitems) {
      var itree = 0;
      while (itree < ntrees) {
        var inode = tnodes(itree, icol);
        val ileft = itrees(inode, itree);
        if (ileft >= 0) {                                 // Has children so step down
          val ifeat = ftrees(inode, itree);
          val ithresh = vtrees(inode, itree);
          val ivfeat = floatConvert(fdata(ifeat, icol));
          if (ivfeat > ithresh) {
            inode = ileft + 1;
          } else {
            inode = ileft;
          }
        }
        if (getcat) {
          fnodes(itree, icol) = ctrees(inode, itree);
        } else {
        	tnodes(itree, icol) = inode;
        }
        itree += 1;
      }
      icol += 1;
    }
    Mat.nflops += 1L * nitems * ntrees; 
  }
  
  def treeWalk(fdata:FMat, tnodes:IMat, fnodes:FMat, itrees:IMat, ftrees:IMat, vtrees:IMat, ctrees:FMat, depth:Int, getcat:Boolean) = {
    val nfeats = fdata.nrows;
    val nitems = fdata.ncols;
    var icol = 0;
    while (icol < nitems) {
      var itree = 0;
      while (itree < ntrees) {
        var inode = 0;
        var id = 0;
        while (id < depth) {
          val ileft = itrees(inode, itree);
          if (ileft == 0) {                           // This is a leaf, so              
            id = depth;                               // just skip out of the loop
          } else {
            val ifeat = ftrees(inode, itree);         // Test this node and branch
            val ithresh = vtrees(inode, itree);
            val ivfeat = floatConvert(fdata(ifeat, icol));
            if (ivfeat > ithresh) {
              inode = ileft + 1;
            } else {
              inode = ileft;
            }
          }
          id += 1;
        }
        if (getcat) {
          fnodes(itree, icol) = ctrees(inode, itree);
        } else {
        	tnodes(itree, icol) = inode;
        }
        itree += 1;
      }
      icol += 1;
    }
    Mat.nflops += 1L * nitems * ntrees * depth;
    fnodes
  }
  
  def gtreeWalk(fdata:GMat, tnodes:GIMat, fnodes:GMat, itrees:GIMat, ftrees:GIMat, vtrees:GIMat, ctrees:GMat, depth:Int, getcat:Boolean) = {
    val nrows = fdata.nrows;
    val ncols = fdata.ncols;
    val err = CUMACH.treeWalk(fdata.data, tnodes.data, fnodes.data, itrees.data, ftrees.data, vtrees.data, ctrees.data, 
    		nrows, ncols, ntrees, nnodes, if (getcat) 1 else 0, nbits, depth);
    if (err != 0) {throw new RuntimeException("gtreeWalk: error " + cudaGetErrorString(err))}
  }
  
  def gtreeStep(gdata:GMat, tnodes:GIMat, fnodes:GMat, itrees:GIMat, ftrees:GIMat, vtrees:GIMat, ctrees:GMat, getcat:Boolean)  {}
    
  def gmakeV(keys:GLMat, vals:GIMat, tmpkeys:GLMat, tmpcounts:GIMat):SVec = {
    Mat.nflops += keys.length;
    val (ginds, gcounts) = GLMat.collectLVec(keys, vals, tmpkeys, tmpcounts);
    Mat.nflops += 1L * keys.length;
    val ovec = SVec(ginds.length);
    ovec.inds <-- ginds;
    ovec.counts <-- gcounts;
    ovec
  }

  def makeV(ind:LMat):SVec = {
    Mat.nflops += ind.length;
    val n = ind.length;
    val indd = ind.data;
    var ngroups = 0;
    var i = 1;
    while (i <= n) {
      if (i == n || indd(i) != indd(i-1)) {
        ngroups += 1;
      }
      i += 1;
    }
    val ovec = SVec(ngroups);
    val okeys = ovec.inds.data;
    val ovals = ovec.counts.data;
    var cc = 0;
    ngroups = 0;
    i = 1;
    while (i <= n) {
      cc += 1;
      if (i == n || indd(i) != indd(i-1)) {
        okeys(ngroups) = indd(i-1);
        ovals(ngroups) = cc;
        ngroups += 1;
        cc = 0;
      }
      i += 1;
    }
    ovec;
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
  
  // Add a short sparse Lvector (first arg) to a short one (2nd arg). Reuses the storage of the long vector. 
  
  def addV(ind1:LMat, counts1:IMat, ind2:LMat, counts2:IMat):(LMat, IMat) = {
    if (ind1.length + ind2.length > ind2.data.length) {
      throw new RuntimeException("temporary sparse Long storage too small %d %d" format (ind1.length+ind2.length, ind2.data.length));
    }
    val offset = ind1.length;
    var i = ind2.length - 1;
    while (i >= 0) {
      ind2.data(i + offset) = ind2.data(i);
      counts2.data(i + offset) = counts2.data(i);
      i -= 1;
    }
    var count = 0;
    var i1 = 0;
    val n1 = ind1.length;
    var i2 = offset;
    val n2 = ind2.length + offset;
    while (i1 < n1 || i2 < n2) {
      if (i1 >= n1 || (i2 < n2 && ind2.data(i2) < ind1.data(i1))) {
        ind2.data(count) = ind2.data(i2)
        counts2.data(count) = counts2.data(i2)
        count += 1
        i2 += 1
      } else if (i2 >= n2 || (i1 < n1 && ind1.data(i1) < ind2.data(i2))) {
        ind2.data(count) = ind1.data(i1)
        counts2.data(count) = counts1.data(i1)
        count += 1
        i1 += 1
      } else {
        ind2.data(count) = ind1.data(i1)
        counts2.data(count) = counts1.data(i1) + counts2.data(i2)
        count += 1
        i1 += 1
        i2 += 1
      }
    }
    (new LMat(1, count, ind2.data), new IMat(1, count, counts2.data))
  }
  
  def gaddV(gix:GLMat, gcx:GIMat, gmidinds:GLMat, gmidcounts:GIMat, gmergedinds:GLMat, gmergedcounts:GIMat):(GLMat, GIMat) = {
    val (ai, ac) = GLMat.mergeLVecs(gix, gcx, gmidinds, gmidcounts, gmergedinds, gmergedcounts);
    GLMat.collectLVec(ai, ac, gmidinds, gmidcounts);
  }
  
  def copyinds(inds:LMat, tmp:LMat) = {
    val out = new LMat(inds.length, 1, tmp.data);
    out <-- inds;
    out
  }
  
  def copycounts(cnts:IMat, tmpc:IMat) = {
    val out = new IMat(cnts.length, 1, tmpc.data);
    out <-- cnts;
    out
  }
  
  def gtreePack(fdata:FMat, tnodes:IMat, icats:IMat, gout:GLMat, seed:Int):GLMat ={
    val nrows = fdata.nrows
    val ncols = fdata.ncols
    val nxvals = ncols * ntrees * nsamps;
    Mat.nflops += 1L * nxvals;
    val gdata = GMat(fdata);
    val gcats = GIMat(icats);
    cudaMemcpy(gtnodes.data, Pointer.to(tnodes.data), ncols*ntrees*Sizeof.INT, cudaMemcpyHostToDevice)
    cudaDeviceSynchronize();
    var err = cudaGetLastError
    if (err != 0) {throw new RuntimeException("fgtreePack: error " + cudaGetErrorString(err))}
    err= CUMACH.treePack(gdata.data, gtnodes.data, gcats.data, gout.data, gfieldlengths.data, nrows, ncols, ntrees, nsamps, seed)
    if (err != 0) {throw new RuntimeException("fgtreePack: error " + cudaGetErrorString(err))}
    new GLMat(1, nxvals, gout.data, gout.realsize);
  }
  
  def gtreePack(gdata:GMat, gtnodes:GIMat, gcats:GIMat, gout:GLMat, seed:Int):GLMat ={
    val nrows = gdata.nrows
    val ncols = gdata.ncols
    val nxvals = ncols * ntrees * nsamps;
    Mat.nflops += 1L * nxvals;
    val err= CUMACH.treePack(gdata.data, gtnodes.data, gcats.data, gout.data, gfieldlengths.data, nrows, ncols, ntrees, nsamps, seed)
    if (err != 0) {throw new RuntimeException("gtreePack: error " + cudaGetErrorString(err))}
    new GLMat(1, nxvals, gout.data, gout.realsize);
  }
  
  def gpsort(gout:GLMat) = {
    val nxvals = gout.length;
    Mat.nflops += 2L * nxvals * math.log(nxvals).toInt;
    val err = CUMAT.lsort(gout.data, nxvals, 1);
    if (err != 0) {throw new RuntimeException("gpsort: error " + cudaGetErrorString(err))}
    cudaDeviceSynchronize()
  }
  
  def jfeatsToIfeats(itree:Int, inodes:IMat, ifeats:IMat, seed:Int, gitree:GIMat, gftree:GIMat) {
    if (onGPU) {
      gjfeatsToIfeats(itree, inodes, ifeats, seed, gitree, gftree)
    } else {
    	val len = inodes.length;
    	var i = 0;
    	while (i < len) {
        val inode = inodes.data(i);
        val jfeat = ifeats.data(i);
        val ifeat = rhash(seed, itree, inode, jfeat, nfeats);
        ifeats(i) = ifeat;
    		i += 1;
    	}   
    }
  }
  
  def gjfeatsToIfeats(itree:Int, inodes:IMat, ifeats:IMat, seed:Int, gitree:GIMat, gftree:GIMat) {
    val len = inodes.length;
    val gi = new GIMat(inodes.nrows, inodes.ncols, gitree.data, gitree.realsize);
    val gf = new GIMat(ifeats.nrows, ifeats.ncols, gftree.data, gftree.realsize);
    gi <-- inodes;
    gf <-- ifeats;
    val err = CUMACH.jfeatsToIfeats(itree, gi.data, gf.data, gf.data, len, nfeats, seed);
    if (err != 0) {throw new RuntimeException("gjfeatsToIfeats: error " + cudaGetErrorString(err))}
    ifeats <-- gf;
  }

  // Find boundaries where JFeat or ITree changes
  
  def findBoundaries(keys:LMat, jc:IMat):(IMat,IMat) = { 
    val fieldshifts = getFieldShifts(fieldlengths);
    val fshift = fieldshifts(JFeat);
    val tshift = fieldshifts(ITree);
    val tmat = izeros(ntrees+1,1);
    var oldv = -1L;
    var v = -1;
    var t = 0;
    var nt = 0;
    var i = 0
    var n = 0;
    while (i < keys.length) {
      v = extractAbove(JFeat, keys(i));
      t = (keys(i) >>> tshift).toInt;
      while (t > nt) {
        tmat(nt+1) = n;
        nt += 1;
      }
      if (oldv != v) {
        jc(n) = i;
        n += 1;
        oldv = v;
      }
      i += 1
    }
    jc(n) = i;
    while (ntrees > nt) {
      tmat(nt+1) = n;
      nt += 1;
    }
    n += 1;
    if ((n-1) % nsamps != 0) throw new RuntimeException("boundaries %d not a multiple of nsamps %d" format (n-1, nsamps));
    (new IMat(n, 1, jc.data), tmat)
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
  
   /*object varImpurity extends imptyType {
    def updatefn(a:Int):Double = { val v = a; v * v }
    def resultfn(acc:Double, tot:Int, n:Int):Double = {val v:Double = tot; acc - v*v/n }
    def combinefn(a1:Double, a2:Double, tot1:Int, tot2:Int, n1:Int, n2:Int):Double = { 
      val n = n1+n2; val tot:Double = tot1 + tot2; (a1 + a2 - tot*tot/n)/n   }
    val update = updatefn _ ;
    val result = resultfn _ ;
    val combine = combinefn _ ;
    }*/

  
  val imptyFunArray = Array[imptyType](entImpurity,giniImpurity)
  
  // Pass in one of the two object above as the last argument (imptyFns) to control the impurity
  // outv should be an nsamps * nnodes array to hold the feature threshold value
  // outf should be an nsamps * nnodes array to hold the feature index
  // outg should be an nsamps * nnodes array holding the impurity gain (use maxi2 to get the best)
  // jc should be a zero-based array that points to the start and end of each group of fixed node, jfeat

  def minImpurity(keys:LMat, cnts:IMat, outv:IMat, outf:IMat, outn:IMat, outg:FMat, outc:FMat, outleft:FMat, outright:FMat, 
		  jc:IMat, jtree:IMat, itree:Int, fnum:Int, regression:Boolean):FMat = {
    
    val update = imptyFunArray(fnum).update
    val result = imptyFunArray(fnum).result
    val combine = imptyFunArray(fnum).combine

    val totcounts = izeros(1,ncats);
    val counts = izeros(1,ncats);
    val fieldshifts = getFieldShifts(fieldlengths);
    val fieldmasks = getFieldMasks(fieldlengths);

    var j = 0;
    var tot = 0;
    var tott = 0;
    var acc = 0.0;
    var acct = 0.0;
    var i = 0;
    val todo = jtree(itree+1) - jtree(itree);
    Mat.nflops += todo * 4L * 10;
    var all = 0.0;
    var impure = 0.0;
    while (i < todo) {
      val jci = jc(i + jtree(itree));
      val jcn = jc(i + jtree(itree) + 1);

      totcounts.clear;
      counts.clear;
      tott = 0;
      j = jci;
      var maxcnt = -1;
      var imaxcnt = -1;
      var totcats = 0.0;
      while (j < jcn) {                     // First get the total counts for each group, and the most frequent cat
        val key = keys(j)
        val cnt = cnts(j)
        val icat = extractField(ICat, key);
        val newcnt = totcounts(icat) + cnt;
        totcounts(icat) = newcnt;
        totcats += cnt * icat;
        tott += cnt;
        if (newcnt > maxcnt) {
          maxcnt = newcnt;
          imaxcnt = icat;
        }
        j += 1;
      }
      val inode = extractField(INode, keys(jci));
      val ifeat = extractField(if (useIfeats) IFeat else JFeat, keys(jci));
      var minImpty = 0.0;
      var lastImpty = 0.0;
      var nodeImpty = 0.0;
      var partv = -1;
      var lastkey = -1L;
      var jmaxcnt = 0;
      var kmaxcnt = 0;
      all += tott;
      var lefttotcats = 0.0;
      var lefttot = 0;
      if (maxcnt < tott) { // This is not a pure node
    	  totcats = 0.0;
        impure += tott;
      	acct = 0; 
      	//      println("totcounts "+totcounts.toString);
      	j = 0;
      	while (j < ncats) {                  // Get the impurity for the node
      		acct += update(totcounts(j));
      		j += 1
      	}
      	nodeImpty = result(acct, tott);

      	var lastival = -1;
      	minImpty = nodeImpty;
      	lastImpty = Double.MaxValue;
      	acc = 0;
      	tot = 0;
      	j = jci;
      	maxcnt = -1;
      	var jmax = j;

      	while (j < jcn) {     
      		val key = keys(j);
      		val cnt = cnts(j);
      		val ival = extractField(IVFeat, key);
      		val icat = extractField(ICat, key);  

      		if (j > jci && ival != lastival) {
      			lastImpty = combine(result(acc, tot), result(acct, tott - tot), tot, tott - tot);   // Dont compute every time!
      			if (lastImpty < minImpty) { 
      				minImpty = lastImpty;
      				partv = lastival;
      				jmax = j;
              lefttotcats = totcats;
              lefttot = tot;
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
      		totcats += cnt * icat;
      		lastkey = key;
      		lastival = ival;
      		j += 1;
      	}
        if (! regression) {
        	counts.clear;
        	maxcnt = -1;
        	while (j > jmax) {
        		j -= 1;
        		val key = keys(j);
        		val cnt = cnts(j);
        		val ival = extractField(IVFeat, key);
        		val icat = extractField(ICat, key);
        		val oldcnt = counts(icat);
        		val newcnt = oldcnt + cnt;
        		counts(icat) = newcnt;
        		if (newcnt > maxcnt) {
        			maxcnt = newcnt;
        			kmaxcnt = icat;
        		}        
        	} 
        }
//      	lastImpty = combine(result(acc, tot), result(acct, tott - tot), tot, tott - tot);   // For checking
      }
//      println("Impurity %f, %f, min %f, %d, %d" format (nodeImpty, lastImpty, minImpty, partv, ifeat))
      outv(i) = partv;
      outg(i) = (nodeImpty - minImpty).toFloat;
      outf(i) = ifeat;
      if (regression) {
        val defv = if (tott > 0) totcats.toFloat / tott else ncats/2.0f;
        outc(i) = defv;
        outleft(i) = if (lefttot > 0) lefttotcats.toFloat / lefttot else defv;
        outright(i) = if (tott - lefttot > 0) (totcats - lefttotcats) / (tott - lefttot) else defv;
      } else {
    	  outc(i) = imaxcnt;
    	  outleft(i) = jmaxcnt;
    	  outright(i) = kmaxcnt;
      }
      outn(i) = inode;
      i += 1;
    }
    if (opts.trace > 0) println("fraction of impure nodes %f" format impure/all);
    new FMat(nsamps, todo/nsamps, outg.data);
  }
  
  def save(fname:String) = {
    saveIMat(fname+"itrees.imat.lz4", itrees);
    saveIMat(fname+"ftrees.imat.lz4", ftrees);
    saveIMat(fname+"vtrees.imat.lz4", vtrees);
    saveFMat(fname+"ctrees.fmat.lz4", ctrees);
  }
  
  def load(fname:String) = {
    itrees = loadIMat(fname+"itrees.imat.lz4");
    ftrees = loadIMat(fname+"ftrees.imat.lz4");
    vtrees = loadIMat(fname+"vtrees.imat.lz4");
    ctrees = loadFMat(fname+"ctrees.fmat.lz4");
  }

}

class SVec(val inds:LMat, val counts:IMat) { 

  def length = inds.length

  def add(b:SVec):SVec = { 
    
    val inds1 = inds.data;
    val counts1 = counts.data;
    val inds2 = b.inds.data;
    val counts2 = b.counts.data;

    var count = 0;
    var i1 = 0;
    val n1 = length;
    var i2 = 0;
    val n2 = b.length;
    // First calculate the output size
    while (i1 < n1 || i2 < n2) {
      if (i1 >= n1 || (i2 < n2 && inds2(i2) < inds1(i1))) {
        count += 1;
        i2 += 1;
      } else if (i2 >= n2 || (i1 < n1 && inds1(i1) < inds2(i2))) {
        count += 1;
        i1 += 1;
      } else {
        count += 1;
        i1 += 1;
        i2 += 1;
      }
    }
    // now make the output vector
    val out = SVec(count);
    val inds3 = out.inds.data;
    val counts3 = out.counts.data;
    count = 0;
    i1 = 0;
    i2 = 0;
    while (i1 < n1 || i2 < n2) {
      if (i1 >= n1 || (i2 < n2 && inds2(i2) < inds1(i1))) {
        inds3(count) = inds2(i2);
        counts3(count) = counts2(i2);
        count += 1;
        i2 += 1;
      } else if (i2 >= n2 || (i1 < n1 && inds1(i1) < inds2(i2))) {
        inds3(count) = inds1(i1);
        counts3(count) = counts1(i1);
        count += 1;
        i1 += 1;
      } else {
        inds3(count) = inds1(i1);
        counts3(count) = counts1(i1) + counts2(i2);
        count += 1;
        i1 += 1;
        i2 += 1;
      }
    }
    out
  }

  def checkInds = { 
    var i = 0;
    val len = length;
    val ii = inds.data;
    while (i < len - 1) { 
      if (ii(i) > ii(i+1)) { 
        throw new RuntimeException("bad order %d %d %d" format (i, ii(i), ii(i+1)));
      }
      i += 1;
    }
  }


}

class SVTree(val n:Int) { 
  val tree = new Array[SVec](n);
  
  def showTree = { 
    var i = 0;
    while (i < n) { 
      if (tree(i) != null) { 
        print(" %d" format tree(i).length);
      } else { 
        print(" 0");
      }
      i += 1;
    }
    println("");
  }

  def addSVec(a:SVec) = { 
    var here = a;
    var i = 0;
    while (tree(i) != null) { 
      here = tree(i).add(here);
      tree(i) = null;
      i += 1;
    }
    tree(i) = here;
  }

  def getSum:SVec = { 
    var i = 0;
    var here:SVec = null;
    while (i < n && tree(i) == null) { 
      i += 1;
    }
    if (i < n) { 
      here = tree(i);
      tree(i) = null;
    }
    i += 1;
    while (i < n) { 
      if (tree(i) != null) { 
        here = tree(i).add(here);
        tree(i) = null;
      }
      i += 1;
    }
    here;
  }
}

object SVec { 
  def apply(n:Int):SVec = { 
    new SVec(lzeros(1,n), izeros(1,n))
  }
}

object RandomForest {
    
  trait Opts extends Model.Opts { 
     var depth = 32;
     var ntrees = 32;
     var nsamps = 32;
     var nnodes = 100000;
     var nbits = 8;
     var gain = 0.01f;
     var catsPerSample = 1f;
     var ncats = 0;
     var training = true;
     var impurity = 0;  // zero for entropy, one for Gini impurity
     var regression = false;
     var seed = 1;
     var useIfeats = false; // explicitly save Ifeat indices (vs. compute them)
     var GPU = true;
     var trace = 0;
  }
  
  class Options extends Opts {}
  
  class RFopts extends Learner.Options with RandomForest.Opts with DataSource.Opts with Batch.Opts;
  
  class RFSopts extends RFopts with MatDS.Opts;
  
  def learner(data:IMat, labels:SMat) = {
    val opts = new RFSopts;
    opts.useGPU = false;
    opts.nbits = countbits(maxi(maxi(data)).dv.toInt);
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
    val nn = new Learner(
        ds, 
        new RandomForest(opts), 
        null, 
        new Batch(opts),
        opts)
    (nn, opts)
  }
  
  def predictor(model:RandomForest, ds:DataSource, opts:RFopts):(Learner, RFopts) = {
    val nn = new Learner(
        ds, 
        model,
        null, 
        new Batch(opts),
        opts)
    (nn, opts)
  }
  
  def predictor(modelname:String, ds:DataSource):(Learner, RFopts) = {
    val opts = new RFopts;
    opts.useGPU = false;
    val model = new RandomForest(opts);
    model.load(modelname);
    predictor(model, ds, opts)
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
  
  def floatToInt(in:GMat, out:Mat, nbits:Int):GIMat = {
    val omat = GIMat.newOrCheckGIMat(in.nrows, in.ncols, out, in.GUID, "floatToInt".##)
    edu.berkeley.bid.CUMACH.floatToInt(in.length, in.data, omat.data, nbits)
    omat
  }
  
  def floatToInt(in:GMat, nbits:Int):GIMat = floatToInt(in, null, nbits)
  
  def countbits(n:Int):Int = {
    var i = 0;
    var j = 1;
    while (j < n) {
      j *= 2;
      i += 1;
    }
    i
  }
}
