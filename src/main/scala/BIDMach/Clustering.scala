package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._


class PAMmodel(val opts:PAMmodel.Options = new PAMmodel.Options) { 
  var a:FMat = null
  var nfeats = 0
  var nsamps = 0
  var ncenters = 0
  var ntrys = 0
  val options = opts
  var maxdepth = 0
  var nspills = 0
  var bestc:IMat = null
  
  def init(a0:FMat) = {
    a = a0
    nfeats = size(a0,2)
    nsamps = size(a0,1)
    ncenters = options.ncenters
    ntrys = options.ntrys
  }
  
  def dists(a:FMat):FMat = {
    val dd = if (Mat.hasCUDA > 0) a xTG a else a xT a;
    val d1 = getdiag(dd)
    dd ~ dd * 2.0f
    dd ~ d1 - dd
    dd ~ dd + (d1.t)
    max(dd, 0f, dd)
    sqrt(dd, dd)
    dd
  }
  
  def mindists(ds:FMat, iss:IMat, isamps:IMat, icenters:IMat, vmin:DMat, imin:IMat) = {
    val ncache = size(iss,1)
    val centmap = accum(icenters, icol(1 to length(icenters)), size(ds,2), 1)
    var i = 0
    while (i < length(isamps)) {
      val ii = isamps(i)
      var continue = true
      var j = 0
      while (j < ncache && continue) {
        if (centmap(iss(j, ii)) > 0) {
        	imin(ii) = centmap(iss(j, ii)) - 1
        	vmin(ii) = ds(j, ii)
        	continue = false
        } 
        j += 1
      }
      maxdepth = math.max(maxdepth, j)
      if (j > 10*nsamps/ncenters) nspills += 1
      Mat.nflops += 4*j
      i += 1
    }
  }
  
  def mindists(ds:FMat, iss:IMat, isamps:IMat, icenters:IMat):(DMat, IMat) = {
    val vmin = dzeros(nsamps,1)
    val imin = izeros(nsamps,1)
    mindists(ds, iss, isamps, icenters, vmin, imin)
    (vmin, imin)
  }
  
  def mindists2(ds:FMat, iss:IMat, isamps0:IMat, icenters:IMat, vmin:FMat, imin:IMat) = {
    val ncache = size(iss,2)
    val centmap = accum(icenters, icol(1 to length(icenters)), size(ds,1), 1)
    var isamps = isamps0
    var idepth = 0
    while (isamps.length > 0) {
      val isx = centmap(iss(isamps, idepth), 0)
    	val ifound = find(isx > 0)
    	imin(isamps(ifound)) = isx(ifound) - 1
    	vmin(isamps(ifound)) = ds(isamps(ifound), idepth)
      Mat.nflops += 4*isamps.length
      isamps = isamps(find(isx == 0))
      idepth += 1
      maxdepth = math.max(maxdepth, idepth)
    }
  }
  
  def mindists2(ds:FMat, iss:IMat, isamps:IMat, icenters:IMat):(FMat, IMat) = {
    val vmin = zeros(nsamps,1)
    val imin = izeros(nsamps,1)
    mindists2(ds, iss, isamps, icenters, vmin, imin)
    (vmin, imin)
  }
  
  def pointdiffs(ds:FMat, iss:IMat, vd:DMat):DMat = {
	  val deltas = dzeros(nsamps,1)                                      // Array to hold improvements in distance over vd
	  var i = 0  
	  while (i < nsamps) {                                               // Calculate improvements over vd for new candidate centers
	  	var j = 0  
	  	while (j < nsamps && ds(j,i) < vd(i)) {                          // using sorted order of ds
	  		deltas(iss(j,i)) += ds(j,i) - vd(i)
	  		j += 1
	  	} 
	  	maxdepth = math.max(maxdepth, j)
	  	Mat.nflops += 16*j
	  	i += 1
	  }
	  deltas
  }
  
  def pointdiffs2(ds:FMat, iss:IMat, vd:FMat):FMat = {
	  val deltas = zeros(nsamps,1)                                       // Array to hold improvements in distance over vd
	  var ii = icol(0->nsamps)
	  var idepth = 0  
	  while (ii.length > 0) {                                            // Calculate improvements over vd for new candidate centers
	  	ii = ii(find(ds(ii,idepth) < vd(ii,0)))
	  	var j = 0
	  	while (j < ii.length) {
	  		deltas(iss(ii(j),idepth)) +=  (ds(ii(j),idepth) - vd(ii(j)))
	  		j += 1
	  	}
	  	Mat.nflops += 16*j
	  	idepth += 1
	  	maxdepth = math.max(maxdepth, idepth)
	  }
	  deltas
  }
  
  def sortgen(dd:FMat):(FMat,IMat) = {
    if (Mat.hasCUDA <= 0) {  // until GPUsort fixed
      sort2(dd,1)
    } else {
      val smat = dd.copy
      val imat = icol(0->nsamps)*iones(1,nsamps)
      GMat.sortGPU(smat, imat)
      (smat, imat)
    }
  }
  
  def run = {
    println("PAM clustering %d points with %d features into %d centers" format (nsamps, nfeats, ncenters))
    flip
    val dd = dists(a)
    val ft1 = gflop
    println("Distances in %f seconds, %f gflops" format (ft1._2,ft1._1))
    flip
    val (ds, iss) = sortgen(dd)                                        // Sort the distances
    Mat.nflops += math.round(math.log(size(ds,1))/math.log(2.0))*size(ds,1)*size(ds,2)
    val ft2 = gflop
    println("Sort in %f seconds, %f gcomps" format (ft2._2,ft2._1))
    var bestv:DMat = null
    var besti:IMat = null
    var bestvd = Double.MaxValue 
    flip
    var itry = 0
    while (itry < ntrys) {
      println("Try %d" format itry)
    	val rr = rand(nsamps,1)                                            // Get a random permutation for the centers
    	val (rs,irs) = sort2(rr,1)
    	val icenters = irs(0->ncenters,0)                                  // Pick centers from the permutation
    	val ics = icol(0->nsamps)                            
    	val (vdists, imin) = mindists(ds, iss, ics, icenters)              // Get min distances from points to centers, and best center ids
    	println("  pass=0, mean dist=%f" format mean(vdists,1).v)
    	val vtmp = vdists.copy
    	val itmp = imin.copy
    	var nchanged = 1
    	var ipass = 0
    	var totchanged = 0
    	while (nchanged > 0 && ipass < options.maxpasses) {                // Keep making passes until no improvements
    		ipass += 1
    		nchanged = 0
    		var ipc = 0  
    		while (ipc < ncenters) {                                         // Try to improve this center (ipc)
    			vtmp <-- vdists                                                // Copy distances 
    			val ifix = find(imin == ipc)                                   // Find points in cluster with this center
    			val tcents = icenters((0->ipc) \ ((ipc+1)->ncenters),0)        // List of centers minus the current one
    			mindists(ds, iss, ifix, tcents, vtmp, itmp)                    // vtmp holds distances to centers minus the current center
    			val deltas = pointdiffs(ds, iss, vtmp)                         // deltas holds improvements for each potential center over vtmp
    			val (vs,is) = mini2(deltas)                                    // Find best new center
    			if (vs.v + sum(vtmp).v < sum(vdists).v && is.v != icenters(ipc,0)) { // Is the new center better than the old (and not equal to it)?
    				icenters(ipc) = is.v                                         // If yes, update the center list
    				mindists(ds, iss, ics, icenters, vdists, imin)               // Compute new distances and centers
    				nchanged += 1
    				if (options.verb) println("    pass=%d, ipc=%d, mean dist=%f, nchanged=%d" format (ipass, ipc, mean(vdists,1).v, nchanged))
    			}
    			ipc += 1
    		}
    		println("  pass=%d, mean dist=%f, nchanged=%d, nspills=%d" format (ipass, mean(vdists,1).v, nchanged, nspills))
    		totchanged += nchanged
    	}
    	val mv = mean(vdists).v
    	if (mv < bestvd) {
    	  bestc = icenters
    	  bestv = vdists
    	  besti = imin
    	  bestvd = mv    	  
    	}
    	itry += 1
    }
    val t3=gflop
    val vdists2 = mini(dd(?,bestc),2)
    println("Optimum in %f secs, %f gflops, mean dist=%f, verify=%f, maxdepth=%d, nspills=%d\nTotal time %f seconds" format 
    		(t3._2, t3._1, bestvd, mean(DMat(vdists2),1).v, maxdepth, nspills, t3._2+ft2._2+ft1._2))
  }
  
}

object PAMmodel { 
  class Options { 
    var ncenters = 1000
    var maxpasses = 10
    var ntrys = 1
    var verb = false
  }
  
  def runit(nsamps:Int, nfeats:Int, ncenters:Int) = {
    println("Generating dataset")
    val c = rand(ncenters, nfeats)
    val a = rand(nsamps, nfeats)*0.3f
    for (i <- 0 until nsamps by ncenters) {val il = math.min(i+ncenters, nsamps); a(i->il,?) += c(0->(il-i),?)}
    val cc = new PAMmodel
    cc.options.ncenters = ncenters
    cc.init(a)
    cc.run
  }
  
  def main(args:Array[String]) = {
    Mat.checkCUDA
    val nsamps= args(0).toInt
    val nfeats = args(1).toInt
    val ncenters = args(2).toInt
    runit(nsamps, nfeats, ncenters)
  }
}
