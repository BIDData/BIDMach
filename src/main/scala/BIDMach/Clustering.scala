package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._


class PAMmodel(opts:PAMmodel.Options = new PAMmodel.Options) { 
  var a:FMat = null
  var nfeats = 0
  var nsamps = 0
  var ncenters = 0
  val options = opts
  
  def init(a0:FMat) = {
    a = a0
    nfeats = size(a0,2)
    nsamps = size(a0,1)
    ncenters = options.ncenters   
  }
  
  def dists(a:FMat):FMat = {
    val dd = a xT a;
    val d1 = getdiag(dd)
    dd ~ dd * 2.0f
    dd ~ d1 - dd
    dd ~ dd + (d1.t)
    max(dd, 0f, dd)
    sqrt(dd, dd)
    dd
  }
  
  def mindists(ds:FMat, iss:IMat, isamps:IMat, icenters:IMat, vmin:FMat, imin:IMat) = {
    val ncache = size(iss,1)
    val centmap = accum(icenters, icol(1 to length(icenters)), size(ds,2), 1)
    var i = 0
    while (i < length(isamps)) {
      val ii = isamps(i, 0)
      var continue = true
      var j = 0
      while (j < ncache && continue) {
        if (centmap(iss(j, ii), 0) > 0) {
        	imin(ii, 0) = centmap(iss(j, ii), 0) - 1
        	vmin(ii, 0) = ds(j, ii)
        	continue = false
        } 
        j += 1
      }
      Mat.nflops += 4*j
      i += 1
    }
  }
  
  def mindists(ds:FMat, iss:IMat, isamps:IMat, icenters:IMat):(FMat, IMat) = {
    val vmin = zeros(nsamps,1)
    val imin = izeros(nsamps,1)
    mindists(ds, iss, isamps, icenters, vmin, imin)
    (vmin, imin)
  }
  
  def pointdiffs(ds:FMat, iss:IMat, vd:FMat):FMat = {
	  val deltas = zeros(nsamps,1)                                       // Array to hold improvements in distance over vd
	  var ipn = 0  
	  while (ipn < nsamps) {                                             // Calculate improvements over vd for new candidate centers
	  	var j = 0  
	  	while (j < nsamps && ds(j,ipn) < vd(ipn,0)) {                    // using sorted order of ds
	  		deltas(iss(j,ipn),0) = deltas(iss(j,ipn),0) + (ds(j,ipn) - vd(ipn,0))
	  		j += 1
	  	} 
	  	Mat.nflops += 16*j
	  	ipn += 1
	  }
	  deltas
  }
  
  def run = {
    flip
    val dd = dists(a)
    val ft1 = gflop
    println("distance in %f seconds, %f gflops" format (ft1._2,ft1._1))
    flip
    val (ds, iss) = sort2(dd,1)                                        // Sort the distances
    Mat.nflops += math.round(math.log(size(ds,1))/math.log(2.0))*size(ds,1)*size(ds,2)
    val ft2 = gflop
    println("sorts in %f seconds, %f gcomps" format (ft2._2,ft2._1))
    val rr = rand(nsamps,1)                                            // Get a random permutation for the centers
    val (rs,irs) = sort2(rr,1)
    val icenters = irs(0->ncenters,0)                                  // Pick centers from the permutation
    val ics = icol(0->nsamps)                            
    val (vdists, imin) = mindists(ds, iss, ics, icenters)              // Get min distances from points to centers, and best center ids
    println("ipass=0, dist=%f" format mean(vdists,1).v)
    flip
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
        val deltas = pointdiffs(ds, iss, vtmp)                         // deltas holds improvements for each center over vtmp
      	val (vs,is) = mini2(deltas)                                    // Find best new center
      	if (vs.v + sum(vtmp).v < sum(vdists).v && is.v != icenters(ipc,0)) { // Is the new center better than the old (and not equal to it)?
      	  icenters(ipc,0) = is.v                                       // If yes, update the center list
      	  mindists(ds, iss, ics, icenters, vdists, imin)               // Compute new distances and centers
      	  nchanged += 1
      	  if (options.verb) println("ipass=%d, ipc=%d, dist=%f, nchanged=%d" format (ipass, ipc, mean(vdists,1).v, nchanged))
      	}
        ipc += 1
      }
      println("ipass=%d, dist=%f, nchanged=%d" format (ipass, mean(vdists,1).v, nchanged))
      totchanged += nchanged
    }
    val (vdists2, imin2) = mini2(dd(?,icenters),2)
    val t3=gflop
    println("optim in %f secs, %f gflops, passes=%d, dist=%f, verify=%f, totchanged=%d" format (t3._2, t3._1, ipass, mean(vdists).v, mean(vdists2,1).v, totchanged))
  }
  
}

object PAMmodel { 
  class Options { 
    var ncenters = 1000
    var maxpasses = 10
    var verb = false
  }
  
  def main(args:Array[String]) = {
    val nfeats = args(0).toInt
    val nsamps = args(1).toInt
    val ncenters = args(2).toInt
    val a = rand(nsamps, nfeats)
    val cc = new PAMmodel
    cc.options.ncenters = ncenters
    cc.init(a)
    cc.run
  }
}
