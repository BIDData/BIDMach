package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._


class PAMmodel(opts:PAMmodel.Options = new PAMmodel.Options) { 


  var a:FMat = null
  var nfeats = 0
  var nsamps = 0
  var ncenters = 0
  var ntrys = 0
  val options = opts
  var maxdepth = 0
  var nspills = 0
  var bestc:IMat = null
  var imin:IMat = null
  var vdists:DMat = null
  var sil:DMat = null
  var mss = 0.0

  var ncache = 0
  

  def dists(x:FMat) = options.metric.dists(x:FMat)
  def dists(x:FMat, y:FMat) = options.metric.dists(x:FMat, y:FMat)
  
  def init(a0:FMat) = {

    a = a0
    nfeats = size(a0,2)
    nsamps = size(a0,1)
    ncenters = options.ncenters
    ntrys = options.ntrys
    vdists = dzeros(nsamps,1)
    imin = izeros(nsamps,1)
    sil = dzeros(nsamps,1)

    ncache = min(nsamps,options.cbal*nsamps/ncenters)(0,0)

  }


  // Silhouette Score
  def silhouetteScore(ds:FMat, iss:IMat, isamps:IMat, icenters:IMat, lab:IMat):DMat = {
    
    var aa = zeros(nsamps,1)
    var bb = zeros(nsamps,1)

    var silhouette = zeros(nsamps,1)

    var dimclu = zeros(length(icenters),1) // Size of each cluster
    for( i <- 0 to nsamps-1 ) { dimclu(lab(i))+=1 }
    
    val ncache = size(iss,1)
    val centmap = accum(icenters, icol(1 to length(icenters)), size(ds,2), 1)
    var i = 0
    while (i < length(isamps)) {
      
      val ii = isamps(i)
      val labi = lab(i)

      var maxsize = dimclu(labi) + 1

      var first = true
      var j = 0
      var count = 0

      while (j < ncache && count < maxsize) {
        

        if(lab(j)==labi){
          aa(i)+=ds(j, ii)
          }else { // pick the second closest center
	    
	    if(first){ 

            val labj = lab(j)
            maxsize += dimclu(labj) - 1
            first = false

	      
            }
            bb(i)+=ds(j, ii)
          }

        j += 1
      }
      silhouette(i) = (bb(i) - aa(i))  / max(aa(i),bb(i))
      i += 1
    }

    silhouette


  }

  
  def mindists(ds:FMat, iss:IMat, isamps:IMat, icenters:IMat, vmin:DMat, imin:IMat) = {
    
    val centmap = accum(icenters, icol(1 to length(icenters)), size(ds,2), 1)
    var i = 0
    var ispills = izeros(1,nsamps)

    var spills = 0

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

      
      if (j >= ncache & continue) 
      {
        
        ispills(spills) = i
        spills += 1
        nspills += 1

        imin(ii) = -1
        vmin(ii) = -1

      }
      Mat.nflops += 4*j
      i += 1
    }

    if(spills>0)
    {

      ispills = ispills(0,0 until spills)


      val dspill = dists(a(icenters(?,0),?),a(ispills,?))

      var (ddd,iii) = maxi2(dspill,1) 

      imin(ispills) = centmap(iii)
      for (i <- 0 until spills){   vmin(ispills(i),0) = ddd(0,i)}

      }

  }
  
  def mindists(ds:FMat, iss:IMat, isamps:IMat, icenters:IMat):(DMat, IMat) = {
    val vmin = dzeros(nsamps,1)
    val imin = izeros(nsamps,1)
    mindists(ds, iss, isamps, icenters, vmin, imin)
    (vmin, imin)
  }
  
  
  def pointdiffs(ds:FMat, iss:IMat, vd:DMat):DMat = {
    val deltas = dzeros(nsamps,1)                                      // Array to hold improvements in distance over vd

    var ispills = izeros(1,nsamps)
    var spills = 0

    var i = 0  
    while (i < nsamps) {                                               // Calculate improvements over vd for new candidate centers
      var j = 0  
      while (j < ncache && ds(j,i) < vd(i)) {                          // using sorted order of ds
        deltas(iss(j,i)) += ds(j,i) - vd(i)
        j += 1
      } 
      maxdepth = math.max(maxdepth, j)
      if (j >= ncache){
        if(ds(j-1,i) < vd(i)) 
        {
          ispills(spills) = i
          nspills += 1
          spills += 1
        }
      }

      Mat.nflops += 4 * j
      i += 1
    }

    if(spills > 0)
    {

      ispills = ispills(?,0 until spills)
      val threshold = ds(ncache-1,ispills)

      val dspill = dists(a)(?,ispills)

      for (i <- 0 until spills){                                               // Calculate improvements over vd for new candidate centers

        var j = 0  
        val ii = ispills(0,i)

        while (j < nsamps) {
          if(dspill(j,i) < vd(ii) & dspill(j,i) > threshold(i)) {
            deltas(j) += dspill(j,i) - vd(ii)
          }
          j += 1
        } 
      }

    }

    deltas
  }
  
  
  def sortgen(dd:FMat):(FMat,IMat) = {      // Sorts the COLUMNS in ascending order...


    if (Mat.hasCUDA <= 0) {  // until GPUsort fixed
      var (smat, imat) = sort2(dd,1)


      if(ncache < nsamps){
        //smat = smat(0 until ncache-1,?)
        //imat = imat(0 until ncache-1,?)
        smat = smat(0 until ncache,?)
        imat = imat(0 until ncache,?)
      }

      (smat, imat)

    } else {

      var smat = dd.copy
      var imat = icol(0->nsamps)*iones(1,nsamps)

      GMat.sortGPU(smat, imat)

      if(ncache < nsamps){
        smat = smat(0 until ncache,?)
        imat = imat(0 until ncache,?)
        //smat = smat(0 until ncache-1,?)
        //imat = imat(0 until ncache-1,?)
      }

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
    val (ds, iss) = sortgen(dd)


    

    Mat.nflops += math.round(math.log(size(ds,1))/math.log(2.0))*size(ds,1)*size(ds,2)
    val ft2 = gflop
    println("Sort in %f seconds, %f gcomps" format (ft2._2,ft2._1))
    var bestv:DMat = null
    var besti:IMat = null
    var bestvd = Double.MaxValue 
    flip
    var itry = 0
    
    while(itry < ntrys) {
      
      println("Try %d" format itry)
      val rr = rand(nsamps,1)                                            // Get a random permutation for the centers
      val (rs,irs) = sort2(rr,1)
      val icenters = irs(0->ncenters,0)                                  // Pick centers from the permutation
      val ics = icol(0->nsamps)                            
      //val (vdists, imin) = mindists(ds, iss, ics, icenters)              // Get min distances from points to centers, and best center ids
      mindists(ds, iss, ics, icenters, vdists, imin)              // Get min distances from points to centers, and best center ids
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
    println("Optimum in %f secs, %f gflops, mean dist=%f, verify=%f\n maxdepth=%d, nspills=%d, ncache=%d\nTotal time %f seconds" format 
        (t3._2, t3._1, bestvd, mean(DMat(vdists2),1).v, maxdepth, nspills, ncache, t3._2+ft2._2+ft1._2))

    val ics = icol(0->nsamps)
    flip
    //sil= medoidSilhouette(ds, iss, ics, bestc)
    sil= silhouetteScore(ds, iss, ics, bestc,imin)
    val t4=gflop
    mss = mean(sil,2)(0,0)

    println("Mean Silhouette Score (MSS) %f \n Elapsed time %f secs" format(mss, t4._1 ))

  }
  
}

object PAMmodel { 

  
  class Options { 
    var ncenters = 1000
    var maxpasses = 10
    var ntrys = 1
    var metric:Distance = null
    var verb = false
    var cbal = 10

  }
  
  def runit(nsamps:Int, nfeats:Int, ncenters:Int,metric:String) = {
    println("Generating dataset")
    val c = rand(ncenters, nfeats)
    val a = rand(nsamps, nfeats)*0.3f
    for (i <- 0 until nsamps by ncenters) {val il = math.min(i+ncenters, nsamps); a(i->il,?) += c(0->(il-i),?)}
    val cc = new PAMmodel
    cc.options.ncenters = ncenters
    cc.options.metric = metric match {
    case "euclid" => new euclidDistance
    case "cosangle" => new cosangleDistance
    case "corr" => new correlationDistance
    }
    cc.init(a)
    cc.run
  }
  
  def main(args:Array[String]) = {
    Mat.checkCUDA
    val nsamps= args(0).toInt
    val nfeats = args(1).toInt
    val ncenters = args(2).toInt
    val metric = args(3)
    runit(nsamps, nfeats, ncenters, metric)
  }
}

