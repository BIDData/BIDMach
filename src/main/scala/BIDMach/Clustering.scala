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
  
  def run = {
    flip
    val dd = dists(a)
    val ft1 = gflop
    println("distance in %f seconds, %f gflops" format (ft1._2,ft1._1))
    flip
    val (ds, iss) = sort2(dd,1)
    val rr = rand(nsamps,1)
    val (rs,irs) = sort2(rr,1)
    val icenters = irs(0->ncenters,0)
    val (vdists, imin) = mini2(dd(?,icenters),2)
    val vtmp = vdists.copy
    val itmp = imin.copy
    Mat.nflops += math.round(math.log(size(dd,1))/math.log(2.0))*size(dd,1)*size(dd,2)
    val ft2 = gflop
    println("sorts in %f seconds, %f gcomps" format (ft2._2,ft2._1))
    println("ipass=0, dist=%f" format mean(vdists,1).v)
    flip
    var nchanged = 1
    var ipass = 0
    var totchanged = 0
    while (nchanged > 0 && ipass<options.maxpasses) {
      ipass += 1
      nchanged = 0
      var ipc = 0
      while (ipc < ncenters) {
      	vtmp <-- vdists
      	itmp <-- imin
      	val deltas = zeros(nsamps,1)
        val ifix = find(imin == ipc)
        val tcents = icenters((0->ipc) \ ((ipc+1)->ncenters),0)
        val (vm, im) = mini2(dd(ifix, tcents),2)
        Mat.nflops += length(ifix)*length(tcents)
        im ~ im + (im >= ipc)
        vtmp(ifix,0) = vm
        itmp(ifix,0) = im
        var ipn = 0
        while (ipn < nsamps) {
          var j = 0
          while (j < nsamps && ds(j,ipn) < vtmp(ipn,0)) {
            deltas(iss(j,ipn),0) = deltas(iss(j,ipn),0) + ds(j,ipn) - vtmp(ipn,0)
            j += 1
          } 
          Mat.nflops += 2*j
          ipn += 1
        }
      	val (vs,is) = mini2(deltas)
      	Mat.nflops += 2*nsamps
      	if (vs.v + sum(vtmp).v < sum(vdists).v && is.v != icenters(ipc,0)) {
      	  icenters(ipc,0) = is.v
      	  var j = 0
      	  while (j < nsamps) {
      	    if (dd(j,is.v) < vtmp(j,0)) {
      	      vdists(j,0) = dd(j,is.v) 
      	      imin(j,0) = ipc
      	    } else {
      	      vdists(j,0) = vtmp(j,0)
      	      imin(j,0) = itmp(j,0)      	    }
      	    j += 1
      	  }
      	  nchanged += 1
      	  Mat.nflops += 2*nsamps
      	  if (options.verb) println("ipass=%d, ipc=%d, dist=%f, nchanged=%d" format (ipass, ipc, mean(vdists,1).v, nchanged))
      	}
        ipc += 1
      }
      println("ipass=%d, dist=%f, nchanged=%d" format (ipass, mean(vdists,1).v, nchanged))
      totchanged += nchanged
    }
    val (vdists2, imin2) = mini2(dd(?,icenters),2)
    val t3=gflop
    println("optim in %f secs, %f gsamps, passes=%d, dist=%f, verify=%f, totchanged=%d" format (t3._2, t3._1, ipass, mean(vdists).v, mean(vdists2,1).v, totchanged))
  }
  
}

object PAMmodel { 
  class Options { 
    var ncenters = 1000
    var maxpasses = 5
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
