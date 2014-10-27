package BayesNet

import BIDMat.{ CMat, CSMat, DMat, Dict, IDict, FMat, GMat, GIMat, GSMat, HMat, IMat, Mat, SMat, SBMat, SDMat }
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import java.io._

object BayesNetMooc {

  var nodeMap: scala.collection.mutable.HashMap[String, Int] = null
  //var dag: FMat = null
  var graph: Graph = null
  var sdata: SMat = null
  var tdata: SMat = null
  // 
  var cpt: FMat = null
  var cpt_old: FMat = null
  var cpt0: FMat = null
  var cptoffset: IMat = null

  // projection matrix
  var iproject: SMat = null
  var pproject: SMat = null

  var state: FMat = null
  var state0: FMat = null
  var state1: FMat = null

  var alpha = 1f
  var beta = 0.1f
  var batchSize = 1
  var niter = 100
  var nsampls = 1

  var llikelihood = 0f

  var predprobs: FMat = null

  def main(args: Array[String]) {
    val nodepath = args(0)
    val dagpath = args(1)
    val datapath = args(2)
    val nq = args(3).toInt
    val ns = args(4).toInt
    batchSize = ns
    niter = args(5).toInt
    init(nodepath, dagpath)
    setup
    loadData(datapath, nq, ns)   
    sampleAll
  }

  def init(nodepath: String, dagpath: String) = {
    nodeMap = loadNodeMap(nodepath)
    val n = nodeMap.size
    val dag = loadDag(dagpath, n)
    graph = new Graph(dag, n)


  }

  def setup = {
    // coloring the graph
    graph.color

    // prepare cpt 
    val np = sum(graph.dag)
    val ns = IMat(pow(2 * ones(1, graph.n), np + 1))
    //setseed(1003)
    val lcpt = sum(ns).v
    cpt = rand(lcpt, 1)
    cpt(1 until lcpt by 2) = 1 - cpt(0 until lcpt by 2)
    // prepare cpt offset 
    cptoffset = izeros(graph.n, 1)
    cptoffset(1 until graph.n) = cumsum(ns)(0 until graph.n - 1)

    // prepare projection matrix
    val dag = graph.dag
    iproject = dag.t

    for (i <- 0 until graph.n) {
      val ps = find(dag(?, i))
      val np = ps.length

      for (j <- 0 until np) {
        iproject(i, ps(j)) = math.pow(2, np - j).toFloat
      }
    }

    iproject = iproject + sparse(IMat(0 until graph.n), IMat(0 until graph.n), ones(1, graph.n))
    pproject = dag + sparse(IMat(0 until graph.n), IMat(0 until graph.n), ones(1, graph.n))

    // load synthetic data
    //
    predprobs = rand(graph.n, batchSize)
  }

  def loadData(datapath:String, nq: Int, ns: Int) = {
    sdata = loadSData(datapath, nq, ns)
    tdata = loadTData(datapath, nq, ns)
  }

  def initState(fdata: FMat) = {

    val ndata = size(fdata, 2)
    state = rand(graph.n, batchSize * nsampls)
    state = (state >= 0.5)
    //state = (state >= 0.5) - (state < 0.5)
    val innz = find(fdata)
    for (i <- 0 until batchSize * nsampls by batchSize) {
      state(innz + i * graph.n) = 0
      state(?, i until i + batchSize) = state(?, i until i + batchSize) + (fdata > 0)
    }

  }

  def initStateColor(fdata: FMat, ids: IMat) = {
    state0 = state.copy
    state1 = state.copy

    state0(ids, ?) = 0
    state1(ids, ?) = 1

    val innz = find(fdata)
    for (i <- 0 until batchSize * nsampls by batchSize) {
      state0(innz + i * graph.n) = 0
      state0(?, i until i + batchSize) = state0(?, i until i + batchSize) + (fdata > 0)
      state1(innz + i * graph.n) = 0
      state1(?, i until i + batchSize) = state1(?, i until i + batchSize) + (fdata > 0)
    }

  }

  def getCpt(index: IMat) = {
    var cptindex = zeros(index.nr, index.nc)
    for (i <- 0 until index.nc) {
      cptindex(?, i) = cpt(index(?, i))
    }
    cptindex
  }

  def getCpt0(index: IMat) = {
    var cptindex = zeros(index.nr, index.nc)
    for (i <- 0 until index.nc) {
      cptindex(?, i) = cpt0(index(?, i))
    }
    cptindex
  }

  def sampleColor(fdata: FMat, ids: IMat) = {

    val nnode = size(ids, 1)
    //val ndata = size(fdata, 2)    
    val nodep0 = ln(getCpt(cptoffset + IMat(iproject * state0)) + 1e-10)
    val nodep1 = ln(getCpt(cptoffset + IMat(iproject * state1)) + 1e-10)

    val p0 = exp(pproject(ids, ?) * nodep0)
    val p1 = exp(pproject(ids, ?) * nodep1)

    val p = p1 / (p0 + p1)
    //val p = 0.5
    var sample = rand(nnode, batchSize * nsampls)
    sample = (sample <= p)

    //check the logic of this part, enforce data
    state(ids, ?) = sample
    val innz = find(fdata)
    for (i <- 0 until batchSize * nsampls by batchSize) {
      var c = innz + i
      state(innz + i * graph.n) = 0
      state(?, i until i + batchSize) = state(?, i until i + batchSize) + (fdata > 0)
    }

  }

  def sampleColor(fdata: FMat, ids: IMat, pids: IMat) = {

    val nnode = size(ids, 1)
    //val ndata = size(fdata, 2)

    //val a = IMat(cptoffset(pids) + iproject(pids,?)*state0)

    val nodep0 = ln(getCpt(cptoffset(pids) + IMat(iproject(pids, ?) * state0)) + 1e-10)
    val nodep1 = ln(getCpt(cptoffset(pids) + IMat(iproject(pids, ?) * state1)) + 1e-10)

    val p0 = exp(pproject(ids, pids) * nodep0)
    val p1 = exp(pproject(ids, pids) * nodep1)

    val p = p1 / (p0 + p1)
    //println(p)
    //val p = 0.5
    var sample = rand(nnode, batchSize * nsampls)
    sample = (sample <= p)

    //check the logic of this part, enforce data
    state(ids, ?) = sample
    val innz = find(fdata)
    for (i <- 0 until batchSize * nsampls by batchSize) {
      var c = innz + i
      state(innz + i * graph.n) = 0
      state(?, i until i + batchSize) = state(?, i until i + batchSize) + (fdata > 0)
    }

    predprobs(ids, ?) = p

  }

  def sample(data: SMat, k: Int) = {
    val fdata = full(data)
    if (k == 0) {
      initState(fdata)
    }
    for (i <- 0 until 1) {
      for (c <- 0 until graph.ncolors) {
        val ids = find(graph.colors == c)
        val pids = find(sum(pproject(ids, ?), 1))
        initStateColor(fdata, ids)
        sampleColor(fdata, ids, pids)

      }
    }
  }

  def sampleAll = {
    val ndata = size(sdata, 2)

    for (k <- 0 until niter) {
      var j = 0;

      // println("iteration %d" format k)
      //testAll
      batchSize = batchSize
      nsampls = nsampls
      for (i <- 0 until ndata by batchSize) {
        sample(sdata(?, i until math.min(ndata, i + batchSize)), k)
        updateCpt
        eval

      }
      //println("delta: %f, %f, %f, %f" format (eval2._1, eval2._2, eval2._3, llikelihood))
      pred(k)

      //println("ll: %f" format llikelihood)
      llikelihood = 0f
      //println("dist cpt - cpt0: %f" format ((cpt-cpt0) dot (cpt-cpt0)).v )

    }

  }

  def pred(i: Int) = {
    val ndata = size(sdata, 1)
    for (j <- 0 until ndata by batchSize) {
      //val jend = math.min(ndata, j + opts.batchSize)
      //val minidata = data(j until jend, ?)
      val minidata = sdata
      // sample mini-batch of data
      sample(minidata, 0)
      // update parameters
    }
    //val vdatat = loadSMat("C:/data/zp_dlm_FT_code_and_data/train4.smat")
    //val vdata = vdatat.t
    val (r, c, v) = find3(tdata)
    var correct = 0f
    var tot = 0f
    for (i <- 0 until tdata.nnz) {
      val ri = r(i).toInt
      val ci = c(i).toInt
      if (predprobs(ri, ci).v != 0.5) {
        val pred = (predprobs(ri, ci).v >= 0.5)
        val targ = (v(i).v >= 0)
        //println(probs(ri, ci).v, v(i).v)
        //println(pred, targ)
        if (pred == targ) {
          correct = correct + 1
        }
        tot = tot + 1
      }
    }
    //println(sdata.nnz, tot, correct)

    println(i + "," + correct / tot)
  }


  def updateCpt = {
    val nstate = size(state, 2)
    val index = IMat(cptoffset + iproject * (state > 0))
    var counts = zeros(cpt.length, 1)
    for (i <- 0 until nstate) {
      counts(index(?, i)) = counts(index(?, i)) + 1
    }
    counts = counts + beta

    // normalize count matrix
    var normcounts = zeros(counts.length, 1)
    normcounts(0 until counts.length - 1 by 2) = counts(1 until counts.length by 2)
    normcounts(1 until counts.length by 2) = counts(0 until counts.length - 1 by 2)
    normcounts = normcounts + counts
    counts = counts / normcounts

    cpt_old = counts.copy
    cpt = (1 - alpha) * cpt + alpha * counts

  }

  def eval = {
    val index = IMat(cptoffset + iproject * (state > 0))
    val ll = sum(sum(ln(getCpt(index)))) / nsampls
    llikelihood += ll.v
  }


  def showCpt(node: String) {
    val id = nodeMap(node)
    val np = sum(graph.dag(?, id)).v
    val offset = cptoffset(id)

    for (i <- 0 until math.pow(2, np).toInt) {
      if (np > 0)
        print(String.format("\t%" + np.toInt + "s", i.toBinaryString).replace(" ", "0"))
    }
    print("\n0")
    for (i <- 0 until math.pow(2, np).toInt)
      print("\t%.2f" format cpt(offset + i * 2))
    print("\n1")
    for (i <- 0 until math.pow(2, np).toInt)
      print("\t%.2f" format cpt(offset + i * 2 + 1))
    print("\n")
  }

  def loadNodeMap(path: String) = {
    var nodeMap = new scala.collection.mutable.HashMap[String, Int]()
    var lines = scala.io.Source.fromFile(path).getLines

    for (l <- lines) {
      var t = l.split(",")
      nodeMap += (t(0) -> (t(1).toInt - 1))
    }
    nodeMap
  }

  def loadDag(path: String, n: Int) = {
    //var dag = zeros(n, n)
    val bufsize = 100000
    var row = izeros(bufsize, 1)
    var col = izeros(bufsize, 1)
    var v = zeros(bufsize, 1)
    var ptr = 0

    var lines = scala.io.Source.fromFile(path).getLines

    for (l <- lines) {
      if (ptr % bufsize == 0 && ptr > 0) {
        row = row on izeros(bufsize, 1)
        col = col on izeros(bufsize, 1)
        v = v on zeros(bufsize, 1)
      }
      var t = l.split(",")
      row(ptr) = nodeMap(t(0))
      col(ptr) = nodeMap(t(1))
      v(ptr) = 1f
      ptr = ptr + 1
    }
    /*
    for(i<-0 until n){
      if(ptr % bufsize == 0 && ptr > 0){
        row = row on izeros(bufsize, 1)
        col = col on izeros(bufsize, 1)
        v = v on zeros(bufsize, 1)
      }
      row(ptr) = i
      col(ptr) = i
      v(ptr) = 0
      ptr = ptr + 1
    }*/
    sparse(row(0 until ptr), col(0 until ptr), v(0 until ptr), n, n)
  }

  def loadSData(path: String, nq: Int, ns: Int) = {
    var lines = scala.io.Source.fromFile(path).getLines

    var sMap = new scala.collection.mutable.HashMap[String, Int]()

    val bufsize = 100000
    var row = izeros(bufsize, 1)
    var col = izeros(bufsize, 1)
    var v = zeros(bufsize, 1)

    var ptr = 0
    var sid = 0
    /*
    val writer: PrintWriter = new PrintWriter(path + "/../sdata_new.txt")
    for(l <- lines) {
      val r = math.random
      if(r<0.9){
    	  writer.println(l + ",1")}
      else{
    	  writer.println(l + ",0")
      }
    }
    writer.close
    */
    for (l <- lines) {

      if (ptr % bufsize == 0 && ptr > 0) {
        row = row on izeros(bufsize, 1)
        col = col on izeros(bufsize, 1)
        v = v on zeros(bufsize, 1)
      }

      var t = l.split(",")
      val shash = t(0)

      if (!(sMap contains shash)) {
        sMap += (shash -> sid)
        sid = sid + 1
      }
      if (t(5) == "1") {
        row(ptr) = sMap(shash)
        col(ptr) = nodeMap("I" + t(2))
        v(ptr) = (t(3).toFloat - 0.5) * 2
        ptr = ptr + 1
      }
    }

    var s = sparse(col(0 until ptr), row(0 until ptr), v(0 until ptr), nq, ns)
    //s
    (s > 0) - (s < 0)
  }

  def loadTData(path: String, nq: Int, ns: Int) = {
    var lines = scala.io.Source.fromFile(path).getLines

    var sMap = new scala.collection.mutable.HashMap[String, Int]()

    val bufsize = 100000
    var row = izeros(bufsize, 1)
    var col = izeros(bufsize, 1)
    var v = zeros(bufsize, 1)

    var ptr = 0
    var sid = 0

    for (l <- lines) {

      if (ptr % bufsize == 0 && ptr > 0) {
        row = row on izeros(bufsize, 1)
        col = col on izeros(bufsize, 1)
        v = v on zeros(bufsize, 1)
      }

      var t = l.split(",")
      val shash = t(0)

      if (!(sMap contains shash)) {
        sMap += (shash -> sid)
        sid = sid + 1
      }
      if (t(5) == "0") {
        row(ptr) = sMap(shash)
        col(ptr) = nodeMap("I" + t(2))
        v(ptr) = (t(3).toFloat - 0.5) * 2
        ptr = ptr + 1
      }
    }

    var s = sparse(col(0 until ptr), row(0 until ptr), v(0 until ptr), nq, ns)
    //s
    (s > 0) - (s < 0)
  }
}