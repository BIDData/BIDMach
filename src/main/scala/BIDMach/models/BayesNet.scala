package BayesNet

import BIDMat.{CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SBMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._

object BayesNet {
  
  var nodeMap:scala.collection.mutable.HashMap[String, Int] = null
  //var dag: FMat = null
  var graph: Graph = null
  var sdata: SMat = null
  // 
  var cpt: FMat = null
  var cptoffset: IMat = null
  // projection matrix
  var pmat: SMat = null
  
  var state: FMat = null
  var state0: FMat = null
  var state1: FMat = null
  
  var alpha = 1f
  var beta = 0.1f
  var batchSize = 1
  var niter=100

  
  def init = {
    nodeMap = loadNodeMap("C:/data/zp_dlm_FT_code_and_data/node.txt")
    val n = nodeMap.size
    val dag = loadDag("C:/data/zp_dlm_FT_code_and_data/dag.txt", n)
    graph = new Graph(dag, n)
    sdata = loadSdata("C:/data/zp_dlm_FT_code_and_data/sdata_cleaned.txt")
  }
  
  def setup = {
    // coloring the graph
    graph.color
    
    // prepare cpt 
    val np = sum(graph.dag)
    val ns = IMat(pow(2*ones(1, graph.n), np+1))

    val lcpt = sum(ns).v
    cpt = rand(lcpt, 1)
    cpt(1 until lcpt by 2) = 1 - cpt(0 until lcpt by 2)
    
    /*
    for(i <- 0 until cpt.length-1 by 2 ){
      if(cpt(i)> cpt(i+1)){
        var temp = cpt(i)
        cpt(i) = cpt(i+1)
        cpt(i+1) = temp
      }
    }*/
    
    // prepare cpt offset 
    cptoffset = izeros(graph.n, 1)
    cptoffset(1 until graph.n) = cumsum(ns)(0 until graph.n-1)
    
    // prepare projection matrix
    pmat = graph.dag.t
    
    //println("dag")
    //println(graph.dag(?, 0))
    //println("pmat")
    //println(pmat(0, ?))
    //println(sum(sum(pmat.t != graph.dag)))
    println("pmat nnz: %d" format pmat.nnz)
    println("dag nnz: %d" format graph.dag.nnz)
    
    //saveSMat("C:/data/zp_dlm_FT_code_and_data/dag.smat", graph.dag)
   
    for(i <- 0 until graph.n){
      //val ps0 = find(pmat(i, ?))
      val ps = find(graph.dag(?, i))
      val np = ps.length
      
      for(j <-0 until np){
        pmat(i, ps(j)) = math.pow(2, np-j).toFloat
      }
      //pmat(i, ps) = pow(2, IMat(np to 1 by -1))
    }
    
    pmat = pmat + sparse(IMat(0 until graph.n), IMat(0 until graph.n), ones(1, graph.n))
  }
  
  def initState(fdata: FMat) = {
    val ndata = size(fdata, 2)
    state = rand(graph.n, ndata)
    state = (state >= 0.5) - (state < 0.5)
    
    state(find(fdata)) = 0
    state = state + fdata
  }
  
  def initStateColor(fdata: FMat, ids: IMat) = {
    state0 = state.copy
    state1 = state.copy
    
    state0(ids) = -1
    state1(ids) = 1
    
    state0(find(fdata)) = 0
    state1(find(fdata)) = 0
    
    state0 = ((state0 + fdata) > 0)
    state1 = ((state1 + fdata) > 0)
  }
  
  def getCpt(index: IMat) = {
    var cptindex = zeros(index.nr, index.nc)
    for(i <-0 until index.nc){
      cptindex(?, i) = cpt(index(?, i))
    }
    cptindex
  }
  
  def sampleColor(fdata: FMat, ids: IMat)={
    
    //val ids = find(graph.colors == c)
    val nnode = size(ids, 1)
    val ndata = size(fdata, 2)
    //val fdata = full(data)
    
    // wouldn't work if bsize > 1
    val nodep0 = ln(getCpt(cptoffset + IMat(pmat*state0)) + 1e-10)
    val nodep1 = ln(getCpt(cptoffset + IMat(pmat*state1)) + 1e-10)
    
    val summat = graph.dag + sparse(IMat(0 until graph.n), IMat(0 until graph.n), ones(1, graph.n))
    
    val p0 = exp(summat(ids, ?) * nodep0)
    val p1 = exp(summat(ids, ?) * nodep1)
    
    val p = p1/(p0+p1)
    
    var sample = rand(nnode, ndata)
    sample = (sample > p) - (sample < p)
    
    state(ids, ?) = sample
    state(find(fdata)) = 0
    state = state + fdata
  }
  
  def sample(data: SMat) = {
    val fdata = full(data)
    initState(fdata)
    for(c <- 0 until graph.ncolors){
      val ids = find(graph.colors == c)
      initStateColor(fdata, ids)
      sampleColor(fdata, ids)
    }
  }
  
  def sampleAll = {
    val ndata = size(sdata, 2)

    
    for(k <- 0 until niter){
    var j = 0;
    for(i <- 0 until ndata by batchSize){
      sample(sdata(?, i until math.min(ndata, i+batchSize)))
      updateCpt
      val ll = eval.v
      if(j%1000==0)
    	  println("ll: %f" format ll)
      j = j + 1
    }

    }

  }
  
  def updateCpt = {
    val nstate = size(state, 2)
	val index = IMat(cptoffset + pmat*(state>0))
	var counts = zeros(cpt.length, 1)
	for(i <- 1 until nstate){
	  counts(index(?, i)) = counts(index(?, i)) + 1
	}
    counts = counts + beta
    	
	// normalize count matrix
	var normcounts = zeros(counts.length, 1)
    normcounts(0 until counts.length-1 by 2) = counts(1 until counts.length by 2) 
    normcounts(1 until counts.length by 2) = counts(0 until counts.length-1 by 2) 
    normcounts = normcounts + counts
    counts = counts / normcounts
    /*
    for(i <- 0 until counts.length-1 by 2 ){
      if(counts(i)> counts(i+1)){
        var temp = counts(i)
        counts(i) = counts(i+1)
        counts(i+1) = temp
      }
    }
    */	
	cpt = (1-alpha) * cpt + alpha * counts
	
  }
  
  def eval = {
	val index = IMat(cptoffset + pmat*(state>0))
	//val summat = graph.dag + sparse(IMat(0 until graph.n), IMat(0 until graph.n), ones(1, graph.n))
	val ll = sum(sum ( ln(getCpt(index))))/size(index, 2)
    ll
  }
  
  def run(alpha0: Float, beta0: Float, batchSize0: Int, niter0: Int) = {
    alpha = alpha0
    beta = beta0
    batchSize = batchSize0
    niter = niter0
    
    init
    setup
    sampleAll
  }
  def showCpt(node: String){
    val id = nodeMap(node)
    val np = sum(graph.dag(?, id)).v
    val offset = cptoffset(id)
    
    for(i <-0 until math.pow(2, np).toInt)
    	print(String.format("\t%" + np.toInt + "s", i.toBinaryString).replace(" ", "0"))
    print("\n0")
    for(i <-0 until math.pow(2, np).toInt)
    	print("\t%.2f" format cpt(offset + i*2))
    print("\n1")
    for(i <-0 until math.pow(2, np).toInt)
    	print("\t%.2f" format cpt(offset + i*2+1))
  }
  
  def loadNodeMap(path: String) = {
    var nodeMap = new scala.collection.mutable.HashMap[String, Int]()
    var lines = scala.io.Source.fromFile(path).getLines

    for(l <- lines){
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

    for(l <- lines){
      if(ptr % bufsize == 0 && ptr > 0){
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
    }
    sparse(row(0 until ptr), col(0 until ptr), v(0 until ptr))
  }
  
  def loadSdata(path: String) = {
    var lines = scala.io.Source.fromFile(path).getLines

    var sMap = new scala.collection.mutable.HashMap[String, Int]()
    
    val bufsize = 100000
    var row = izeros(bufsize, 1)
    var col = izeros(bufsize, 1)
    var v = zeros(bufsize, 1)
    
    var ptr = 0
    var sid = 0
    
    for(l <- lines){
      
      if(ptr % bufsize == 0 && ptr > 0){
        row = row on izeros(bufsize, 1)
        col = col on izeros(bufsize, 1)
        v = v on zeros(bufsize, 1)
      }
      
      var t = l.split(",")
      val shash = t(0)
      
      if(!(sMap contains shash)){
        sMap += (shash -> sid)
        sid = sid + 1
      }
      row(ptr) = sMap(shash)
      col(ptr) = nodeMap("I"+t(2))
      v(ptr) = (t(3).toFloat - 0.5) * 2
      ptr = ptr + 1
    }
    
    var s = sparse(col(0 until ptr), row(0 until ptr), v(0 until ptr))
    //s
    (s>0) - (s<0)
  }

}