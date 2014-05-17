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
  var niter=100
  var nsampls=10
  
  var llikelihood=0f

  
  def init = {
    nodeMap = loadNodeMap("C:/data/zp_dlm_FT_code_and_data/node.txt")
    val n = nodeMap.size
    val dag = loadDag("C:/data/zp_dlm_FT_code_and_data/dag.txt", n)
    graph = new Graph(dag, n)
    sdata = loadSdata("C:/data/zp_dlm_FT_code_and_data/sdata_cleaned.txt")
    ///val data = loadIMat("C:/data/zp_dlm_FT_code_and_data/sdata_test.txt")
    //sdata = sparse(data(?,0),data(?,1),data(?,2), n, size(data,1))
    sdata = sdata(?, 0 until 4000)
    //sdata = loadSMat("C:/data/zp_dlm_FT_code_and_data/sdata.smat")
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
    cpt0 = loadFMat("C:/data/zp_dlm_FT_code_and_data/cpt_test.txt")
    
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
    val dag = graph.dag
    iproject = dag.t
    
       
    for(i <- 0 until graph.n){
      val ps = find(dag(?, i))
      val np = ps.length
      
      for(j <-0 until np){
        iproject(i, ps(j)) = math.pow(2, np-j).toFloat
      }
      //pmat(i, ps) = pow(2, IMat(np to 1 by -1))
    }
    
    iproject = iproject + sparse(IMat(0 until graph.n), IMat(0 until graph.n), ones(1, graph.n))
    pproject = dag + sparse(IMat(0 until graph.n), IMat(0 until graph.n), ones(1, graph.n))
    
    /*
    println("iproject")
    var (r,c,v) = find3(iproject)
    for(i <-0 until iproject.nnz){
      println(r(i),c(i),v(i))
    }
    
    println("pproject")
    var (rp,cp,vp) = find3(pproject)
    for(i <-0 until pproject.nnz){
      println(rp(i),cp(i),vp(i))
    }
    */
    // load synthetic data
    //
  }
  
  def initState(fdata: FMat) = {
    val ndata = size(fdata, 2)
    state = rand(graph.n, batchSize*nsampls)
    state = (state >= 0.5)
    //state = (state >= 0.5) - (state < 0.5)
    val innz = find(fdata)
    for(i <- 0 until batchSize*nsampls by batchSize){
    	state(innz + i * graph.n) = 0
    	state(?, i until i+batchSize) = state(?, i until i+batchSize) + (fdata > 0)
    }

  }
  
  def initStateColor(fdata: FMat, ids: IMat) = {
    state0 = state.copy
    state1 = state.copy
    
    state0(ids, ?) = 0
    state1(ids, ?) = 1
    
    val innz = find(fdata)
    for(i <- 0 until batchSize*nsampls by batchSize){
    	state0(innz + i * graph.n) = 0
    	state0(?, i until i+batchSize) = state0(?, i until i+batchSize) + (fdata > 0)
    	state1(innz + i * graph.n) = 0
    	state1(?, i until i+batchSize) = state1(?, i until i+batchSize) + (fdata > 0)
    }

  }
  
  def getCpt(index: IMat) = {
    var cptindex = zeros(index.nr, index.nc)
    for(i <-0 until index.nc){
      cptindex(?, i) = cpt(index(?, i))
    }
    cptindex
  }
  
  def sampleColor(fdata: FMat, ids: IMat)={
    

    val nnode = size(ids, 1)
    //val ndata = size(fdata, 2)    
    
    val nodep0 = ln(getCpt(cptoffset + IMat(iproject*state0)) + 1e-10)
    val nodep1 = ln(getCpt(cptoffset + IMat(iproject*state1)) + 1e-10)
        
    val p0 = exp(pproject(ids, ?) * nodep0)
    val p1 = exp(pproject(ids, ?) * nodep1)
    
    val p = p1/(p0+p1)
    
    var sample = rand(nnode, batchSize*nsampls)
    sample = (sample <= p)
    
    state(ids, ?) = sample
    val innz = find(fdata)
    for(i <- 0 until batchSize*nsampls by batchSize){
    	var c = innz+i
    	state(innz + i * graph.n) = 0
    	state(?, i until i+batchSize) = state(?, i until i+batchSize) + (fdata > 0)
    }

  }
  
  def sample(data: SMat) = {
    val fdata = full(data)
    initState(fdata)
    for(i <- 0 until 8){
    for(c <- 0 until graph.ncolors){
      val ids = find(graph.colors == c)
      initStateColor(fdata, ids)
      sampleColor(fdata, ids)
    }
    }
  }
  
  def sampleAll = {
    val ndata = size(sdata, 2)
    
    for(k <- 0 until niter){
    var j = 0;
    
    for(i <- 0 until ndata by batchSize){
      sample(sdata(?, i until math.min(ndata, i+batchSize)))
      updateCpt
      eval
    }
    
    println("ll: %f" format llikelihood)
    llikelihood = 0f
    //println("dist cpt - cpt0: %f" format ((cpt-cpt0) dot (cpt-cpt0)).v )

    }

  }
  
  def updateCpt = {
    val nstate = size(state, 2)
	val index = IMat(cptoffset + iproject*(state>0))
	var counts = zeros(cpt.length, 1)
	for(i <- 0 until nstate){
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
    //for(i<-0 until cpt.length){
    //  print("%f," format cpt(i))
    //}
    //println("")
	
  }
  
  def synthesize(n: Int, sp: Float) = {
    var syndata = izeros(graph.n, n)
    for(k <-0 until n){
    	for(i <- 0 until graph.n){
    		 val ps = find(graph.dag(?, i))
    		 val np = ps.length
    		 var ind = cptoffset(i).v
    		 for(j <- 0 until np){
    		   var pv = if(syndata(ps(j),k) > 0) 1 else 0
    		   ind = ind +  math.pow(2f, np-j).toInt * pv
    		 }
    		 val p0 = cpt(ind)
    		 val p1 = cpt(ind + 1)
    		 val p = p1/(p0+p1)
    		 if(i==4) println(p)
    		 var r = rand(1,1)
    		 r = ( r <= p )	- (r > p)	 
    		 syndata(i, k) = r.v.toInt
    	}
    }
    saveFMat("C:/data/zp_dlm_FT_code_and_data/sdata.smat",syndata)
    
    var l = (n * graph.n * sp * 1.5).toInt
    var r = izeros(l ,1)
    var c = izeros(l ,1)
    var v = zeros(l ,1)
    var ptr = 0
    for(k <-0 until n){
      val rd = (rand(graph.n, 1) < sp)
      val ids = find(rd==1)
      val vs = syndata(ids, k)
      val len = ids.length
      if(len > 0){
      r(ptr until ptr + len,0) = ids
      v(ptr until ptr + len,0) = vs
      c(ptr until ptr + len,0) = k * iones(len, 1)
      ptr = ptr + len
      }
    }
    val smat = sparse(r(0 until ptr), c(0 until ptr), v(0 until ptr), graph.n, n)
    saveSMat("C:/data/zp_dlm_FT_code_and_data/sdata.smat",smat)
  }
  
  def eval = {
	val index = IMat(cptoffset + iproject*(state>0))
	val ll = sum(sum ( ln(getCpt(index))))/nsampls
    llikelihood += ll.v
  }
  
  def run(alpha0: Float, beta0: Float, batchSize0: Int, niter0: Int, nsampls0: Int) = {
    alpha = alpha0
    beta = beta0
    batchSize = batchSize0
    niter = niter0
    nsampls = nsampls0
    
    init
    setup
    //synthesize(5000, 0.3f)
    sampleAll
  }
  
  def main(args: Array[String]){
    run(0.2f, 1f, 500, 10, 100)
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