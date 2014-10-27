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
  var batchSizeTr = 1
  var batchSizeTe = 1
  var niter=100
  var nsampls = 10
  var nsamplsTr=10
  var nsamplsTe=10
  
  var llikelihood=0f
  
  var predprobs: FMat = null
  
  var nodepath = "C:/data/zp_dlm_FT_code_and_data/node.txt"
  var dagpath = "C:/data/zp_dlm_FT_code_and_data/dag.txt"
  var cptpath = "C:/data/zp_dlm_FT_code_and_data/cpt.fmat"
  var datapath = "C:/data/zp_dlm_FT_code_and_data/dl-10000-0.100000.smat"
  //var datapath = "C:/data/zp_dlm_FT_code_and_data/train4.smat"
  
  def run(alpha0: Float, beta0: Float, batchSize0: Int, batchSize1: Int, niter0: Int, nsampls0: Int, nsampls1: Int) = {
    alpha = alpha0
    beta = beta0
    batchSizeTr = batchSize0
    batchSizeTe = batchSize1
    niter = niter0
    nsamplsTr = nsampls0
    nsamplsTe = nsampls1
    println("alpha: %f, beta: %f, batchSize: %d, nsampls: %d" format (alpha, beta, batchSizeTr, nsamplsTr))
    init
    setup
    synthesize(4300, 0.03f, "dl")
    loadData

    sampleAll

    //testAll
  }
  
  def main(args: Array[String]){
    //run(0.05f, 0.1f, 100, 1, 40, 10, 50)
    //run(1f, 0.1f, 3600, 1, 50, 1, 50)
    //run(1f, 1f, 3600, 1, 50, 1, 50)
    
    //run(1f, 0.1f, 3600, 1, 50, 5, 50)
    //run(1f, 1f, 3600, 1, 50, 5, 50)
    
    //run(1f, 0.1f, 3600, 1, 50, 10, 50)
    //run(1f, 1f, 3600, 1, 50, 10, 50)
    
    //run(0.05f, 0.1f, 200, 1, 50, 1, 50)
    //run(0.05f, 0.1f, 200, 1, 50, 5, 50)

    //flip
    //run(1f, 0.1f, 10000, 1, 50, 5, 50)
    //val f1 = gflop
    flip
    run(1f, 0.1f, 10000, 1, 200, 1, 1)
    val f2 = gflop
    println(f2)
   
    //run(1f, 0.1f, 10000, 1, 20, 20, 50)
    //run(0.05f, 0.2f, 1000, 1, 50, 20, 50)
    //run(0.1f, 0.1f, 400, 1, 50, 1, 50)
    //run(0.05f, 0.1f, 400, 1, 40, 5, 50)
    //run(0.1f, 0.1f, 900, 1, 40, 10, 50)
    
    //run(0.05f, 0.1f, 200, 1, 60, 1, 50)
    //run(0.05f, 0.1f, 200, 1, 40, 5, 50)
    //run(0.05f, 0.1f, 200, 1, 40, 10, 50)
    //run(0.1f, 0.1f, 400, 1, 60, 1, 50)
    //run(0.05f, 0.1f, 400, 1, 40, 5, 50)
    //run(0.1f, 0.1f, 900, 1, 40, 10, 50)
    
  }
  
  def init = {
    nodeMap = loadNodeMap(nodepath)
    val n = nodeMap.size
    val dag = loadDag(dagpath, n)
    saveSMat("C:/data/zp_dlm_FT_code_and_data/dag.smat",dag)
    graph = new Graph(dag, n)
    //var data = loadSdata("C:/data/zp_dlm_FT_code_and_data/sdata_cleaned.txt")
    //val data = loadIMat("C:/data/zp_dlm_FT_code_and_data/sdata_test.txt")
    
  }
  
  def setup = {
    // coloring the graph
    graph.color
    
    // prepare cpt 
    val np = sum(graph.dag)
    val ns = IMat(pow(2*ones(1, graph.n), np+1))
    //setseed(1003)
    val lcpt = sum(ns).v
    cpt = rand(lcpt, 1)
    cpt(1 until lcpt by 2) = 1 - cpt(0 until lcpt by 2)
    //saveFMat("C:/data/zp_dlm_FT_code_and_data/cptinit.fmat", cpt)
    cpt0 = loadFMat(cptpath)
    
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
    
    //saveSMat("C:/data/zp_dlm_FT_code_and_data/debug/mat11.smat", iproject)
    //saveSMat("C:/data/zp_dlm_FT_code_and_data/debug/mat12.smat", pproject)
    
    // load synthetic data
    //
    predprobs = rand(graph.n, batchSizeTr)
  }
  
  def loadData = {
    var data = loadSMat(datapath)
    //val perm = randperm(4300)
    //data = data(?, perm)
    //val cutoff = 4000
    //sdata = data(?, 0 until 3600)
    //val perm2 = randperm(4000)
    //tdata = data(?, 3600 until 4300)
    //tdata = data(?, 0 until cutoff)
    //tdata = data(?, cutoff until 4300)
    sdata = data.copy
    tdata = data.copy
    //sdata = data.t
    //tdata = data.t
    //val data = loadSMat("C:/data/zp_dlm_FT_code_and_data/sdata.smat")
    //val perm = randperm(5000)
    //data = data(?, perm)
    //val cutoff = 4000
    //sdata = data(?, 0 until cutoff)
    //tdata = data(?, 0 until cutoff)
    //tdata = data(?, cutoff until 5000)
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
  
  def getCpt0(index: IMat) = {
    var cptindex = zeros(index.nr, index.nc)
    for(i <-0 until index.nc){
      cptindex(?, i) = cpt0(index(?, i))
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
    //val p = 0.5
    var sample = rand(nnode, batchSize*nsampls)
    sample = (sample <= p)

    //check the logic of this part, enforce data
    state(ids, ?) = sample
    val innz = find(fdata)
    for(i <- 0 until batchSize*nsampls by batchSize){
    	var c = innz+i
    	state(innz + i * graph.n) = 0
    	state(?, i until i+batchSize) = state(?, i until i+batchSize) + (fdata > 0)
    }

  }
  
  def sampleColor(fdata: FMat, ids: IMat, pids: IMat)={
    

    val nnode = size(ids, 1)
    //val ndata = size(fdata, 2)

    //val a = IMat(cptoffset(pids) + iproject(pids,?)*state0)

    val nodep0 = ln(getCpt(cptoffset(pids) + IMat(iproject(pids,?)*state0)) + 1e-10)
    val nodep1 = ln(getCpt(cptoffset(pids) + IMat(iproject(pids,?)*state1)) + 1e-10)
        
    val p0 = exp(pproject(ids, pids) * nodep0)
    val p1 = exp(pproject(ids, pids) * nodep1)
    
    val p = p1/(p0+p1)
    //println(p)
    //val p = 0.5
    var sample = rand(nnode, batchSize*nsampls)
    sample = (sample <= p)

    //check the logic of this part, enforce data
    state(ids, ?) = sample
    val innz = find(fdata)
    for(i <- 0 until batchSize*nsampls by batchSize){
    	var c = innz+i
    	state(innz + i * graph.n) = 0
    	state(?, i until i+batchSize) = state(?, i until i+batchSize) + (fdata > 0)
    }

    predprobs(ids, ?) = p

  }
  
  def sample(data: SMat, k: Int) = {
    val fdata = full(data)
    if(k==0){
    	initState(fdata)
    }
    for(i <- 0 until 1){
    for(c <- 0 until graph.ncolors){
      val ids = find(graph.colors == c)
      val pids = find( sum(pproject(ids, ?), 1))
      initStateColor(fdata, ids)
      sampleColor(fdata, ids, pids)

    }
    }
  }
  
  def sampleAll = {
    val ndata = size(sdata, 2)

    for(k <- 0 until niter){
    var j = 0;
    
   // println("iteration %d" format k)
    //testAll
    batchSize = batchSizeTr
    nsampls = nsamplsTr
    for(i <- 0 until ndata by batchSize){
      sample(sdata(?, i until math.min(ndata, i+batchSize)), k)
      updateCpt      
      eval
      
    }
    println("delta: %f, %f, %f, %f" format (eval2._1, eval2._2, eval2._3, llikelihood))
    //pred

    //println("ll: %f" format llikelihood)
    llikelihood = 0f
    //println("dist cpt - cpt0: %f" format ((cpt-cpt0) dot (cpt-cpt0)).v )
  

    }

  }
  
   def pred = {
    val ndata = size(sdata, 1)
    for(j <- 0 until ndata by batchSizeTr){         
    	   //val jend = math.min(ndata, j + opts.batchSize)
    	   //val minidata = data(j until jend, ?)
         val minidata = sdata
    	   // sample mini-batch of data
    	 sample(minidata, 0)
    	   // update parameters
    }
    val vdatat = loadSMat("C:/data/zp_dlm_FT_code_and_data/train4.smat")
    val vdata = vdatat.t
    val (r, c, v) = find3(vdata)
    var correct = 0f
    var tot = 0f
    for(i <- 0 until vdata.nnz){
      val ri = r(i).toInt
      val ci = c(i).toInt
      if(predprobs(ri, ci).v != 0.5){
      val pred = ( predprobs(ri, ci).v >= 0.5)
      val targ = (v(i).v >= 0)
      //println(probs(ri, ci).v, v(i).v)
      //println(pred, targ)
      if(pred == targ){
        correct = correct + 1
      }
      tot = tot + 1
      }
    }
    println(vdata.nnz, tot, correct)
   
    println("prediction accuracy: " + correct/tot)
  }
  
  def testAll = {
    val ndata = size(tdata, 2)  
    var n = 0f;
    var nc = 0f;
    batchSize = batchSizeTe
    nsampls = nsamplsTe
    //setseed(123456)
    for(i <- 0 until ndata ){
      var test:SMat = tdata(?, i)
      var (ids: IMat, d, v) = find3(test)
      if(ids.length>=2){

      var targeti = ids((rand(1,1)*ids.length).v.toInt)
      var target = full(test)(targeti).toInt
      
      var r0 = izeros(ids.length-1, 1)
      var v0 = zeros(ids.length-1, 1)
      var ptr = 0
      
      for(j <-0 until ids.length){
        if(targeti != ids(j)){
          r0(ptr) = ids(j)
          v0(ptr) = v(j)
          ptr += 1
        }
      }
      var test0= sparse(r0, izeros(ids.length-1, 1) ,v0, 334, 1)
      sample(test0, 0)
      var pred = if(sum(state(targeti, ?)).v/state(targeti, ?).length >= 0.5) 1 else -1
      n = n+1
      //println(i, pred, target, pred==target)
      if(pred == target) nc = nc + 1
      }
    }
    println("Test accuracy: %f" format nc.toFloat/n)

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
    
    cpt_old = counts.copy
	cpt = (1-alpha) * cpt + alpha * counts
    
  }
  
  def synthesize(n: Int, sp: Float, name: String) = {
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
    		 val p0 = cpt0(ind)
    		 val p1 = cpt0(ind + 1)
    		 val p = p1/(p0+p1)
    		 var r = rand(1,1)
    		 r = ( r <= p )	- (r > p)	 
    		 syndata(i, k) = r.v.toInt
    	}
    }
    saveFMat("C:/data/zp_dlm_FT_code_and_data/%s-%d-%f.fmat" format (name, n, sp), syndata)
    
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
    saveSMat("C:/data/zp_dlm_FT_code_and_data/%s-%d-%f.smat" format (name, n, sp), smat)
  }
  
  def eval = {
	val index = IMat(cptoffset + iproject*(state>0))
	val ll = sum(sum ( ln(getCpt(index))))/nsampls
    llikelihood += ll.v
  }
  
  def eval2 = {
    (sqrt(((cpt - cpt0) dot (cpt - cpt0))/((cpt0) dot (cpt0))).v, sqrt(((cpt_old - cpt0) dot (cpt_old - cpt0))/((cpt0) dot (cpt0))).v, sqrt(((cpt_old - cpt) dot (cpt_old - cpt))/cpt.length).v)
  }
  

  def showCpt(node: String){
    val id = nodeMap(node)
    val np = sum(graph.dag(?, id)).v
    val offset = cptoffset(id)
    
    for(i <-0 until math.pow(2, np).toInt){
    	if(np > 0)
    		print(String.format("\t%" + np.toInt + "s", i.toBinaryString).replace(" ", "0"))
    }
    print("\n0")
    for(i <-0 until math.pow(2, np).toInt)
    	print("\t%.2f" format cpt(offset + i*2))
    print("\n1")
    for(i <-0 until math.pow(2, np).toInt)
    	print("\t%.2f" format cpt(offset + i*2+1))
    print("\n")
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