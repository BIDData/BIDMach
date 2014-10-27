package BayesNet

import BIDMat.{CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SBMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._


// data 1, -1 sparse matrix
class BayesNet2 (val data: SMat, val vdata: SMat, val graph: Graph, val opts: Options, val cpt0: FMat) {

  // in GPU or not?
  var params: FMat = null
  
  var paramsOffset: IMat = null
  
  // nexamples x nnodes 0, 1 matrix 
  var state: FMat = null
  
  var stateFixed: FMat = null
  
  var igtz: IMat = null
  
  var iltz: IMat = null
  
  var hashCoef: Mat = null
  
  var sumLogProbs: Mat = null
  
  var idsByColor: Array[IMat] = null
  
  var extendedIdsByColor: Array[IMat] = null
  
  // probs for prediction
  var predprobs: FMat = null
  
  def initParams = { 
    val nparents = sum(graph.dag)
    val ncombos = IMat(pow(2 * ones(1, graph.n), nparents+1))
    val nparams = sum(ncombos).v
    params = rand(1, nparams)
    params(1 until nparams by 2) = 1 - params(0 until nparams by 2)
    //cpt0 = loadFMat(cptpath)
 
    paramsOffset = izeros(1, graph.n)
    paramsOffset(1 until graph.n) = cumsum(ncombos)(0 until graph.n-1)
  }
  
  def initIdsByColor = {
    idsByColor = new Array[IMat](graph.ncolors)
    extendedIdsByColor = new Array[IMat](graph.ncolors)
    val m = graph.dag + sparse(IMat(0 until graph.n), IMat(0 until graph.n), ones(1, graph.n))
    for(c <- 0 until graph.ncolors){
      idsByColor(c) = find(graph.colors == c).t
      // the union of idsByColor(c) and all children of idsByColor(c)
      extendedIdsByColor(c) = find( sum(m(idsByColor(c), ?), 1)).t
      
    }
  }
  def initState = {
    if(opts.useGPU){
    	//state = grand(opts.batchSize, graph.n)
    	//stateFixed = grand(opts.batchSize * graph.nsymbols, graph.n)
    }else{
    	state = (rand(opts.batchSize, graph.n) > 0.5)
    	stateFixed = rand(opts.batchSize * graph.nsymbols, graph.n)
    	
    	predprobs = rand(opts.batchSize, graph.n)
    }
  }
  
  def initHashCoef = {
    val dag = graph.dag.copy
    for(i <- 0 until graph.n){
      val parentIds = find(dag(?, i))
      val nparents = parentIds.length
      for(j <-0 until nparents){
        dag(parentIds(j), i) = math.pow(graph.nsymbols, nparents-j).toFloat
      }
    }
    val dag2 = dag + sparse(IMat(0 until graph.n), IMat(0 until graph.n), ones(1, graph.n))
    if(opts.useGPU){
      //hashCoef = GSMat(dag2)
    }else{
      hashCoef = dag2
    }
  }
  
  def initSumLogProbs = {
    val m = graph.dag.t + sparse(IMat(0 until graph.n), IMat(0 until graph.n), ones(1, graph.n))
    if(opts.useGPU){
      //sumLogProbs = GSMat(m)
    }else{
      sumLogProbs = m
    }
  }
  
  def initObsIndex = {
    val fdata = FMat(full(data))
    igtz = find(fdata>0)
    iltz = find(fdata<0)
  }
  
  def init = {
    
    initHashCoef
    initSumLogProbs
    initState
    initIdsByColor
    initParams
    initObsIndex
    
    //saveSMat("C:/data/zp_dlm_FT_code_and_data/debug/mat21.smat", hashCoef.asInstanceOf[SMat])
    //saveSMat("C:/data/zp_dlm_FT_code_and_data/debug/mat22.smat", sumLogProbs.asInstanceOf[SMat])
    
  }
  
  def paramsHelper(hashIndex: IMat) = {
    val nnodes = size(hashIndex, 2)
    val nstates = size(hashIndex, 1)
    val probs = zeros(nstates, nnodes)
    for(i <- 0 until nnodes){
      probs(?, i) = params(hashIndex(?, i))
      //println(params(hashIndex(?, i)))
    }
    probs
  }
   
  def parSample(fdata: FMat, ids: IMat, extendedIds: IMat) = {
    
    // mem management
    //for(i <- 0 until graph.nsymbols){
    	//stateFixed(i*opts.batchSize until (i+1)*opts.batchSize, ?) = state.copy
    	// only works if ids is imat, mat does not work
    	// stateFixed has to be fmat, mat does not work
    	//stateFixed(IMat(i*opts.batchSize until (i+1)*opts.batchSize), ids) = i
    	// it's wrong! 
    	//stateFixed(igtz + i*opts.batchSize*graph.n) = 1
    	//stateFixed(iltz + i*opts.batchSize*graph.n) = 0
    //}
    stateFixed(0 until opts.batchSize, ?) = state.copy
    stateFixed(opts.batchSize until 2*opts.batchSize, ?) = state.copy
    stateFixed(IMat(0 until opts.batchSize), ids) = 0
    stateFixed(IMat(opts.batchSize until 2*opts.batchSize), ids) = 1
    val (r, c, v) = find3(data)
    var c1 = 0
    var c0 = 0
    for(i <- 0 until data.nnz){
      val vs: Float = if(v(i)>0) 1 else 0
      stateFixed(r(i), c(i)) = vs
      stateFixed(r(i) + opts.batchSize, c(i)) = vs
    }

    //saveFMat("C:/data/zp_dlm_FT_code_and_data/debug/statefixed.fmat", stateFixed)
    //println(ids)
    //println(extendedIds)
    //println(stateFixed(IMat(0 until opts.batchSize), ids))
    //println(stateFixed(IMat(opts.batchSize until 2*opts.batchSize), ids))
    val nnodes = size(ids, 1)
    // mem nnodes is changing, convert to IMat
    // edge operator does not work for generic types

    //val hashIds = IMat(paramsOffset(extendedIds) +  stateFixed * hashCoef(?, extendedIds))

    //val logProbs = ln( paramsHelper( hashIds ) + opts.eps )    
    
    //val probsRaw = exp(logProbs * sumLogProbs(extendedIds, ids))
    
    val hashIds = IMat(paramsOffset +  stateFixed * hashCoef)

    val logProbs = ln( paramsHelper( hashIds ) + opts.eps )    
    
    val probsRaw = exp(logProbs * sumLogProbs(?, ids))
    
    //println(probsRaw(opts.batchSize until 2*opts.batchSize, ?))
    //println(probsRaw(0 until opts.batchSize, ?))

    //probs <-- probs / sum(probs, 2)
    val probs = probsRaw(opts.batchSize until 2*opts.batchSize, ?) / (probsRaw(0 until opts.batchSize, ?) + probsRaw(opts.batchSize until 2*opts.batchSize, ?))
    //println(probs)
    
    val rsamples = rand(opts.batchSize, nnodes)

    //assumes 2 symbols!
    val samples = (rsamples <= probs)

    state(?, ids) = samples.copy
    //state(igtz) = 1
    //state(iltz) = 0
    predprobs(?, ids) = probs.copy
  }
  
  def sample( data: SMat ) = {
    
    val fdata = full(data)
    //state(igtz) = 1
    //state(iltz) = 0
    for(c <- 0 until graph.ncolors){
      val ids = idsByColor(c)
      val extendedIds = extendedIdsByColor(c)
      parSample(fdata, ids, extendedIds)
    }
      
    
  }
  
  def update = {
    // enforce states
    state(igtz) = 1
    state(iltz) = 0
    val nstates = size(state, 1)
    val hashIds = IMat(paramsOffset +  state * hashCoef)
    
	var counts = zeros(1, params.length)
	for(i <- 0 until nstates){
	  counts(hashIds(i, ?)) = counts(hashIds(i, ?)) + 1
	}
    counts = counts + opts.beta
	
	// normalize count matrix
	var normcounts = zeros(1, counts.length)
    normcounts(0 until counts.length-1 by 2) = counts(1 until counts.length by 2) 
    normcounts(1 until counts.length by 2) = counts(0 until counts.length-1 by 2) 
    normcounts = normcounts + counts
    counts = counts / normcounts
    
    val alpha = 0.0f
    params = alpha * params + (1-alpha) * counts

    
    //println(params)
  }
  
   def eval2 = {
    sqrt(((params - cpt0).t dot (params - cpt0).t)/((cpt0).t dot (cpt0).t)).v
  }
  def learn = {
      
    val ndata = size(data, 1)    
    for(i <-0 until opts.npasses){ 
       println("iteration: " + i)
       for(j <- 0 until ndata by opts.batchSize){         
    	   //val jend = math.min(ndata, j + opts.batchSize)
    	   //val minidata = data(j until jend, ?)
           val minidata = data
    	   // sample mini-batch of data
    	   sample(minidata)
    	   // update parameters
    	   update      
       }
       println(eval2)
       //pred
    }
  }
    
  def pred = {
    val ndata = size(data, 1)
    for(j <- 0 until ndata by opts.batchSize){         
    	   //val jend = math.min(ndata, j + opts.batchSize)
    	   //val minidata = data(j until jend, ?)
         val minidata = data
    	   // sample mini-batch of data
    	 sample(minidata)
    	   // update parameters
    }
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
  
	
	
}

class Options {
  
	var beta = 0.1
	var npasses = 200
	var batchSize = 10000
	var eps = 1e-10
	var useGPU = false
}

object TestBayesNet2{
  
  def main(args: Array[String]) = {
    
    val opts = new Options()
    val dag = loadSMat("C:/data/zp_dlm_FT_code_and_data/dag.smat")
    val graph = new Graph(dag, 334)
    //val sdata = loadSMat("C:/data/zp_dlm_FT_code_and_data/sdata_cleaned.smat")
    val sdatat = loadSMat("C:/data/zp_dlm_FT_code_and_data/dl-10000-0.100000.smat")
    val vdatat = loadSMat("C:/data/zp_dlm_FT_code_and_data/dl-10000-0.100000.smat")
    //val sdata = loadSMat("C:/data/zp_dlm_FT_code_and_data/train.smat")
    //val vdata = loadSMat("C:/data/zp_dlm_FT_code_and_data/test.smat")
    val sdata = sdatat.t
    val vdata = vdatat.t
    
    // currently supports only batch update
    opts.batchSize = size(sdata, 1)
    setseed(100099)
    // the BayesNet constructor takes a colored the graph
    graph.color

    val cpt0t = loadFMat("C:/data/zp_dlm_FT_code_and_data/cpt.fmat")
    val cpt0 = cpt0t.t
    //flip;
    val bayesnet = new BayesNet2(sdata, vdata, graph, opts, cpt0)
    bayesnet.init
    bayesnet.learn
    //bayesnet.pred
    //println(gflop)
    
  }
}
	