package BayesNet

import BIDMat.{CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SBMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._


// data 1, -1 sparse matrix
class BayesNet2 (val data: Mat, val graph: Graph, val opts: Options) {

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
    	state = rand(opts.batchSize, graph.n)
    	stateFixed = rand(opts.batchSize * graph.nsymbols, graph.n)
    }
  }
  
  def initHashCoef = {
    val dag = graph.dag
    for(i <- 0 until graph.n){
      val parentIds = find(dag(?, i))
      val nparents = parentIds.length
      for(j <-0 until nparents){
        dag(parentIds(j), i) = math.pow(graph.nsymbols, nparents-j).toFloat
      }
    }
    
    if(opts.useGPU){
      //hashCoef = GSMat(dag)
    }else{
      hashCoef = dag
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
    
  }
  
  def paramsHelper(hashIndex: IMat) = {
    val nnodes = size(hashIndex, 2)
    val nstates = size(hashIndex, 1)
    val probs = zeros(nstates, nnodes)
    for(i <- 0 until nnodes){
      probs(?, i) = params(hashIndex(?, i))
    }
    probs
  }
   
  def parSample(fdata: Mat, ids: IMat, extendedIds: IMat) = {
    
    // mem management
    for(i <- 0 until graph.nsymbols){
    	stateFixed(i*opts.batchSize until (i+1)*opts.batchSize, ?) = state.copy
    	// only works if ids is imat, mat does not work
    	// stateFixed has to be fmat, mat does not work
    	stateFixed(ids + i*opts.batchSize*graph.n) = i 
    	stateFixed(igtz + i*opts.batchSize*graph.n) = 1
    	stateFixed(iltz + i*opts.batchSize*graph.n) = 0
    }
    
    val nnodes = size(ids, 1)
    // mem nnodes is changing, convert to IMat
    // edge operator does not work for generic types
    val hashIds = IMat(paramsOffset(extendedIds) +  stateFixed * hashCoef(?, extendedIds))
    val logProbs = ln( paramsHelper( hashIds ) + opts.eps )    

    val probsRaw = exp(logProbs * sumLogProbs(extendedIds, ids))
    //probs <-- probs / sum(probs, 2)
    val probs = probsRaw(opts.batchSize until 2*opts.batchSize, ?) / (probsRaw(0 until opts.batchSize, ?) + probsRaw(opts.batchSize until 2*opts.batchSize, ?))
    
    val rsamples = rand(opts.batchSize, nnodes)

    //assumes 2 symbols!
    val samples = (rsamples <= probs)

    state(?, ids) = samples
    //state(igtz) = 1
    //state(iltz) = 0
  }
  
  def sample( data: Mat ) = {
    
    val fdata = full(data)
    
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
	var counts = zeros(params.length, 1)
	for(i <- 0 until nstates){
	  counts(hashIds(i, ?)) = counts(hashIds(i, ?)) + 1
	}
    counts = counts + opts.beta
    	
	// normalize count matrix
	var normcounts = zeros(counts.length, 1)
    normcounts(0 until counts.length-1 by 2) = counts(1 until counts.length by 2) 
    normcounts(1 until counts.length by 2) = counts(0 until counts.length-1 by 2) 
    normcounts = normcounts + counts
    counts = counts / normcounts
    
    params = counts.copy
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
    }
  }
	
	
}

class Options {
  
	var beta = 0.1
	var npasses = 100 
	var batchSize = 10000
	var eps = 1e-10
	var useGPU = false
}

object TestBayesNet2{
  
  def main(args: Array[String]) = {
    
    val opts = new Options()
    val dag = loadSMat("C:/data/zp_dlm_FT_code_and_data/dag.smat")
    val graph = new Graph(dag, 334)
    val sdata = loadSMat("C:/data/zp_dlm_FT_code_and_data/sdata_cleaned.smat")
    
    // currently supports only batch update
    opts.batchSize = size(sdata, 1)
    
    // the BayesNet constructor takes a colored the graph
    graph.color
    
    flip;
    val bayesnet = new BayesNet2(sdata, graph, opts)
    bayesnet.init
    bayesnet.learn
    println(gflop)
    
  }
}
	