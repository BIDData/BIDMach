package BIDMach.models

import BIDMat.{ CMat, CSMat, DMat, Dict, IDict, FMat, GMat, GIMat, GSMat, HMat, IMat, Mat, SMat, SBMat, SDMat }
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import java.io._

/**
 * A Bayesian Network implementation with fast parallel Gibbs Sampling on a MOOC dataset.
 * 
 * Haoyu Chen and Daniel Seita are building off of Huasha Zhao's original code.
 */
object BayesNetMooc3 {

  /*
   * nodeMap, maps questions/concept codes (start with "I" or "M") into {0,1,...}
   * graph, the Graph data structure of the Bayesian network, with moralizing and coloring capabilities
   * sdata/tdata, the train/test data, respectively
   * statesPerNode, array [s0,s1,...] where index i means number of possible states for node i
   * cpt is an array where each component indicates Pr(X_i = k | parent combos), ordered by X_i
   * cptOffset determines correct offset in the cpt array where we look up probabilities
   * state TODO
   */
  var nodeMap: scala.collection.mutable.HashMap[String, Int] = null
  var graph: Graph = null
  var sdata: SMat = null
  var tdata: SMat = null
  var statesPerNode: IMat = null  
  var cpt: FMat = null
  var cptOffset: IMat = null
  var state: FMat = null

  // variables I added (Daniel: these should be SMats, not FMats, also should explain their purpose later TODO)
  var iproject: SMat = null
  var pproject: SMat = null

  /*
   * bufSize, the default size of matrices we create for data
   * batchSize, the number of columns we analyze at a time
   * niter, the number of Gibbs sampling iterations
   * predprobs TODO explain clearly, it's what we use to compute if we're right/wrong I think
   * llikelihood TODO explain clearly what data's log-l we are measuring.
   * alpha TODO
   * beta TODO
   */
  val bufsize = 100000
  var batchSize = 1
  var niter = 100
  var predprobs: FMat = null
  var llikelihood = 0f
  var alpha = 1f
  var beta = 0.1f


  /** Set up paths and variables. Then sample. */
  def main(args: Array[String]) { 
    val nodepath = args(0)
    val dagpath = args(1)
    val datapath = args(2)
    val numQuestions = args(3).toInt
    val numStudents = args(4).toInt
    batchSize = numStudents
    niter = args(5).toInt
    val stateSizePath = args(6)       // here I add a stateSizePath to read in the state size
    init(nodepath, dagpath, stateSizePath)
    loadData(datapath, numQuestions, numStudents)   
    setup
    println("start sample All")
    sampleAll
    //println("Node map is: " + nodeMap)
    //println("Graph DAG is: " + graph.dag)
    //println("statesPerNode is: " + statesPerNode)
  }

  /** Loads the nodes, the DAG, and create a graph with coloring and moralizing capabilities. */
  def init(nodepath: String, dagpath: String, stateSizePath: String) = {
    loadNodeMap(nodepath)
    val n = nodeMap.size
    val dag = loadDag(dagpath, n)
    graph = new Graph(dag, n)
    loadStateSize(stateSizePath, n)
    println("finish the init")
    println(statesPerNode)
  }

  /** Puts train/test data in sdata/tdata, respectively. The train/test split should be known in advance. */
  def loadData(datapath:String, nq: Int, ns: Int) = {
    sdata = loadSData(datapath, nq, ns)
    tdata = loadTData(datapath, nq, ns)
    //println("finish load Data")
    //println("the full s data is " + full(sdata))
    //println("the full t data is " + full(tdata))

  }

  /**
   * Performs several crucial steps before we can begin sampling (e.g., sets the CPTs).
   *
   * numSlotsInCpt is a 1-D vector that records the size of CPT for each node
   * 
   * (0) Keep the graph as it is, so no coloring, yet (TODO).
   * (1) Create numSlotsInCpt, so i^{th} index = # of slots the i^{th} variable needs for all
   *    combinations to be present in the CPT. If i^{th} variable has parents indexed at
   *    {p1,...,pk}, then # of slots is numStates(i)*numStates(p1)*...*numStates(pk). I hope a for
   *    loop here is not too bad, since this is a one-time setup.
   * (2) The cpt is initialized randomly, but then it must be normalized for each node's "block"
   *    (this is really each node's CPT, but all are concatenated together in this giant table we
   *    call "cpt"). For instance, the first node, index 0, may not have any parents, and may take
   *    on three values. Thus, its slot block is 3 and those 3 elements must sum to one.
   * (3) Create cptOffset, so cptOffset[i] = index in the cpt where node i's CPT ("block") begins.
   *    Use the cumulative sum of numSlotsInCpt.
   * (4) Then randomly initialize predprobs, which was what Huasha had originally.
   */
  def setup = {
    graph.color
    val parentsPerNode = sum(graph.dag)
    pproject = graph.dag + sparse(IMat(0 until graph.n), IMat(0 until graph.n), ones(1, graph.n))
    //println("run here")
    //println(full(pproject.t))
   
    val numSlotsInCpt = IMat(exp(DMat(full(pproject.t)) * ln(DMat(statesPerNode))) + 1e-3)     
    // here add 1e-3 just to ensure that IMat can get the correct Int
    
    val lengthCPT = sum(numSlotsInCpt).v
    cptOffset = izeros(graph.n, 1)
    cptOffset(1 until graph.n) = cumsum(numSlotsInCpt)(0 until graph.n-1)
    cpt = rand(lengthCPT,1)
    //println("the tot len of the cpt array is " + lengthCPT)
    println("the cptOffset array is " + cptOffset)
    // TODO Daniel: I'll leave this here for now but come up with a better way to normalize
    for (i <- 0 until graph.n-1) {
      var offset = cptOffset(i)
      val endOffset = cptOffset(i+1)
      while (offset < endOffset) {
        val normConst = sum( cpt(offset until offset+statesPerNode(i)) )
        cpt(offset until offset+statesPerNode(i)) = cpt(offset until offset+statesPerNode(i)) / normConst
        offset = offset + statesPerNode(i)
      }
    }
    var lastOffset = cptOffset(graph.n-1)
    while (lastOffset < cpt.length) {
      val normConst = sum( cpt(lastOffset until lastOffset+statesPerNode(graph.n-1)) )
      cpt(lastOffset until lastOffset+statesPerNode(graph.n-1)) = cpt(lastOffset until lastOffset+statesPerNode(graph.n-1)) / normConst
      lastOffset = lastOffset + statesPerNode(graph.n-1)
    }
    //println("\nNORMALIZED cpt.t = " + cpt.t)
    //println("the normalize cpt is " + cpt)
    // Compute iproject here (pproject already computed) 
    iproject = (pproject.copy).t
    for (i <- 0 until graph.n) {
      val parents = find(pproject(?, i))   // parents is the index of i's parents AND itself
      var cumRes = 1
      val n = parents.length
      for (j <- 1 until n) {
        cumRes = cumRes * statesPerNode(parents(n - j))    // it's the cumulative size
        iproject(i, parents(n - j - 1)) = cumRes.toFloat
      }
    }
    println("the iproject is + " + full(iproject))
    predprobs = rand(graph.n, batchSize)
  }

  /** 
   * Called from the main method and iterates through the color groups to sample. It samples on a
   * block of consecutive columns of the sdata, then updates the CPTs, and evaluates. Then repeats.
   * After each iteration, predict on the testing data and report log likelihood.
   */
  def sampleAll = {
    val ndata = size(sdata, 2)
    println("the num of data is " + ndata)
    for (k <- 0 until niter) {
      var j = 0;
      batchSize = batchSize
      for (i <- 0 until ndata by batchSize) { // Default is to do this just once
        val data = sdata(?, i until math.min(ndata, i + batchSize))
        val realBatchSize = data.ncols
        sample(data, k)
        updateCpt
        eval
      }
      //println("delta: %f, %f, %f, %f" format (eval2._1, eval2._2, eval2._3, llikelihood))
      //pred(k)
      //println("ll: %f" format llikelihood)
      llikelihood = 0f
      //println("dist cpt - cpt0: %f" format ((cpt-cpt0) dot (cpt-cpt0)).v )
    }
  }

  /** 
   * Main framework for the sampling, which iterates through the color groups and samples in parallel.
   * 
   * > numState is the max number of states for this color group, so we know how many state_i matrices to make.
   * > stateSet holds all the state_i matrices, which we use for sampling.
   * > pSet holds all the P_i matrices, which we use for predictions.
   * > pMatrix records the sum of all P_i matrices, for normalizing purposes.
   * > ids is the indices of nodes in a color group, AND who have >= i states, so we can use statei with them.
   * > pids is the same as ids, except we augment it with the collective children indices of ids.
   * 
   * @param data Training data matrix. Rows = nodes/variables, columns = instances of variable assignments.
   * @param k Iteration of Gibbs sampling we are on.
   */
  def sample(data: SMat, k: Int) = {
    val fdata = full(data)
    if (k == 0) {
      initState(fdata)
    }
    for (c <- 0 until graph.ncolors) {
      // TODO Investigate the impact of stateSet and pSet on the matrix caching for GPUs.
      val idInColor = find(graph.colors == c)
      println("color in this group is " + idInColor)
      //val numState = IMat(maxi(maxi(statesPerNode(idInColor),1),2) + 1).v // here we don't need to add 1
      val numState = IMat(maxi(maxi(statesPerNode(idInColor),1),2)).v
      println("numState is " + numState)
      var stateSet = new Array[FMat](numState)
      var pSet = new Array[FMat](numState)
      var pMatrix = zeros(idInColor.length, batchSize)
      for (i <- 0 until numState) {
        //println("the statePerNode in idInColor is " + (statesPerNode(idInColor) >= 0))
        val saveID = find(statesPerNode(idInColor) >= i)
        val ids = idInColor(saveID)
        val pids = find(sum(pproject(ids, ?), 1))

        println("the ids is " + ids)
        println("the pids is " + pids)
        initStateColor(fdata, ids, i, stateSet)
        computeP(ids, pids, i, pSet, pMatrix, stateSet(i), saveID, idInColor.length)
      }
      sampleColor(fdata, numState, idInColor, pSet, pMatrix)
    }
  }

  /**
   * this method init the state matrix (a global variable) with the input training data "fdata"
   * I assume the rule is: unknown variable is -1, the valid value is from 0, 1, 2...
   * Thus, for this data (i.e fdata), we just need to copy the fdata to the state
   */
  def initState(fdata: FMat) = {
    /**
    state = rand(graph.n, batchSize)
    // Tricky: for each row i, randomly initialize it {0,1,...,k-1} where k = statesPerNode(i).
    // TODO Not done with the previously mentioned step, because rand(...) needs to be broken down into integers
    println(full(fdata))
    println((fdata >= 0))
    val innz = find(fdata >= 0)
    state(innz) = 0
    // I think the following line should work
    state = state + fdata(innz) // If fdata is -1 at unknowns (innz will skip), and 0, 1, ..., k at known values.
    **/
    state = fdata.copy
    //println("the state is " + state)
  }

  /**
   * this method init the statei for the data 
   * it should fill the unknown values at ids location with the int i
   * then we can use it to compute the pi in the function computeP
   */
  def initStateColor(fdata: FMat, ids: IMat, i: Int, stateSet: Array[FMat]) = {
    var statei = state.copy
    //println("the original statei is " + statei)
    statei(ids,?) = i
    val innz = find(fdata >= 0)   // since we assume that -1 represents the null in the data set
    statei(innz) = 0
    // TODO One last step, but depends on how our fdata is. We need to put in all "known" values. Something like:
    //println("the innz is " + innz)
    //println("the fdata is " + fdata)
    //println(fdata(innz))
    //println(statei)
    statei(innz) = statei(innz) + fdata(innz) // IF fdata is -1 at unknowns (innz will skip), and 0, 1, ..., k at known values.
    //println("the updated statei is " + statei)
    stateSet(i) = statei
  }

  /** 
   * this method calculate the Pi to prepare to compute the real probability 
   */
  def computeP(ids: IMat, pids: IMat, i: Int, pi: Array[FMat], pMatrix: FMat, statei: FMat, saveID: IMat, numPi: Int) = {
    val nodei = ln(getCpt(cptOffset(pids) + IMat(iproject(pids, ?) * statei)) + 1e-10)
    //println("nodei is " + nodei)
    /**
    println("offsect debug")
    println("the statei is ")
    println("the offsect is " + cptOffset)
    printMatrix(statei)
    println("the pids is " + pids)
    println((cptOffset(pids) + IMat(iproject(pids, ?) * statei)))
    println("end of debug")
    **/
    var pii = zeros(numPi, batchSize)
    pii(saveID, ?) = exp(pproject(ids, pids) * nodei)   // Un-normalized probs, i.e., Pr(X_i | parents_i)
    //println("pi is " + pi(i))
    //println("the ids is " + ids)
    //println("the saveID is " + saveID)
    //println("the pm(ids, ?) is " + pMatrix(saveID, ?))
    pi(i) = pii
    pMatrix(saveID, ?) = pMatrix(saveID, ?) + pii(saveID, ?)      // Save this un-normalized prob into the pMatrix for norm later
    
    println(pii)
    println(pMatrix)
  }

  /** This method sample the state for the given color group */
  def sampleColor(fdata: FMat, numState: Int, idInColor: IMat, pSet: Array[FMat], pMatrix: FMat) = {
    val sampleMatrix = rand(idInColor.length, batchSize)     // A matrix w/random values for sampling
    //var sampleProb = new Array[FMat](numState)          // It's ugly here, but easy to map the data..
    println("begin sample color")
    println("sample matrix is " + sampleMatrix)
    println("num_state is " + numState)
    for (i <- 0 until numState) {
      val saveID = find(statesPerNode(idInColor) >= i)
      val saveID_before = find(statesPerNode(idInColor) >= (i - 1))
      val ids = idInColor(saveID)
      val pids = find(sum(pproject(ids, ?), 1))
      
      // problem! here it is not cumulateive!!! TODO here
      //println("the normalized matrix is " + pSet(i))
      // update the state matrix by sampleMatrix random value
      if (i == 0) {
        pSet(i) = pSet(i) / pMatrix  // Normalize and get the cumulative prob
        state(ids, ?) = i * (sampleMatrix(saveID, ?) <= pSet(i)(saveID, ?))
        //println((sampleMatrix <= pSet(i)))
        //println("if *2: " + 2* (sampleMatrix <= pSet(i)))
      } else {
        pSet(i) = pSet(i) / pMatrix + pSet(i-1) // Normalize and get the cumulative prob
        //println("<=: " + ((sampleMatrix(saveID, ?) <= pSet(i))))
        //println(">=: " + (sampleMatrix(saveID,?) >= pSet(i - 1)))
        //println("*: " + ((sampleMatrix(saveID, ?) <= pSet(i)) *@ (sampleMatrix(saveID, ?) >= pSet(i - 1))))
        //println("state(ids: " + state(ids, ?))
        state(ids, ?) = i * ((sampleMatrix(saveID, ?) <= pSet(i)(saveID, ?)) *@ (sampleMatrix(saveID, ?) >= pSet(i - 1)(saveID, ?)))
      }
      println("the normalized matrix is " + pSet(i))
    }
    // then re-write the known state into the state matrix
    //println(fdata(find(fdata >= 0)))
    val saveIndex = find(fdata >= 0)
    state(saveIndex) = fdata(saveIndex)       // here we assume the unknown state to be -1 
    println("the updated state is " + state)
  }

  /** Returns a conditional probability table specified by indices from the "index" matrix. */
  def getCpt(index: IMat) = {
    var cptindex = zeros(index.nr, index.nc)
    for (i <- 0 until index.nc) {
      cptindex(?, i) = cpt(index(?, i))
    }
    cptindex
  }

  // update the cpt method
  def updateCpt = {
    val nstate = size(state, 2)       // num of students
    val maxState = maxi(statesPerNode, 1).v
    //println("the max state is " + maxState)
    // here we need count each different state value by a loop
    println("updateCPT")
    printMatrix(state)
    println((state>0))
    val index = IMat(cptOffset + iproject * state)
    var counts = zeros(cpt.length, 1)
    for (i <- 0 until nstate) {
      counts(index(?, i)) = counts(index(?, i)) + 1
    }
    counts = counts + beta

    for (i <- 0 until graph.n-1) {
      var offset = cptOffset(i)
      val endOffset = cptOffset(i+1)
      while (offset < endOffset) {
        val normConst = sum( counts(offset until offset+statesPerNode(i)) )
        counts(offset until offset+statesPerNode(i)) = counts(offset until offset+statesPerNode(i)) / normConst
        offset = offset + statesPerNode(i)
      }
    }
    var lastOffset = cptOffset(graph.n-1)
    while (lastOffset < counts.length) {
      val normConst = sum( counts(lastOffset until lastOffset+statesPerNode(graph.n-1)) )
      counts(lastOffset until lastOffset+statesPerNode(graph.n-1)) = counts(lastOffset until lastOffset+statesPerNode(graph.n-1)) / normConst
      lastOffset = lastOffset + statesPerNode(graph.n-1)
    }
    cpt = (1 - alpha) * cpt + alpha * counts
    println("the cpt is")
    printMatrix(cpt)
    
  }

  // evaluation method
  def eval() = {
    
  }

  // prediction method, i = Gibbs sampling iteration number
  def pred(i: Int) = {
    /**
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
    **/
  }

  // The subsequent methods are all for preparing the data, so not relevant to Gibbs Sampling. 

  /** 
   * Loads the node file to create a node map. The node file has each line like:
   *   question_or_concept_ID,integer_index
   * Thus, we can form one entry of the map with one line in the file.
   *
   * UPDATE! We now need the number of states per node. In general, we probably should force the
   * node.txt file to have a third argument which is the number of states. For now, let's make the
   * default number as 2.
   *
   * We can assign the values directly here because the two data structures are global variables.
   * The first is a map from strings (i.e., questions/concepts) to integers (0 to n-1), and the
   * second is an IMat array with the # of states for each node.
   *
   * @param path The file path to the node file (e.g, node.txt).
   */
  def loadNodeMap(path: String) = {
    var tempNodeMap = new scala.collection.mutable.HashMap[String, Int]()
    var lines = scala.io.Source.fromFile(path).getLines
    for (l <- lines) {
      var t = l.split(",")
      tempNodeMap += (t(0) -> (t(1).toInt - 1))
    }
    nodeMap = tempNodeMap
    // statesPerNode = IMat(2 * ones(1,nodeMap.size)) // TODO For now... I separate into another file
  }

  /**
   *  this method loads the size of state to create state size array: statesPerNode
   * In the input file, each line looks like N1, 3 (i.e. means there are 3 states for N1 node, which are 0, 1, 2) 
   * @param path the state size file
   * @param n: the number of the nodes
   */
  def loadStateSize(path: String, n: Int) = {
    statesPerNode = izeros(n, 1)
    var lines = scala.io.Source.fromFile(path).getLines
    for (l <- lines) {
      var t = l.split(",")
      //println(l)
      //println("the node id is: " + nodeMap(t(0)))
      statesPerNode(nodeMap(t(0))) = t(1).toInt
    }

  }

  /** 
   * Loads the dag file and converts it to an adjacency matrix so that we can create a graph object.
   * Each line consists of two nodes represented by Strings, so use the previously-formed nodeMap to
   * create the actual entries in the dag matrix. Column i is such that a 1 in a position somewhere
   * indicates a parent of node i.
   *
   * @param path The path to the dag file (e.g., dag.txt).
   * @return An adjacency matrix (of type SMat) for use to create a graph.
   */
   // TODO: we need to change it to be read into the 1,2,3,4,.., and fill unknown with -1
  def loadDag(path: String, n: Int) = {
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
    sparse(row(0 until ptr), col(0 until ptr), v(0 until ptr), n, n)
  }

  /**
   * Loads the data and stores the training samples in 'sdata'. 
   *
   * The path refers to a text file that consists of five columns: the first is the line's hash, the
   * second is the student number, the third is the question number, the fourth is a 0 or a 1, and
   * the fifth is the concept ID.
   *
   * There's a little bit of out-of-place code here because the PrintWriter will create a data file
   * "sdata_new" that augments the original sdata we have (in "sdata_cleaned") with a 0 or a 1 at
   * the end of each line, randomly chosen in a ratio we pick. A "1" indicates that the line should
   * be part of training; a "0" means it should be test data.
   *
   * If we do already have sdata_new, we do not need the PrintWriter code and can directly split the
   * lines to put data in "row", "col" and "v", which are then put into an (nq x ns) sparse matrix.
   * 
   * @param path The path to the sdata_new.txt file, assuming that's what we're using.
   * @param nq The number of questions (334 on the MOOC data)
   * @param ns The number of students (4367 on the MOOC data)
   * @return An (nq x ns) sparse matrix that represents the training data. It's sparse because
   *    students only answered a few questions each, and values will be {-1, 0, 1}.
   */

   // TODO: we need to change it to be read into the 1,2,3,4,.., and fill unknown with -1
  def loadSData(path: String, nq: Int, ns: Int) = {
    var lines = scala.io.Source.fromFile(path).getLines
    var sMap = new scala.collection.mutable.HashMap[String, Int]()
    var row = izeros(bufsize, 1)
    var col = izeros(bufsize, 1)
    var v = zeros(bufsize, 1)
    var ptr = 0
    var sid = 0
   
    // Involves putting data in row, col, and v so that we can then create a sparse matrix, "sdata".
    // Be aware of the maps we use: sMap and nodeMap.
    for (l <- lines) {
      if (ptr % bufsize == 0 && ptr > 0) {
        row = row on izeros(bufsize, 1)
        col = col on izeros(bufsize, 1)
        v = v on zeros(bufsize, 1)
      }
      var t = l.split(",")
      val shash = t(0)
      // add this new line to hash table
      if (!(sMap contains shash)) {
        sMap += (shash -> sid)
        sid = sid + 1
      }
      if (t(5) == "1") {
        row(ptr) = sMap(shash)
        col(ptr) = nodeMap("I" + t(2))
        // I change this one to be:
        // orgininal one: v(ptr) = (t(3).toFloat - 0.5) * 2
        v(ptr) = t(3).toFloat
        ptr = ptr + 1
      }
    }
    var s = sparse(col(0 until ptr), row(0 until ptr), v(0 until ptr), nq, ns)
    s
  }

  /**
   * Loads the data and stores the testing samples in 'tdata' in a similar manner as loadSData().
   * 
   * @param path The path to the sdata_new.txt file, assuming that's what we're using.
   * @param nq The number of questions (334 on the MOOC data)
   * @param ns The number of students (4367 on the MOOC data)
   * @return An (nq x ns) sparse matrix that represents the training data. It's sparse because
   *    students only answered a few questions each, and values will be {-1, 0, 1}.
   */
   // TODO: we need to change it to be read into the 1,2,3,4,.., and fill unknown with -1
  def loadTData(path: String, nq: Int, ns: Int) = {
    var lines = scala.io.Source.fromFile(path).getLines
    var sMap = new scala.collection.mutable.HashMap[String, Int]()
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
        v(ptr) = t(3).toFloat
        ptr = ptr + 1
      }
    }
    var s = sparse(col(0 until ptr), row(0 until ptr), v(0 until ptr), nq, ns)
    s
  }

  /**
   * Splits the 'sdata' file into training and testing, at random, according to a user-determined
   * split ratio. This means taking the file and adding in 0s and 1s to the end (0 = testing). Note
   * that this should not in general be called when we do BayesNetMooc because we should already
   * know the training and testing split.
   * 
   * @param path The path to the sdata_new.txt file, assuming that's what we're using.
   * @param ratio A number between 0 and 1 to indicate the train/testing split.
   * @param fileName The name to call the file that has extra 1s and 0s at the end.
   */
  def splitTrainTest(path: String, ratio: Double, fileName: String) = {
    var lines = scala.io.Source.fromFile(path).getLines
    val writer: PrintWriter = new PrintWriter(fileName)
    for(l <- lines) {
      val r = math.random
      if(r < ratio) {
        writer.println(l + ",1")}
      else {
        writer.println(l + ",0")
      }
    }
    writer.close
  }
  
  //-----------------------------------------------------------------------------------------------//
  // debug method below
  def printMatrix(mat: FMat) = {
    
    for(i <- 0 until mat.nrows) {
      for (j <- 0 until mat.ncols) {
        print(mat(i,j) + " ")
      }
      println()
    }
  }
}
