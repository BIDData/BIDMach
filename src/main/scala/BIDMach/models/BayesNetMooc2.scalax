package BIDMach.models

import BIDMat.{ CMat, CSMat, DMat, Dict, IDict, FMat, GMat, GIMat, GSMat, HMat, IMat, Mat, SMat, SBMat, SDMat }
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import java.io._
import scala.util.Random

/**
 * A Bayesian Network implementation with fast parallel Gibbs Sampling on a MOOC dataset.
 * 
 * Haoyu Chen and Daniel Seita are building off of Huasha Zhao's original code.
 *
 * Note: do not use this code for important things since it's still in development and we are
 * transitioning over to a learner framework.
 */
object BayesNetMooc2 {

  /*
   * nodeMap, maps questions/concept codes (start with "I" or "M") into {0,1,...}
   * graph, the Graph data structure of the Bayesian network, with moralizing/coloring capabilities
   * sdata/tdata, train/test data, respectively; a -1 indicates unknowns, {0,1,..,k} are known values.
   * statesPerNode, array [s0,s1,...] where index i means number of possible states for node i
   * cpt, an array where each component indicates Pr(X_i = k | parent combos), ordered by X_i
   * cptOffset, determines correct offset in the cpt array where we look up probabilities
   * state, a (# nodes) x (batchSize) matrix that contains sampled values.
   * iproject, a matrix that provides LOCAL offsets in the CPT for Pr(X_i | parents_i).
   * pproject, a matrix that gets left multiplied to parameters to get Pr(X_i | all other nodes).
   * globalPMatrices, an array of matrices that contain all the probabilities we need for predictions.
   * normConstMatrix, a sparse matrix to left-multiply to our un-normalized cpt to get the dividing factor.
   */
  var nodeMap: scala.collection.mutable.HashMap[String, Int] = null
  var graph: Graph1 = null
  var sdata: SMat = null
  var tdata: SMat = null
  var statesPerNode: IMat = null  
  var cpt: FMat = null
  var cptOffset: IMat = null
  var state: FMat = null
  var iproject: SMat = null
  var pproject: SMat = null
  var globalPMatrices: Array[FMat] = null
  var normConstMatrix: SMat = null

  /*
   * bufSize, the default size of matrices we create for data
   * batchSize, the number of columns of data we analyze at a time
   * niter, the number of Gibbs sampling iterations (100 seems like a reasonable amount)
   * llikelihood, log likelihood and measures sum over all log probabilities for a given state matrix.
   * alpha, a parameter in [0,1] to determine how closely we want to follow current data.
   * beta, a smoothing parameter we add in to the counts to allow nonzero probability for everything
   * verbose, a parameter that we can tick to true if we want to have more verbose output
   */
  val bufsize = 100000
  var batchSize = 1
  var niter = 100
  var llikelihood = 0f
  var alpha = 1f
  var beta = 0.1f
  var verbose = false

  /**
   * Set up paths and variables. Then sample. 
   * 
   * Usage: BayesNetMooc2 <node> <dag> <data> <#nodes> <#columns> <#iters> <state_size>
   */
  def main(args: Array[String]) { 
    val nodepath = args(0)
    val dagpath = args(1)
    val datapath = args(2)
    val numQuestions = args(3).toInt
    val numStudents = args(4).toInt
    batchSize = numStudents
    niter = args(5).toInt
    val stateSizePath = args(6)

    // Include these if you want to just do a split or something like that
    //splitDataStudent(datapath, 0.8, "sdata_student_split.txt")
    // or do: splitTrainTest(path: String, ratio: Double, fileName: String)
    //sys.exit

    init(nodepath, dagpath, stateSizePath)
    loadData(datapath, numQuestions, numStudents)   
    setup

    //println("After main(), our randomly initialized CPT is:")
    //printMatrix(cpt.t)
    //println("Our sdata matrix is:")
    //printMatrix(full(sdata)(0 until 76, 0 until 96))
    //println("Now here is information about color groups:")
    //for (c <- 0 until graph.ncolors) {
    //  val ids = find(graph.colors == c)
    //  println("For color c = " + c + ", the actual nodes here are:")
    //  println(ids.t)
    //}

    sampleAll
  }

  /** Loads the nodes, the DAG, and create a graph with coloring and moralizing capabilities. */
  def init(nodepath: String, dagpath: String, stateSizePath: String) = {
    loadNodeMap(nodepath)
    val n = nodeMap.size
    val dag = loadDag(dagpath, n)
    graph = new Graph1(dag, n)
    loadStateSize(stateSizePath, n)
  }

  /** Puts train/test data in sdata/tdata, respectively. The train/test split should be known in advance. */
  def loadData(datapath:String, nq: Int, ns: Int) = {
    sdata = loadSData(datapath, nq, ns)
    tdata = loadTData(datapath, nq, ns)
  }

  /**
   * Performs several crucial steps before we can begin sampling (e.g., sets the CPTs).
   *
   * Specifically, it performs the following steps:
   *  - colors the graph to get "moralized" graph
   *  - creates pproject, which is just Dag + Identity
   *  - creates numSlotsInCpt, a 1-D vector that records the size of CPT for each node
   *  - creates cptOffset, so cptOffset[i] = index in the cpt where node i's CPT block begins
   *  - initialize cpt randomly, and normalize it for each node's "block"
   *  - initialize iproject, a lower-triangular matrix with ones on the diagonal
   *  - finally, initialize globalPMatrices so that we can assign to them and use for predictions 
   */
  def setup = {
    graph.color
    val parentsPerNode = sum(graph.dag)
    pproject = graph.dag + sparse(IMat(0 until graph.n), IMat(0 until graph.n), ones(1, graph.n))
    val numSlotsInCpt = IMat(exp(DMat(full(pproject.t)) * ln(DMat(statesPerNode))) + 1e-3)     
    // here add 1e-3 just to ensure that IMat can get the correct Int

    val lengthCPT = sum(numSlotsInCpt).v
    cptOffset = izeros(graph.n, 1)
    cptOffset(1 until graph.n) = cumsum(numSlotsInCpt)(0 until graph.n-1)
    cpt = rand(lengthCPT,1)
    normConstMatrix = getNormConstMatrix(cpt)
    cpt = cpt / (normConstMatrix * cpt)

    // Compute iproject. Note that parents = indices of i's parents AND itself, and that we use cumulative sizes.
    iproject = (pproject.copy).t
    for (i <- 0 until graph.n) {
      val parents = find(pproject(?, i))
      var cumRes = 1
      val n = parents.length
      for (j <- 1 until n) {
        cumRes = cumRes * statesPerNode(parents(n - j))
        iproject(i, parents(n - j - 1)) = cumRes.toFloat
      }
    }
    
    //println("here is iproject:")
    //printMatrix((full(iproject))(0 until 50, 0 until 50))
    //println("transposed, which si what we do in the BayesNet versino")
    //printMatrix((full(iproject.t))(0 until 50, 0 until 50))
    //sys.exit

    // Initializing everything to 0 because later we assign to it, so if we never assign to it, it's still 0.
    val maxState = maxi(maxi(statesPerNode,1),2).v
    globalPMatrices = new Array[FMat](maxState)
    for (i <- 0 until maxState) {
      globalPMatrices(i) = zeros(graph.n, batchSize)
    }
  }

  /**
   * Creates normalizing matrix N that we can then multiply with the cpt, i.e., N * cpt, to get a
   * column vector of the same length as the cpt, but such that cpt / (N * cpt) is normalized.
   */
  def getNormConstMatrix(cpt: Mat) : SMat = {
    var ii = izeros(1,1)
    var jj = izeros(1,1)
    for (i <- 0 until graph.n-1) {
      var offset = cptOffset(i)
      val endOffset = cptOffset(i+1)
      val ns = statesPerNode(i)
      var indices = find2(ones(ns,ns))
      while (offset < endOffset) {
        ii = ii on (indices._1 + offset)
        jj = jj on (indices._2 + offset)
        offset = offset + ns
      }
    }
    var offsetLast = cptOffset(graph.n-1)
    var indices = find2(ones(statesPerNode(graph.n-1), statesPerNode(graph.n-1)))
    while (offsetLast < cpt.length) {
      ii = ii on (indices._1 + offsetLast)
      jj = jj on (indices._2 + offsetLast)
      offsetLast = offsetLast + statesPerNode(graph.n-1)
    }
    return sparse(ii(1 until ii.length), jj(1 until jj.length), ones(ii.length-1, 1), cpt.length, cpt.length)
  }

  /** 
   * Called from the main method and iterates through the color groups to sample. It samples on a
   * block of consecutive columns of the sdata, then updates the CPTs, and evaluates. Then repeats.
   * After each iteration, predict on the testing data and report log likelihood.
   */
  def sampleAll = {
    val ndata = size(sdata, 2)
    for (k <- 0 until niter) {
      var j = 0;
      for (i <- 0 until ndata by batchSize) { // Default is to do this just once
        val data = sdata(?, i until math.min(ndata, i + batchSize))
        sample(data, k)
        updateCpt(k)
        eval
      }
      pred(k)
      if (verbose) {
        println("ll: %f" format llikelihood)
      }
      llikelihood = 0f
      //if (k % 10 == 0) {
      //  println("Our cpt:")
      //  printMatrix(FMat(cpt.t))
      //}
    }
    val showThisData = true
    if (showThisData) {
      println("After our sampling, here is the cpt in row-vector form:\n" + cpt.t)
      println("\nNow let's go through each node --- in order by index --- and print out its cpt:\n")
      for ((node,index) <- nodeMap.toSeq.sortBy(_._2)) {
        println("\nAt index " + index + ", CPT for node " + node + " is ...")
        showCpt(node)
      }
      saveFMat("cpt_BNM2_" + niter + "iters.lz4", cpt)
    }
  }

  /** 
   * Main framework for the sampling, which iterates through color groups and samples in parallel.
   * 
   * Some details on the variables:
   *  - numState, max number of states for this color group, so we know the # of state_i matrices.
   *  - stateSet holds all the state_i matrices, which we use for sampling.
   *  - pSet holds all the P_i matrices, which we use for predictions.
   *  - pMatrix records the sum of all P_i matrices, for normalizing purposes.
   *  - ids, the indices of nodes in a color group, AND who have >= i states, so we can use statei.
   *  - pids is the same as ids, except we augment it with the collective CHILDREN indices of ids.
   * 
   * @param data Training data matrix. Rows = nodes, columns = instances of variable assignments.
   * @param k Iteration of Gibbs sampling.
   */
  def sample(data: SMat, k: Int) = {
    val fdata = full(data)-1 // Note the -1 here!
    if (k == 0) {
      initState(fdata)
    }
    for (c <- 0 until graph.ncolors) {
      val idInColor = find(graph.colors == c)
      //println("Now on color c = " + c + " with nodes " + idInColor.t)
      val numState = IMat(maxi(maxi(statesPerNode(idInColor),1),2)).v
      var stateSet = new Array[FMat](numState)
      var pSet = new Array[FMat](numState)
      var pMatrix = zeros(idInColor.length, batchSize)
      for (i <- 0 until numState) {
        val saveID = find(statesPerNode(idInColor) > i)
        val ids = idInColor(saveID)
        val pids = find(sum(pproject(ids, ?), 1))
        initStateColor(fdata, ids, i, stateSet)
        computeP(ids, pids, i, pSet, pMatrix, stateSet(i), saveID, idInColor.length)
      }

      //println("Our state0 matrix is")
      //printMatrix(stateSet(0)(0 until 76, 0 until 96))
      //println("Our state1 matrix is")
      //printMatrix(stateSet(1)(0 until 76, 0 until 96))
      //println("Our un-normalized p0 matrix is")
      //printMatrix(pSet(0)(?, 0 until 20))
      //println("Our un-normalized p1 matrix is")
      //printMatrix(pSet(1)(?, 0 until 20))
      //sys.exit

      sampleColor(fdata, numState, idInColor, pSet, pMatrix)
    }

    //println("We have done a complete gibbs iteration. Here is the state matrix")
    //printMatrix(state(0 until 76, 0 until 96))
  }

  /**
   * Initialize the state matrix with the input training data.
   * 
   * In general, we will have sparse data, with mostly -1s for the elements. Those are the places
   * where we randomize the value, but be careful to randomize appropriately. Rows corresponding
   * to binary variables must only randomize between {0,1}, ternary in {0,1,2}, and so on. To do
   * this, create a random row, rand(1,batchSize), multiply by statesPerNode(row), then IMat(...)
   * will truncate the result. It will truncate, not round, and that distinction is important.
   * 
   * @param fdata Training data matrix, with -1 at unknown values and known values in {0,1,...,k}.
   */
  def initState(fdata: FMat) = {
    state = fdata.copy
    if (!checkState(state)) {
      println("problem with start of initState(), max elem is " + maxi(maxi(state,1),2).dv)
    }
    // New way of randomizing. Wrap a min() around it since rand can sometimes return 1.0.
    state = rand(graph.n, batchSize)
    state = min( FMat(trunc(statesPerNode *@ state)) , statesPerNode-1)
    val innz = find(fdata >= 0)
    state(innz) = fdata(innz)
    if (!checkState(state)) {
      println("problem with end of initState(), max elem is " + maxi(maxi(state,1),2).dv)
    }
  }

  /**
   * Initializes the statei matrix for this particular color group and for this particular value.
   * It fills in the unknown values at the ids locations with i, then we can use it in computeP.
   * 
   * @param fdata Training data matrix, with unknowns of -1 and known values in {0,1,...,k}.
   * @param ids Indices of nodes in this color group that can also attain value/state i.
   * @param i An integer representing a value/state (we use these terms interchangeably).
   * @param stateSet An array of statei matrices, each of which has "i" in the unknowns of "ids".
   */
  def initStateColor(fdata: FMat, ids: IMat, i: Int, stateSet: Array[FMat]) = {
    var statei = state.copy
    statei(ids,?) = i
    if (!checkState(statei)) {
      println("problem with initStateColor(), max elem is " + maxi(maxi(statei,1),2).dv)
    }
    //val innz = find(fdata >= 0)
    //statei(innz) = fdata(innz)
    //if (!checkState(statei)) {
    //  println("problem with end of initStateColor(), max elem is " + maxi(maxi(statei,1),2).dv)
    //}
    stateSet(i) = statei
  }

  /** 
   * Computes the un-normalized probability matrix for attaining a particular state. We also do
   * the cumulative sum for pMatrix so we can eventually use it as a normalizing constant.
   * 
   * @param ids Indices of nodes in this color group that can also attain value/state i.
   * @param pids Indices of nodes in "ids" AND the union of all the children of "ids" nodes.
   * @param i An integer representing a value/state (we use these terms interchangeably).
   * @param pSet The array of matrices, each of which represents probabilities of nodes attaining i.
   * @param pMatrix The matrix that represents normalizing constants for probabilities.
   * @param statei The matrix with unknown values at "ids" locations of "i".
   * @param saveID Indices of nodes in this color group that can attain i. (TODO Is this needed?)
   * @param numPi The number of nodes in the color group of "ids", including those that can't get i.
   */
  def computeP(ids: IMat, pids: IMat, i: Int, pSet: Array[FMat], pMatrix: FMat, statei: FMat, saveID: IMat, numPi: Int) = {
    val nodei = ln(getCpt(cptOffset(pids) + IMat(iproject(pids, ?) * statei)) + 1e-9)
    //println("For color group with nodes " + ids.t + ", our indices we use to access the cpt are:")
    //printMatrix((IMat(iproject(pids,?) * statei))(0 until 50, 0 until 50))
    //println("With global offsets:")
    //printMatrix((cptOffset(pids) + IMat(iproject(pids,?) * statei))(0 until 50, 0 until 50))

    var pii = zeros(numPi, batchSize)
    pii(saveID, ?) = exp(pproject(ids, pids) * nodei)
    pSet(i) = pii
    pMatrix(saveID, ?) = pMatrix(saveID, ?) + pii(saveID, ?)
  }

  /** 
   * For a given color group, after we have gone through all its state possibilities, we sample it.
   * 
   * To start, we use a matrix of random values. Then, we go through each of the possible states and
   * if random values fall in a certain range, we assign the range's corresponding integer {0,1,...,k}.
   * Important! BEFORE changing pSet(i)'s, store them in the globalPMatrices to save for later.
   * 
   * @param fdata Training data matrix, with unknowns of -1 and known values in {0,1,...,k}.
   * @param numState The maximum number of state/values possible of any variable in this color group.
   * @param idInColor Indices of nodes in this color group.
   * @param pSet The array of matrices, each of which represents probabilities of nodes attaining i.
   * @param pMatrix The matrix that represents normalizing constants for probabilities.
   */
  def sampleColor(fdata: FMat, numState: Int, idInColor: IMat, pSet: Array[FMat], pMatrix: FMat) = {
    // Put inside globalPMatrices now because later we overwrite these pSet(i) matrices.
    // For this particular sampling, we are only concerned with idInColor nodes.
    for (i <- 0 until numState) {
      globalPMatrices(i)(idInColor,?) = pSet(i).copy
    }
    val sampleMatrix = rand(idInColor.length, batchSize)
    pSet(0) = pSet(0) / pMatrix
    state(idInColor,?) = 0 * state(idInColor,?)

    //println("\nRemember that we are sampling nodes " + idInColor.t)
    //println("state matrix before sampling is:")
    //printMatrix(state(0 until 76, 0 until 96))
    //println("pSet(0), NORMALIZED, matrix before sampling is:")
    //printMatrix(pSet(0)(?, 0 until 20))
    //println("pSet(1), NORMALIZED, matrix before sampling is:")
    //printMatrix((pSet(1) / pMatrix)(?, 0 until 20))

    // Each time, we check to make sure it's <= pSet(i), but ALSO exceeds the previous \sum (pSet(j)).
    for (i <- 1 until numState) {
      val saveID = find(statesPerNode(idInColor) > i)
      val saveID_before = find(statesPerNode(idInColor) > (i - 1))
      val ids = idInColor(saveID)
      val pids = find(sum(pproject(ids, ?), 1))
      pSet(i) = (pSet(i) / pMatrix) + pSet(i-1) // Normalize and get the cumulative prob
      // Use Hadamard product to ensure that both requirements are held.
      state(ids, ?) = state(ids,?) + i * ((sampleMatrix(saveID, ?) <= pSet(i)(saveID, ?)) *@ (sampleMatrix(saveID, ?) >= pSet(i - 1)(saveID, ?)))
      if (!checkState(state)) {
        println("problem with loop in sampleColor(), max elem is " + maxi(maxi(state,1),2).dv)
      }
    }

    //println("state matrix after sampling is (but before overriding)")
    //printMatrix(state(0 until 76, 0 until 96))

    // Finally, re-write the known state into the state matrix
    val saveIndex = find(fdata >= 0)
    state(saveIndex) = fdata(saveIndex)
    if (!checkState(state)) {
      println("problem with end of sampleColor(), max elem is " + maxi(maxi(state,1),2).dv)
    }

    //println("state matrix after sampling is (and before overriding)")
    //printMatrix(state(0 until 76, 0 until 96))
  }

  /** 
   * After doing sampling with all color groups, update the CPT according to the state matrix.
   * We use beta for a smoothing parameter and alpha in case we want to have a weighed average.
   * Notice here that we are normalizing counts = counts / (normConstMatrix * counts).
   */
  def updateCpt(k : Int) = {
    val nstate = size(state, 2)
    val index = IMat(cptOffset + iproject * state)
    var counts = zeros(cpt.length, 1)
    for (i <- 0 until nstate) {
      counts(index(?, i)) = counts(index(?, i)) + 1
    }
    counts = counts + beta
    //if (k % 5 == 0) {
    //  println("Here is our counts matrix")
    //  printMatrix(counts.t)
    //}
    counts = counts / (normConstMatrix * counts)
    cpt = (1 - alpha) * cpt + alpha * counts
  }

  /** Returns a conditional probability table specified by indices from the "index" matrix. */
  def getCpt(index: IMat) = {
    var cptindex = zeros(index.nr, index.nc)
    for (i <- 0 until index.nc) {
      cptindex(?, i) = cpt(index(?, i))
    }
    cptindex
  }

  /** A debugging method to show the CPT of a node (i.e., a question or concept in the MOOC data). */
  def showCpt(node: String) {
    val id = nodeMap(node)
    val np = sum(graph.dag(?, id)).v
    val offset = cptOffset(id)
    for (i <- 0 until math.pow(2, np).toInt) {
      if (np > 0)
        print(String.format("\t\t%" + np.toInt + "s", i.toBinaryString).replace(" ", "0"))
    }
    print("\n0")
    for (i <- 0 until math.pow(2, np).toInt)
      print("\t%.4f" format cpt(offset + i * 2))
    print("\n1")
    for (i <- 0 until math.pow(2, np).toInt)
      print("\t%.4f" format cpt(offset + i * 2 + 1))
    print("\n")
  }

  /** Evaluates log likelihood based on the "state" matrix. */
  def eval() = {
    val a = cptOffset + IMat(iproject * state)
    val b = maxi(maxi(a,1),2).dv
    if (b >= cpt.length) {
      println("ERROR! In eval(), we have max index " + b + ", but cpt.length = " + cpt.length)
    }
    val index = IMat(cptOffset + iproject * state)
    val ll = ( sum(sum(ln(getCpt(index)))) ) / (index.nrows * index.ncols)
    llikelihood += ll.v   
  }

  /**
   * Predict on the test data using the probabilities we have computed earlier.
   * 
   * @param i The Gibbs sampling iteration number.
   */
  def pred(i: Int) = {
    // For now, start with this since Huasha had it, and it should still work in our code
    // Normally, update parameters on mini-batch of data. But here, use full data.
    val ndata = size(sdata, 1)
    for (j <- 0 until ndata by batchSize) {
      sample(sdata, 0)
    }
    
    // Now load the test data and perform predictions on all known values (i.e., those >= 0).
    val (row, col, values) = find3(tdata >= 0)
    var correct = 0f
    var tot = 0f
    for (i <- 0 until values.length) {
      val rIndex = row(i).toInt
      val maxState = statesPerNode(rIndex)
      val cIndex = col(i).toInt
      val trueValue = values(i).toInt
      var maxProb = globalPMatrices(0)(rIndex,cIndex)
      var predictedValue = 0
      for (j <- 1 until maxState) { // maxState <= globalPMatrices.length
        val newProb = globalPMatrices(1)(rIndex,cIndex)
        if (newProb > maxProb) {
          maxProb = newProb 
          predictedValue = j
        }
      }
      if (predictedValue == trueValue) {
        correct = correct + 1
      }
      tot = tot + 1
    }
    println(i + "," + correct/tot)
  }

  //-----------------------------------------------------------------------------------------------//
  // The subsequent methods are for debugging or preparing the data, so not relevant to Gibbs Sampling

  /** 
   * Loads the node file to create a node map. The node file has each line like:
   *   question_or_concept_ID,integer_index
   * Thus, we can form one entry of the map with one line in the file.
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
  }

  /**
   * Loads the size of state to create state size array: statesPerNode. In the input file, each
   * line looks like N1, 3 (i.e. means there are 3 states for N1 node, which are 0, 1, 2) 
   * 
   * @param path The state size file
   * @param n The number of the nodes
   */
  def loadStateSize(path: String, n: Int) = {
    statesPerNode = izeros(n, 1)
    var lines = scala.io.Source.fromFile(path).getLines
    for (l <- lines) {
      var t = l.split(",")
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
   * the second is the question number, the third is a 0 or a 1, the fourth is the concept ID, and
   * the fifth indicates a test/train. We should be able to directly split the lines to put data in
   * "row", "col" and "v", which are then put into an (nq x ns) sparse matrix.
   * 
   * Note: with new data, the third column should probably be an arbitrary value in {0,1,...,k}. In
   * general, check with the data rather than these comments.
   * 
   * Note: the data is still sparse, but later we do full(data)-1 to get -1 as unknowns.
   * 
   * @param path The path to the sdata_new.txt file, assuming that's what we're using.
   * @param nq The number of nodes (e.g., 334 "concepts/questions" on the MOOC data)
   * @param ns The number of columns (e.g., 4367 "students" on the MOOC data)
   * @return An (nq x ns) sparse training data matrix, should have values in {0,1,...,k} where the 0
   *    indicates an unknown value.
   */
  def loadSData(path: String, nq: Int, ns: Int) = {
    var lines = scala.io.Source.fromFile(path).getLines
    var sMap = new scala.collection.mutable.HashMap[String, Int]()
    var coordinatesMap = new scala.collection.mutable.HashMap[(Int,Int), Int]()
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
      if (t(4) == "1") {
        // NEW! Only add this if we have never seen the pair...
        val a = sMap(shash)
        val b = nodeMap("I"+t(1))
        if (!(coordinatesMap contains (a,b))) {
          coordinatesMap += ((a,b) -> 1)
          row(ptr) = a
          col(ptr) = b
          // Originally for binary data, this was: v(ptr) = (t(3).toFloat - 0.5) * 2
          v(ptr) = t(2).toFloat+1
          ptr = ptr + 1
        }
      }
    }
    var s = sparse(col(0 until ptr), row(0 until ptr), v(0 until ptr), nq, ns)

    // Sanity check, to make sure that no element here exceeds max of all possible states
    if (maxi(maxi(s,1),2).dv > maxi(maxi(statesPerNode,1),2).dv) {
      println("ERROR, max value is " + maxi(maxi(s,1),2).dv)
    }
    s
  }

  /**
   * Loads the data and stores the testing samples in 'tdata' in a similar manner as loadSData().
   * 
   * Note: the data is still sparse, but later we do full(data)-1 to get -1 as unknowns.
   * 
   * @param path The path to the sdata_new.txt file, assuming that's what we're using.
   * @param nq The number of nodes (334 "concepts/questions" on the MOOC data)
   * @param ns The number of columns (4367 "students" on the MOOC data)
   * @return An (nq x ns) sparse training data matrix, should have values in {0,1,...,k} where the 0
   *    indicates an unknown value.
   */
  def loadTData(path: String, nq: Int, ns: Int) = {
    var lines = scala.io.Source.fromFile(path).getLines
    var sMap = new scala.collection.mutable.HashMap[String, Int]()
    var coordinatesMap = new scala.collection.mutable.HashMap[(Int,Int), Int]()
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
      if (t(4) == "0") {
        val a = sMap(shash)
        val b = nodeMap("I"+t(1))
        if (!(coordinatesMap contains (a,b))) {
          coordinatesMap += ((a,b) -> 1)
          row(ptr) = a
          col(ptr) = b
          // Originally for binary data, this was: v(ptr) = (t(3).toFloat - 0.5) * 2
          v(ptr) = t(2).toFloat+1
          ptr = ptr + 1
        }
      }
    }

    // Sanity check, to make sure that no element here exceeds max of all possible states
    var s = sparse(col(0 until ptr), row(0 until ptr), v(0 until ptr), nq, ns)
    if (maxi(maxi(s,1),2).dv > maxi(maxi(statesPerNode,1),2).dv) {
      println("ERROR, max value is " + maxi(maxi(s,1),2).dv)
    }
    s
  }

  /**
   * Splits the 'sdata' file into training and testing, at random, according to a user-determined
   * split ratio. This means taking the file and adding in 0s and 1s to the end (0 = testing). Note
   * that this should not in general be called when we do BayesNetMooc because we should already
   * know the training and testing split.
   * 
   * @param path The path to the sdata_new.txt file, assuming that's what we're using.
   * @param ratio A number between 0 and 1 to indicate the testing/training split.
   * @param fileName The name to call the file that has extra 1s and 0s at the end.
   */
  def splitTrainTest(path: String, ratio: Double, fileName: String) = {
    var lines = scala.io.Source.fromFile(path).getLines
    val writer: PrintWriter = new PrintWriter(fileName)
    for (l <- lines) {
      val r = math.random
      if (r < ratio) {
        writer.println(l + ",1")}
      else {
        writer.println(l + ",0")
      }
    }
    writer.close
  }

  /**
   * A new debug method to split the 'sdata' file into training and testing, but this time, we are
   * splitting the data based on columns, which are students in the MOOC case. This way, it is like
   * we are predicting on unknown STUDENTS, and not just unknown (student, question) pairs.
   * 
   * @param path The path to the sdata.txt file, assuming that's what we're using. For now, let's 
   *    assume there is NO last column that is 0/1 to indicate a test/train.
   * @param ratio A number between 0 and 1 to indicate the testing/training split. If it's 0.8, that
   *    means 80% of the students will be assigned to the training set.
   * @param fileName The name to call the file that has extra 1s and 0s at the end.
   */ 
  def splitDataStudent(path: String, ratio: Double, fileName: String) {
    var lines = scala.io.Source.fromFile(path).getLines
    val writer: PrintWriter = new PrintWriter(fileName)
    // Map from student hash to 0 (=testing) or 1 (=training)
    var sMap = new scala.collection.mutable.HashMap[String, Int]()
    for (l <- lines) {
      var t = l.split(",")
      val shash = t(0)
      if (!(sMap contains shash)) {
        val r = math.random
        if (r < ratio) {
          sMap += (shash -> 1) // Training
        } else {
          sMap += (shash -> 0)
        }
      }
      if (sMap(shash) == 0) {
        writer.println(l + ",0") 
      } else {
        writer.println(l + ",1")  // Training 
      }
    }
    writer.close   
  }
  
  /** A debugging method to print matrices, without being constrained by the command line's cropping. */
  def printMatrix(mat: Mat) = {
    for(i <- 0 until mat.nrows) {
      for (j <- 0 until mat.ncols) {
        print(mat(IMat(i),j) + " ")
      }
      println()
    }
  }

  /** Returns FALSE if there's an element at least size 2, which is BAD. */
  def checkState(state: FMat) : Boolean = {
    val a = maxi(maxi(state,2),1).dv
    if (a >= 2) {
      return false
    }
    return true
  }

  /*
  def checkState(stateToCheck: FMat) = {
    for (i<- 0 until stateToCheck.nrows) {
      for (j <- 0 until stateToCheck.ncols) {
        if (stateToCheck(i,j) >=  2) {
          println("ERROR!!!! We have state(i,j) = " + stateToCheck(i,j))
        }
      }
    }
  } */
}

