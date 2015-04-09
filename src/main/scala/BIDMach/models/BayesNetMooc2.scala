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
 * This is "v2" so Daniel Seita is trying to start a fresh version of this, based on Huasha's
 * original code.
 */
object BayesNetMooc2 {

  // state contains the current values (0 or 1) for each node-student combination
  // state0 and state1 are similar to state, but with extra 0s and 1s in the current rows we're sampling
  //var state: FMat = null
  //var state0: FMat = null
  //var state1: FMat = null
  //var alpha = 1f
  //var beta = 0.1f
  //var nsampls = 1
  //var llikelihood = 0f






  /*
   * nodeMap, maps questions/concept codes (start with "I" or "M") into {0,1,...}
   * graph, the Graph data structure of the bayesian network, with moralizing and coloring capabilities
   * sdata/tdata, the train/test data, respectively
   * statesPerNode, array [s0,s1,...] where index i means number of possible states for node i
   * cpt is an array where each component indicates Pr(X_i = k | parent combos), ordered by X_i
   * cptOffset determines correct offset in the cpt array where we look up probabilities
   */
  var nodeMap: scala.collection.mutable.HashMap[String, Int] = null
  var graph: Graph = null
  var sdata: SMat = null
  var tdata: SMat = null
  var statesPerNode: IMat = null
  var cpt: FMat = null
  var cptOffset: IMat = null

  /*
   * bufSize, the default size of matrices we create for data
   * batchSize, the number of columns we analyze at a time
   * niter, the number of Gibbs sampling iterations
   * predprobs TODO explain clearly, it's what we use to compute if we're right/wrong I think
   */
  val bufsize = 100000
  var batchSize = 1
  var niter = 100
  var predprobs: FMat = null

  /** Set up paths and variables. Then sample. */
  def main(args: Array[String]) { 
    val nodepath = args(0)
    val dagpath = args(1)
    val datapath = args(2)
    val numQuestions = args(3).toInt
    val numStudents = args(4).toInt
    batchSize = numStudents
    niter = args(5).toInt
    init(nodepath, dagpath)
    loadData(datapath, numQuestions, numStudents)   
    setup
    //sampleAll
    //println("Node map is: " + nodeMap)
    //println("Graph DAG is: " + graph.dag)
    //println("statesPerNode is: " + statesPerNode)
  }

  /** Loads the nodes, the DAG, and create a graph with coloring and moralizing capabilities. */
  def init(nodepath: String, dagpath: String) = {
    loadNodeMap(nodepath)
    val n = nodeMap.size
    val dag = loadDag(dagpath, n)
    graph = new Graph(dag, n)
  }

  /** Puts train/test data in sdata/tdata, respectively. The train/test split should be known in advance. */
  def loadData(datapath:String, nq: Int, ns: Int) = {
    sdata = loadSData(datapath, nq, ns)
    tdata = loadTData(datapath, nq, ns)
  }

  /**
   * Performs several crucial steps before we can begin sampling (e.g., sets the CPTs).
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
    val parentsPerNode = sum(graph.dag)
    val numSlotsInCpt = statesPerNode.copy
    for (i <- 0 until graph.n) {
      val parentIndices = find(graph.dag(?,i))
      if (parentIndices.length > 0) {
        val statesPerParent = statesPerNode(parentIndices)
        numSlotsInCpt(i) = numSlotsInCpt(i) * prod(statesPerParent)
      }
    }
    val lengthCPT = sum(numSlotsInCpt).v
    cpt = rand(lengthCPT,1)
    cptOffset = izeros(graph.n, 1)
    cptOffset(1 until graph.n) = cumsum(numSlotsInCpt)(0 until graph.n-1)
    //println("parentsPerNode = " + parentsPerNode)
    //println("numSlotsInCpt = " + numSlotsInCpt)
    //println("lengthCPT = " + lengthCPT)
    //println("\ncpt.t = " + cpt.t)
    //println("\ncptOffset.t = " + cptOffset.t)
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
    predprobs = rand(graph.n, batchSize)
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
    statesPerNode = IMat(2 * ones(1,nodeMap.size)) // For now...
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
    (s > 0) - (s < 0)
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
        v(ptr) = (t(3).toFloat - 0.5) * 2
        ptr = ptr + 1
      }
    }
    var s = sparse(col(0 until ptr), row(0 until ptr), v(0 until ptr), nq, ns)
    (s > 0) - (s < 0)
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
 
}
