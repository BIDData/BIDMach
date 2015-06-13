package BIDMach.models

import BIDMat.{ CMat, CSMat, DMat, Dict, IDict, FMat, GMat, GIMat, GSMat, HMat, IMat, Mat, SMat, SBMat, SDMat }
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import java.io._

/**
 * A Bayesian Network implementation with fast parallel Gibbs Sampling on a MOOC dataset. 
 * This is Huasha Zhao's versions, with comments added by Daniel Seita.
 * Do not use this version. Use BayesNetMooc2.scala.
 */
object BayesNetMooc {

  /*
   * nodeMap maps questions/concept codes (start with "I" or "M") into {0,1,...}
   * graph is the graph data structure of the bayesian network, with moralizing and coloring capabilities
   * sdata/tdata are the train/test data, respectively
   * cpt is an array where each component indicates Pr(X_i = {0,1} | parent combos), ordred by X_i.
   * cptoffset determines correct offset in the cpt array where we look up probabilities
   * iproject is the dag transposed, plus extra 2s, 4s, 8s, and 16s, plus the identity matrix.
   * pproject is the dag + identity matrix, or the A_{\mathcal{V}_c} matrix in Huasha's writeup.
   * state contains the current values (0 or 1) for each node-student combination
   * state0 and state1 are similar to state, but with extra 0s and 1s in the current rows we're sampling
   */
  var nodeMap: scala.collection.mutable.HashMap[String, Int] = null
  var graph: Graph1 = null
  var sdata: SMat = null
  var tdata: SMat = null
  var cpt: FMat = null
  var cptoffset: IMat = null
  var iproject: SMat = null
  var pproject: SMat = null
  var state: FMat = null
  var state0: FMat = null
  var state1: FMat = null

  // These are various parameters; note that batchSize gets overriden with user-determined arguments.
  var alpha = 1f  // How much weight we want to put on the new cpt (1 = we put all weight on it)
  var beta = 0.1f // Smoothing parameter for counts in the cpt
  var batchSize = 1
  var niter = 100
  var nsampls = 1
  var llikelihood = 0f
  var predprobs: FMat = null

  /** Set up paths and variables. nq = num questions/concepts, ns = num students. Then sample. */
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

    println("Our cpt.t is:")
    println(cpt.t)
    println("Our sdata matrix is:")
    println(full(sdata))
    sys.exit

    sampleAll
  }

  /** Loads the nodes, the DAG, and create a graph with coloring and moralizing capabilities. */
  def init(nodepath: String, dagpath: String) = {
    nodeMap = loadNodeMap(nodepath)
    val n = nodeMap.size
    val dag = loadDag(dagpath, n)
    graph = new Graph1(dag, n)
  }

  /**
   * Performs several crucial steps before we can begin sampling (e.g., setting the CPTs).
   *
   * First, the graph is colored. Then we prepare the CPT (conditional probability tables) by first
   * determining how many states we need to store for each node. Our "ns" is a vector encoding the
   * offset for each node. If node "i" has one parent, then ns(i) = 4 because it will take up four
   * spots in the CPT array; if "p" is its parent, then  Pr(i={0,1}|p={0,1}) has four possibilities.
   *
   * The cpt is initialized randomly, with every 2 elements summing to one because of probabiilty
   * rules with binary variables. Then the "cptoffset" stores the actual offsets (its values are
   * strictly increasing). Finally, the "iproject" and "pproject" matrices are formed. The
   * "pproject" is the original dag plus the identity matrix, and is the A_{V_c} in Huasha's
   * writeup. The "iproject" is a bit more complicated: for each row i, it is zero except at the
   * i^th spot (which is 1), and for any j s.t. j is a parent of i, we have iproject(i,j) = 2,4,8,
   * or 16, depending on which parent it was. Usually it's just 2, though.
   */
  def setup = {
    graph.color
    val np = sum(graph.dag)
    val ns = IMat(pow(2 * ones(1, graph.n), np + 1))
    //setseed(1003)
    val lcpt = sum(ns).v
    cpt = rand(lcpt, 1)
    cpt(1 until lcpt by 2) = 1 - cpt(0 until lcpt by 2)
    cptoffset = izeros(graph.n, 1)
    cptoffset(1 until graph.n) = cumsum(ns)(0 until graph.n - 1)
    // Prepare projection matrix (this matrix gives LOCAL cpt offsets for Pr(X_i | parents))
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
    predprobs = rand(graph.n, batchSize)
  }

  /** Puts train/test data in sdata/tdata, respectively. The train/test split should be known in advance. */
  def loadData(datapath:String, nq: Int, ns: Int) = {
    sdata = loadSData(datapath, nq, ns)
    tdata = loadTData(datapath, nq, ns)
  }

  /** 
   * Called from the main method and iterates through the color groups to sample. It samples on a
   * block of consecutive columns of the sdata, then updates the CPTs, and evalues. Then repeats.
   * After each iteration, predict on the testing data and report log likelihood.
   */
  def sampleAll = {
    val ndata = size(sdata, 2)
    for (k <- 0 until niter) {
      var j = 0;
      batchSize = batchSize
      nsampls = nsampls
      for (i <- 0 until ndata by batchSize) { // Default is to do this just once
        sample(sdata(?, i until math.min(ndata, i + batchSize)), k)
        updateCpt
        eval
      }
      pred(k)
      llikelihood = 0f
    }
  }

  /**
   * Performs Gibbs sampling on a slice of the data. This is where we iterate through color groups.
   *
   * @param data A sparse data matrix that is a block of the sdata (but with same # of rows). It
   *    contains 1s and -1s.
   * @param k The current iteration of Gibbs sampling. If k=0, we need to initialize the states.
   *    Note that this is also 0 when we call sample from pred(k) in sampleAll().
   */
  def sample(data: SMat, k: Int) = {
    val fdata = full(data)
    if (k == 0) {
      initState(fdata)
    }
    for (i <- 0 until 1) { // ??? Is there a benefit to increasing the number of iterations?
      for (c <- 0 until graph.ncolors) {
        val ids = find(graph.colors == c)
        val pids = find(sum(pproject(ids, ?), 1))
        initStateColor(fdata, ids)
        sampleColor(fdata, ids, pids)
      }
    }
  }

  /**
   * For the first iteration of Gibbs sampling, initialize "state" to be of the same dimensions as
   * fdata (assuming nsampls=1) with 0s and 1s at random. But then, take the data from fdata and put
   * 0s and 1s in "state" corresponding to -1s and +1s, respectively, in fdata. Basically, this
   * randomizes the unknown node-student pairs (and there are a lot), but keeps the known ones.
   *
   * @param fdata A sparse data matrix that is a block of the sdata (but with same # of rows). It
   *    contains 1s and -1s.
   */
  def initState(fdata: FMat) = {
    val ndata = size(fdata, 2)
    state = rand(graph.n, batchSize * nsampls)
    state = (state >= 0.5)
    //state = (state >= 0.5) - (state < 0.5)
    val innz = find(fdata)
    for (i <- 0 until batchSize * nsampls by batchSize) {
      state(innz + i * graph.n) = 0 // The i*graph.n term shifts all the indices over
      state(?, i until i + batchSize) = state(?, i until i + batchSize) + (fdata > 0)
    }
  }

  /**
   * Creates state0 and state1 by copying over from "state". For the nodes in the current color
   * group, their rows in state0 and state1 are all 0s and 1s, respectively. Then, we fill in the
   * known node-student pairs in a similar manner as initState(), though here we are careful to
   * reset the nonzeros back to 0, which means state0 and state1 only have 0s and 1s. In short,
   * state0 and state1 are like state except in the rows that correspond to the ids.
   *
   * @param fdata The sparse data matrix consisting of -1s and 1s.
   * @param ids The indices (0 to 333 for MOOC data) of the nodes in this current color group.
   */
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

  /**
   * Samples all the nodes in the color group (from "ids") in parallel. The nodep0 and nodep1
   * matrices contain the log probabilities of the nodes in the color group given their parents,
   * plus the log probabilities of those children in the set. Then we exponentiate the result of
   * those multipled with pproject. Finally, we sample from a uniform distribution and then decide
   * on the final sampled value for each node (either a 0 or a 1). Notice that the way we sample is
   * not exactly the same as in Huasha's writeup, but the outcome should be equivalent.
   *
   * TODO Double check if the NaNs are going to be a problem (from the p1 / (p0 + p1) line).
   *
   * @param fdata the sparse data matrix consisting of -1s and 1s.
   * @param ids Indices of the nodes we are coloring.
   * @param pids Indices of the nodes we are coloring and their children set.
   */
  def sampleColor(fdata: FMat, ids: IMat, pids: IMat) = {
    val nnode = size(ids, 1)
    //val ndata = size(fdata, 2)
    //val a = IMat(cptoffset(pids) + iproject(pids,?)*state0)
    val nodep0 = ln(getCpt(cptoffset(pids) + IMat(iproject(pids, ?) * state0)) + 1e-10)
    val nodep1 = ln(getCpt(cptoffset(pids) + IMat(iproject(pids, ?) * state1)) + 1e-10)
    val p0 = exp(pproject(ids, pids) * nodep0)
    val p1 = exp(pproject(ids, pids) * nodep1)
    val p = p1 / (p0 + p1)
    var sample = rand(nnode, batchSize * nsampls)
    sample = (sample <= p)
    //check the logic of this part, enforce data  (Daniel: I am not sure what this comment means?)
    state(ids, ?) = sample
    val innz = find(fdata)
    for (i <- 0 until batchSize * nsampls by batchSize) { // Keep "correcting" as we go
      state(innz + i * graph.n) = 0
      state(?, i until i + batchSize) = state(?, i until i + batchSize) + (fdata > 0)
    }
    predprobs(ids, ?) = p
  }

  /** Returns a CPT (conditional probability table). The index.nr = number of elements in pids. */
  def getCpt(index: IMat) = {
    var cptindex = zeros(index.nr, index.nc)
    for (i <- 0 until index.nc) {
      cptindex(?, i) = cpt(index(?, i))
    }
    cptindex
  }

  /** 
   * Updates the CPT after each sampling round, by using the "state" matrix that contains samples.
   * The "index" dictates the indices of "counts" that are incremented for each iteration. Note
   * that normcounts is the same as counts, except for every 2 consecutive spots, the elements are
   * flipped. Then normcounts + counts solves things and gets a correct normalizing constant.
   */
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
    cpt = (1 - alpha) * cpt + alpha * counts
  }

  /** Computes log likelihood of the data. Remember that "cpt" stores raw probabilities, not logarithms. */
  def eval = {
    val index = IMat(cptoffset + iproject * (state > 0))
    val ll = sum(sum(ln(getCpt(index)))) / nsampls
    llikelihood += ll.v
  }

  /**
   * After every iteration of Gibbs sampling, run it on the test data and print out result. Random
   * guessing is 50%; we are currently getting 65% for the best accuracy. If predprobs(i,j) = p,
   * then the Gibbs sampler thinks this question/student pair is 1 (i.e., student answered
   * correctly) with probability p. If it's at least 0.5, we pick true, else it's false.  We check
   * the true value in tdata. Accuracy is simply total correct divided by total possible.
   * I am not sure why we need to sample before doing this again, because we have already sampled...
   *
   * @param i The current iteration of Gibbs Sampling.
   */
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

  /** A debugging method to show the CPT of a node (i.e., a question or concept in the MOOC data). */
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

  /** 
   * Similar to sampleColor, but w/out pids, and deleting this method doesn't cause errors.
   * Keep this arond in case we want to sample but assume all other nodes are in the children set?
   */
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

  /** 
   * Loads the node file to create a node map. The node file has each line like:
   *   question_or_concept_ID,integer_index
   * Thus, we can form one entry of the map with one line in the file.
   *
   * @param path The file path to the node file (e.g, node.txt).
   * @return A map from strings (questions or concepts) to integers (0 to n-1).
   */
  def loadNodeMap(path: String) = {
    var nodeMap = new scala.collection.mutable.HashMap[String, Int]()
    var lines = scala.io.Source.fromFile(path).getLines
    for (l <- lines) {
      var t = l.split(",")
      nodeMap += (t(0) -> (t(1).toInt - 1))
    }
    nodeMap
  }

  /** 
   * Loads the dag file and converts it to an adjacency matrix so that we can create a graph object.
   * Each line consists of two nodes represented by Strings, so use the previously-formed nodeMap to
   * create the actual entries in the dag matrix.
   *
   * @param path The path to the dag file (e.g., dag.txt).
   * @return An adjacency matrix (of type SMat) for use to create a graph.
   */
  def loadDag(path: String, n: Int) = {
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
    var coordinatesMap = new scala.collection.mutable.HashMap[(Int,Int), Int]()
    val bufsize = 100000
    var row = izeros(bufsize, 1)
    var col = izeros(bufsize, 1)
    var v = zeros(bufsize, 1)
    var ptr = 0
    var sid = 0

    /* //val writer: PrintWriter = new PrintWriter(path + "/../sdata_new.txt")
    val writer: PrintWriter = new PrintWriter("sdata_new.txt")
    for(l <- lines) {
      val r = math.random
      if(r < 0.8) { // Huasha reported 0.8 so we'll use that
        writer.println(l + ",1")}
      else {
        writer.println(l + ",0")
      }
    }
    writer.close */
   
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
      if (t(4) == "1") {
        row(ptr) = sMap(shash)
        col(ptr) = nodeMap("I" + t(2))
        v(ptr) = (t(2).toFloat - 0.5) * 2
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
    var coordinatesMap = new scala.collection.mutable.HashMap[(Int,Int), Int]()
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
      if (t(4) == "0") {
        row(ptr) = sMap(shash)
        col(ptr) = nodeMap("I" + t(2))
        v(ptr) = (t(2).toFloat - 0.5) * 2
        ptr = ptr + 1
      }
    }
    var s = sparse(col(0 until ptr), row(0 until ptr), v(0 until ptr), nq, ns)
    (s > 0) - (s < 0)
  }
}









/**
* A graph structure for Bayesian Networks. Includes features for:
*
* (1) moralizing graphs, 'moral' matrix must be (i,j) = 1 means node i is connected to node j
* (2) coloring moralized graphs, not sure why there is a maxColor here, though...
*
* @param dag An adjacency matrix with a 1 at (i,j) if node i has an edge TOWARDS node j.
* @param n The number of vertices in the graph.
*/
class Graph1(val dag: SMat, val n: Int) {
    var mrf: FMat = null
    var colors: IMat = null
    var ncolors = 0
    val maxColor = 100
   /**
    * Connects the parents of a certain node, a single step in the process of moralizing the graph.
    *
    * Iterates through the parent indices and insert 1s in the 'moral' matrix to indicate an edge.
    *
    * TODO Is there a way to make this GPU-friendly?
    *
    * @param moral A matrix that represents an adjacency matrix "in progress" in the sense that it
    * is continually getting updated each iteration from the "moralize" method.
    * @param parents An array representing the parent indices of the node of interest.
    */
    def connectParents(moral: FMat, parents: IMat) = {
        val l = parents.length
        for(i <- 0 until l)
            for(j <- 0 until l){
                if(parents(i) != parents(j)){
                    moral(parents(i), parents(j)) = 1f
                }
            }
            moral
    }
    /**
    * TODO No idea what this is yet, and it isn't in the newgibbs branch. Only used here for
    * BayesNet.
    */
    def iproject = {
        var ip = dag.t
        for (i <- 0 until n) {
            val ps = find(dag(?, i))
            val np = ps.length
            for (j <-0 until np) {
                ip(i, ps(j)) = math.pow(2, np-j).toFloat
            }
        }
        ip + sparse(IMat(0 until n), IMat(0 until n), ones(1, n))
    }
    /**
    * TODO No idea what this is yet, and it isn't in the new gibbs branch. Only used here for
    * BayesNet.
    */
    def pproject = {
        dag + sparse(IMat(0 until n), IMat(0 until n), ones(1, n))
    }
    /**
    * Moralize the graph.
    *
    * This means we convert the graph from directed to undirected and connect parents of nodes in
    * the directed graph. First, copy the dag to the moral graph because all 1s in the dag matrix
    * are 1s in the moral matrix (these are adjacency matrices). For each node, find its parents,
    * connect them, and update the matrix. Then make it symmetric because the graph is undirected.
    *
    * TODO Is there a way to make this GPU-friendly?
    */
    def moralize = {
        var moral = full(dag)
        for (i <- 0 until n) {
            var parents = find(dag(?, i))
            moral = connectParents(moral, parents)
        }
        mrf = ((moral + moral.t) > 0)
    }
    /**
    * Sequentially colors the moralized graph of the dag so that one can run parallel Gibbs
    * sampling.
    *
    * Steps: first, moralize the graph. Then iterate through each node, find its neighbors, and
    * apply a
    * "color mask" to ensure current node doesn't have any of those colors. Then find the legal
    * color
    * with least count (a useful heuristic). If that's not possible, then increase "ncolor".
    *
    * TODO Is there a way to make this GPU-friendly?
    */
    def color = {
        moralize
        var colorCount = izeros(maxColor, 1)
        colors = -1 * iones(n, 1)
        ncolors = 0
        // Access nodes sequentially. Find the color map of its neighbors, then find the legal color
        // w/least count
        val seq = IMat(0 until n)
        // Can also access nodes randomly
        // val r = rand(n, 1); val (v, seq) = sort2(r)
        for (i <- 0 until n) {
            var node = seq(i)
            var nbs = find(mrf(?, node))
            var colorMap = iones(ncolors, 1)
            for (j <- 0 until nbs.length) {
                if (colors(nbs(j)) > -1) {
                    colorMap(colors(nbs(j))) = 0
                }
            }
            var c = -1
            var minc = 999999
            for (k <- 0 until ncolors) {
                if ((colorMap(k) > 0) && (colorCount(k) < minc)) {
                    c = k
                    minc = colorCount(k)
                }
            }
            if (c == -1) {
                c = ncolors
                ncolors = ncolors + 1
            }
            colors(node) = c
            colorCount(c) += 1
        }
        colors
    }
}
