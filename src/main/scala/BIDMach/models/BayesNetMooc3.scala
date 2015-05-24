package BIDMach.models

import BIDMat.{ CMat, CSMat, DMat, Dict, IDict, FMat, GMat, GIMat, GSMat, HMat, IMat, Mat, SMat, SBMat, SDMat }
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import BIDMach._
import BIDMach.datasources._
import BIDMach.updaters._
import java.io._
import scala.util.Random
import scala.collection.mutable._

/**
 * A Bayesian Network implementation with fast parallel Gibbs Sampling (e.g., for MOOC data).
 * 
 * Haoyu Chen and Daniel Seita are building off of Huasha Zhao's original code.
 * 
 * The input needs to be (1) a graph, (2) a sparse data matrix, and (3) a states-per-node file
 * 
 * This is still a WIP.
 */

// Put general reminders here:
// TODO Check if all these (opts.useGPU && Mat.hasCUDA > 0) tests are necessary.
// TODO Investigate opts.nsampls. For now, have to do batchSize * opts.nsampls or something like that.
// TODO Check if the BayesNet should be re-using the user for now
// To be honest, right now it's far easier to assume the datasource is providing us with a LENGTH ONE matrix each step
// TODO Be sure to check if we should be using SMats, GMats, etc. ANYWHERE we use Mats.
// TODO Worry about putback and multiple iterations LATER
// TODO Only concerning ourselves with SMats here, right?
class BayesNetMooc3(val dag:SMat, 
                    val states:Mat, 
                    override val opts:BayesNetMooc3.Opts = new BayesNetMooc3.Options) extends Model(opts) {

  var graph:Graph1 = null
  var mm:Mat = null
  var iproject:Mat = null
  var pproject:Mat = null
  var cptOffset:IMat = null
  var statesPerNode:IMat = IMat(states)
  var normConstMatrix:SMat = null
  
  /**
   * Performs a series of initialization steps.
   * 
   * - Builds iproject/pproject for local offsets and computing probabilities, respectively.
   * - Build the CPT and its normalizing constant matrix (though we might not use the latter)
   *   - Note that the CPT itself is a cpt of counts, which I initialize to 1 for now
   */
  def init() = {
    val useGPU = opts.useGPU && (Mat.hasCUDA > 0)
    graph = new Graph1(dag, opts.dim, statesPerNode)
    graph.color

    // TODO Problem is that GSMat has no transpose method! Use CPU Matrices for now!
    iproject = if (useGPU) GSMat(graph.iproject) else graph.iproject
    pproject = if (useGPU) GSMat(graph.pproject) else graph.pproject
   
    // Now start building the CPT by computing offset vector.
    val numSlotsInCpt = if (useGPU) {
      GIMat(exp(GMat((pproject.t)) * ln(GMat(statesPerNode))) + 1e-3) 
    } else {
      IMat(exp(DMat(full(pproject.t)) * ln(DMat(statesPerNode))) + 1e-3)     
    }
    cptOffset = izeros(graph.n, 1)
    if (useGPU) {
      cptOffset(1 until graph.n) = cumsum(GMat(numSlotsInCpt))(0 until graph.n-1)
    } else {
      cptOffset(1 until graph.n) = cumsum(IMat(numSlotsInCpt))(0 until graph.n-1)
    }

    // Build the CPT. For now, it stores counts. To avoid div by 0, initialize w/ones.
    val lengthCPT = sum(numSlotsInCpt).dv.toInt
    var cpt = rand(lengthCPT,1)
    //val cpt = ones(lengthCPT, 1)
    normConstMatrix = getNormConstMatrix(lengthCPT)
    cpt = cpt / (normConstMatrix * cpt)
    println("Our cpt.t =\n" + cpt.t)
    setmodelmats(new Array[Mat](1))
    modelmats(0) = if (useGPU) GMat(cpt) else cpt
    mm = modelmats(0)
    updatemats = new Array[Mat](1)
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols)
  }

  /**
   * Does a uupdate/mupdate on a datasource segment, called from dobatchg() in Model.scala. 
   * 
   * The data we get in mats(0) is such that 0s mean unknowns, and the knowns are {1, 2, ...}. The
   * purpose here is to run Gibbs sampling and use the resulting samples to "improve" the mm (i.e.,
   * the CPT). We randomize the state/user matrix here because we need to randomize the assignment
   * to all variables. TODO Should this only happen if ipass=0 and we have datasources put back?
   * 
   * @param mats An array of matrices representing a segment of the original data.
   * @param ipass The current pass over the data source (not the Gibbs sampling iteration number).
   * @param here The total number of elements seen so far, including the ones in this current batch.
   */
  def dobatch(mats:Array[Mat], ipass:Int, here:Long) = {
    val useGPU = opts.useGPU && (Mat.hasCUDA > 0)
    val sdata = mats(0)
    var state = rand(sdata.nrows, sdata.ncols * opts.nsampls) // Because 0 = unknown
    if (useGPU) {
      // TODO GMat not working?
      state = min( FMat(trunc(statesPerNode *@ state)) , statesPerNode-1)
    } else {
      state = min( FMat(trunc(statesPerNode *@ state)) , statesPerNode-1) 
    }
    val kdata = if (useGPU) { GMat(sdata)-1 } else { FMat(sdata)-1 }
    val positiveIndices = find(FMat(kdata) >= 0)
    for (i <- 0 until opts.nsampls) {
      state(positiveIndices + i*(kdata.nrows*kdata.ncols)) = kdata(positiveIndices)
    }
    uupdate(kdata, state, ipass)
    mupdate(kdata, state, ipass)
  }

  /**
   * Evaluates the current datasource; called from evalbatchg() in Model.scala.
   * 
   * TODO For now it seems like we may not even need a uupdate call. What often happens here is that
   * we do uupdate then evaluate, but if we have opts.updateAll then we will already have done a full
   * uupdate/mupdate call. And we are just evaluating the log likelihood of the cpt here, right? That
   * involves summing up the log of its normalized elements (use the normConstMatrix).
   * 
   * @param mats An array of matrices representing a segment of the original data.
   * @param ipass The current pass over the data source (not the Gibbs sampling iteration number).
   * @param here The total number of elements seen so far, including the ones in this current batch.
   * @return The log likelihood of the data on this datasource segment (i.e., sdata).
   */
  def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat = {
    println("Inside evalbatch, right now nothing here (debugging) but later we'll report a score.")
    return 1f
  }

  /**
   * Performs a complete, parallel Gibbs sampling over the color groups, for opts.uiter iterations over 
   * the current batch of data. For a given color group, we need to know the maximum state number. The
   * sampling process will assign new values to the "user" matrix.
   * 
   * @param kdata A data matrix of this batch where a -1 indicates an unknown value, and {0,1,...,k} are
   *    the known values. Each row represents a random variable.
   * @param user Another data matrix with the same number of rows as kdata, and whose columns represent
   *    various iid assignments to all the variables. The known values of kdata are inserted in the same
   *    spots in this matrix, but the unknown values are appropriately randomized to be in {0,1,...,k}.
   * @param ipass The current pass over the full data source (not the Gibbs sampling iteration number).
   */
  def uupdate(kdata:Mat, user:Mat, ipass:Int):Unit = {
    println("Inside uupdate with user = ")
    printMatrix(FMat(user))
    var numGibbsIterations = opts.samplingRate
    if (ipass == 0) {
      numGibbsIterations = numGibbsIterations + opts.numSamplesBurn
    }
    for (k <- 0 until numGibbsIterations) {
      println("In uupdate(), Gibbs iteration " + k + " of " + numGibbsIterations)
      for (n <- 0 until graph.n) {
        
        println("\nCURRENTLY ON NODE " + n)
        // firstIndices gives indices to the contiguous block of Pr(Xn = ? | parents).
        val assignment = user.copy
        assignment(n,?) = 0
        val numStates = statesPerNode(n)
        val globalOffset = cptOffset(n)
        val localOffset = SMat(iproject(n,?)) * FMat(assignment)
        val firstIndices = globalOffset + localOffset + (icol(0 until numStates) + zeros(numStates, localOffset.length))
        println("globalOffset = " + globalOffset)
        println("localOffset = " + localOffset)
        println(firstIndices)

        // Some book-keeping
        val childrenIDs = find(graph.dag(n,?)) 
        val predIndices = zeros(numStates, user.ncols * (childrenIDs.length+1))
        predIndices(?, (0 until user.ncols) ) = firstIndices
        println("predIndices = ") 
        printMatrix(FMat(predIndices))

        // Next, for EACH of its children, we have to compute appropriate vectors to add. Avoid looping over children.
        val globalOffsets = cptOffset(childrenIDs)
        val localOffsets = SMat(iproject)(childrenIDs,?) * FMat(assignment)
        val strides = SMat(iproject)(childrenIDs,n)

        println("localoffsets followed by strides:")
        printMatrix(localOffsets)
        printMatrix(FMat(strides))
        sys.exit
        
        for (i <- 0 until childrenIDs.length) {
          val child = childrenIDs(i)
          val globalOffset2 = cptOffset(child)
          val localOffset2 = SMat(iproject(child,?)) * FMat(assignment)
          val stride = SMat(iproject)(child, n) // Crucial
          val indices = globalOffset2 + localOffset2 + (stride * icol(0 until numStates))
          println("For child = " + child + ", goffset = " + globalOffset2 + ", loffset = " + localOffset2 + " stride = " + stride)
          println(indices)
          predIndices(?,i+1) = indices
        }
        
        // Combine all previously computed vectors, transpose it, then do matrix multiplication to compute probabilities
        println("After all has been concluded, predIndices and predIndices.t =")
        printMatrix(FMat(predIndices))
        printMatrix(FMat(predIndices.t))

        //val nodei = ln(getCpt(cptOffset(pids) + IMat(SMat(iproject(pids, ?)) * FMat(statei))) + opts.eps)
        val logProbs = ln(getCpt( predIndices.t )) // + opts.eps) 
        println("Log probs matrix =")
        printMatrix(FMat(logProbs))
        val result = exp( ones(1,childrenIDs.length+1) * logProbs )
        println("result matrix =")
        printMatrix(FMat(result))

        // results = a vector with length statesPerNode(n), which references all necessary probabilities, so now sample
        val resultNormalized = result / sum(result,2)
        // For each row, pick a number in [0,1], then check which range it goes in, and that is the value we sample for that row.
        // Result, btw, is a ROW vector, representing one row and the different values that row can take on!
        // TODO (though not difficult once we know what row we have I hope) If one row, can normalize across rows,
        // Of course, at the end, we have to override everything with the values we know!
      }
      sys.exit
      // Once we've sampled everything, accumulate counts in a CPT (but don't normalize).
      updateCPT(user)
    }
    
    /*
    for (k <- 0 until numGibbsIterations) {
      println("In uupdate, Gibbs iteration " + k)
      for (c <- 0 until graph.ncolors) {
        val idInColor = find(graph.colors == c)
        val numState = IMat(maxi(maxi(statesPerNode(idInColor),1),2)).v
        var stateSet = new Array[Mat](numState)
        var pSet = new Array[Mat](numState)
        var pMatrix = zeros(idInColor.length, kdata.ncols * opts.nsampls)
        for (i <- 0 until numState) {
          val saveID = find(statesPerNode(idInColor) > i)
          val ids = idInColor(saveID)
          val pids = find(FMat(sum(pproject(ids, ?), 1)))
          initStateColor(kdata, ids, i, stateSet, user)
          computeP(ids, pids, i, pSet, pMatrix, stateSet(i), saveID, idInColor.length)
        }
        sampleColor(kdata, numState, idInColor, pSet, pMatrix, user)
      } 
      updateCPT(user)
    }
    */
  }
  
  /**
   * After one full round of Gibbs sampling iterations, we put in the local cpt, mm, into the updatemats(0)
   * value so that it gets "averaged into" the global cpt, modelmats(0). The reason for doing this is that
   * it is like "thinning" the samples so we pick every n-th one, where n is an opts parameter. This also
   * works if we assume that modelmats(0) stores counts instead of normalized probabilities.
   * 
   * @param kdata A data matrix of this batch where a -1 indicates an unknown value, and {0,1,...,k} are
   *    the known values. Each row represents a random variable.
   * @param user Another data matrix with the same number of rows as kdata, and whose columns represent
   *    various iid assignments to all the variables. The known values of kdata are inserted in the same
   *    spots in this matrix, but the unknown values are appropriately randomized to be in {0,1,...,k}.
   * @param ipass The current pass over the full data source (not the Gibbs sampling iteration number).
   */
  def mupdate(kdata:Mat, user:Mat, ipass:Int):Unit = {
    println("In mupdate, mm(0 until 100).t=\n" + mm(0 until 100))
    updatemats(0) <-- mm
  } 
  
  /** 
   * Evaluates the log likelihood of this datasource segment (i.e., sdata). 
   * 
   * The sdata matrix has instances as columns, so one column is an assignment to variables. If each column
   * of sdata had a full assignment to all variables, then we easily compute log Pr(X1, ..., Xn) which is
   * the sum of elements in the CPT (i.e., the mm), sum(Xi | parents(Xi)). For the log-l of the full segment,
   * we need another addition around each column, so the full log-l is the sum of the log-l of each column.
   * This works because we assume each column is iid so we split up their joint probabilities. Taking logs 
   * then results in addition.
   * 
   * In reality, data is missing, and columns will not have a full assignment. That is where we incorporate
   * the user matrix to fill in the missing details. The user matrix fills in the unknown assignment values.
   * TODO Actually, since user contains all of sdata's known values, we DON'T NEED sdata, but we DO need to
   * make sure that "user" is correctly initialized...
   * 
   * @param sdata The data matrix, which may have missing assignments indicated by a -1.
   * @param user A matrix with a full assignment to all variables from sampling or randomization.
   * @return The log likelihood of the data.
   */
  def evalfun(sdata: Mat, user: Mat) : FMat = {
    val a = cptOffset + IMat(SMat(iproject) * FMat(user))
    val b = maxi(maxi(a,1),2).dv
    val index = IMat(cptOffset + SMat(iproject) * FMat(user))
    val ll = sum(sum(ln(getCpt(index))))
    return ll.dv
  }

  /**
   * Method to update the cpt table (i.e. mm). This method is called after we finish one iteration of Gibbs 
   * sampling. And this method only updates the local cpt table (mm), it has nothing to do with the learner's 
   * cpt parameter (which is modelmats(0)).
   * 
   * @param user The state matrix, updated after the sampling.
   */
  def updateCPT(user: Mat): Unit = {
    val numCols = size(user, 2)
    val index = IMat(cptOffset + SMat(iproject) * FMat(user))
    var counts = zeros(mm.length, 1)
    for (i <- 0 until numCols) {
      counts(index(?, i)) = counts(index(?, i)) + 1
    }
    counts = counts + opts.alpha  
    //mm = (1 - opts.alpha) * mm + counts / (normConstMatrix * counts)
    mm = counts // For now, in case we just want raw counts
  }

  /**
   * Initializes the statei matrix for this particular color group and for this particular value. It
   * fills in the unknown values at the ids locations with i, then we can use it in computeP. I.e.,
   * the purpose of this is to make future probability computations/lookups efficient.
   * 
   * @param kdata Training data matrix, with unknowns of -1 and known values in {0,1,...,k}.
   * @param ids Indices of nodes in this color group that can also attain value/state i.
   * @param i An integer representing a value/state (we use these terms interchangeably).
   * @param stateSet An array of statei matrices, each of which has "i" in the unknowns of "ids".
   * @param user A data matrix with the same rows as kdata, and whose columns represent various iid
   *    assignments to all the variables. The known values of kdata are inserted in the same spots in
   *    this matrix, but the unknown values are appropriately randomized to be in {0,1,...,k}.
   */
  def initStateColor(kdata: Mat, ids: IMat, i: Int, stateSet: Array[Mat], user:Mat) = {
    var statei = user.copy
    statei(ids,?) = i
    val nonNegativeIndices = find(FMat(kdata) >= 0)
    for (i <- 0 until opts.nsampls) {
      statei(nonNegativeIndices + i*(kdata.nrows*kdata.ncols)) <-- kdata(nonNegativeIndices)
    }
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
  def computeP(ids: IMat, pids: IMat, i: Int, pSet: Array[Mat], pMatrix: Mat, statei: Mat, saveID: IMat, numPi: Int) = {
    val nodei = ln(getCpt(cptOffset(pids) + IMat(SMat(iproject(pids, ?)) * FMat(statei))) + opts.eps)
    var pii = zeros(numPi, statei.ncols)
    pii(saveID, ?) = exp(SMat(pproject(ids, pids)) * FMat(nodei))
    pSet(i) = pii
    pMatrix(saveID, ?) = pMatrix(saveID, ?) + pii(saveID, ?)
  }

  /** 
   * For a given color group, after we have gone through all its state possibilities, we sample it.
   * 
   * To start, we use a matrix of random values. Then, we go through each of the possible states and
   * if random values fall in a certain range, we assign the range's corresponding integer {0,1,...,k}.
   * 
   * @param fdata Training data matrix, with unknowns of -1 and known values in {0,1,...,k}.
   * @param numState The maximum number of state/values possible of any variable in this color group.
   * @param idInColor Indices of nodes in this color group.
   * @param pSet The array of matrices, each of which represents probabilities of nodes attaining i.
   * @param pMatrix The matrix that represents normalizing constants for probabilities.
   * @param user A matrix with a full assignment to all variables from sampling or randomization.
   */
  def sampleColor(kdata: Mat, numState: Int, idInColor: IMat, pSet: Array[Mat], pMatrix: Mat, user: Mat) = {
    val sampleMatrix = rand(idInColor.length, kdata.ncols * opts.nsampls)
    pSet(0) = pSet(0) / pMatrix
    //user(idInColor,?) <-- 0 * user(idInColor,?) // THIS IS NOT THE SAME!
    user(idInColor,?) = 0 * user(idInColor,?)
    
    // Each time, we check to make sure it's <= pSet(i), but ALSO exceeds the previous \sum (pSet(j)).
    for (i <- 1 until numState) {
      val saveID = find(statesPerNode(idInColor) > i)
      val ids = idInColor(saveID)
      val pids = find(FMat(sum(pproject(ids, ?), 1)))
      pSet(i) = (pSet(i) / pMatrix) + pSet(i-1) // Normalize and get the cumulative prob
      // Use Hadamard product to ensure that both requirements are held.
      user(ids, ?) = user(ids,?) + i * ((sampleMatrix(saveID, ?) <= pSet(i)(saveID, ?)) *@ (sampleMatrix(saveID, ?) >= pSet(i - 1)(saveID, ?)))
    }

    // Finally, re-write the known state into the state matrix
    val nonNegativeIndices = find(FMat(kdata) >= 0)
    for (j <- 0 until opts.nsampls) {
      user(nonNegativeIndices + j*(kdata.nrows*kdata.ncols)) = kdata(nonNegativeIndices)
    }
  }

  /**
   * Creates normalizing matrix N that we can then multiply with the cpt, i.e., N * cpt, to get a column
   * vector of the same length as the cpt, but such that cpt / (N * cpt) is normalized. Use SMat to save
   * on memory, I think.
   */
  def getNormConstMatrix(cptLength : Int) : SMat = {
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
    while (offsetLast < cptLength) {
      ii = ii on (indices._1 + offsetLast)
      jj = jj on (indices._2 + offsetLast)
      offsetLast = offsetLast + statesPerNode(graph.n-1)
    }
    return sparse(ii(1 until ii.length), jj(1 until jj.length), ones(ii.length-1, 1), cptLength, cptLength)
  }

  /** Returns a conditional probability table specified by indices from the "index" matrix. */
  def getCpt(index: Mat) = {
    var cptindex = mm.zeros(index.nrows, index.ncols)
    for(i <-0 until index.ncols){
      cptindex(?, i) = mm(IMat(index(?, i)))
    }
    cptindex
  }

  /** Returns FALSE if there's an element at least size 2, which is BAD (well, only for binary data...). */
  def checkState(state: Mat) : Boolean = {
    val a = maxi(maxi(state,2),1).dv
    if (a >= 2) {
      return false
    }
    return true
  }

  /** A debugging method to print matrices, without being constrained by the command line's cropping. */
  def printMatrix(mat: FMat) = {
    for(i <- 0 until mat.nrows) {
      for (j <- 0 until mat.ncols) {
        print(mat(i,j) + " ")
      }
      println()
    }
  }
  
}



/**
 * There are three things the BayesNet needs as input:
 * 
 *  - A states per node array. Each value needs to be an integer that is at least two.
 *  - A DAG array, in which column i represents node i and its parents.
 *  - A sparse data matrix, where 0 indicates an unknown element, and rows are variables.
 * 
 * We don't need anything else for now, except for the number of gibbs iterations as input.
 */
object BayesNetMooc3  {
  
  trait Opts extends Model.Opts {
    var nsampls = 1
    var alpha = 1f
    var beta = 0.1f
    var samplingRate = 1
    var eps = 1e-9
    var numSamplesBurn = 100
  }
  
  class Options extends Opts {}

  /** 
   * A learner with a matrix data source, with states per node, and with a dag prepared. Call this
   * using (assuming proper names):
   * 
   * val (nn,opts) = BayesNetMooc3.learner(loadIMat("states.lz4"), loadSMat("dag.lz4"), loadSMat("sdata.lz4"))
   */
  def learner(statesPerNode:Mat, dag:Mat, data:Mat) = {

    class xopts extends Learner.Options with BayesNetMooc3.Opts with MatDS.Opts with IncNorm.Opts 
    val opts = new xopts
    opts.dim = dag.ncols
    opts.batchSize = 4367
    opts.isprob = false     // Because our cpts should NOT be normalized across their one column (lol).
    opts.useGPU = false     // Temporary TODO test with GPUs, delete this line later
    opts.updateAll = true   // Temporary TODO (this means that we do the same IncNorm update even with "evalfun"
    //opts.power = 1f         // TODO John suggested that we do not need 1f here
    opts.putBack = 1        // Temporary TODO because we will assume we have npasses = 1 for now
    opts.npasses = 1        // Temporary TODO because I would like to get one pass working correctly.

    val nn = new Learner(
        new MatDS(Array(data:Mat), opts),
        new BayesNetMooc3(SMat(dag), statesPerNode, opts),
        null,
        new IncNorm(opts),
        opts)
    (nn, opts)
  }
}



/**
 * A graph structure for Bayesian Networks. Includes features for:
 * 
 *   (1) moralizing graphs, 'moral' matrix must be (i,j) = 1 means node i is connected to node j
 *   (2) coloring moralized graphs, not sure why there is a maxColor here, though...
 *
 * @param dag An adjacency matrix with a 1 at (i,j) if node i has an edge TOWARDS node j.
 * @param n The number of vertices in the graph. 
 * @param statesPerNode A column vector where elements denote number of states for corresponding variables.
 */
// TODO making this Graph1 instead of Graph so that we don't get conflicts
// TODO investigate all non-Mat matrices (IMat, FMat, etc.) to see if these can be Mats to make them usable on GPUs
// TODO Also see if connectParents, moralize, and color can be converted to GPU friendly code
class Graph1(val dag: SMat, val n: Int, val statesPerNode: IMat) {
 
  var mrf: FMat = null
  var colors: IMat = null
  var ncolors = 0
  val maxColor = 100
   
  /**
   * Connects the parents of a certain node, a single step in the process of moralizing the graph.
   * 
   * Iterates through the parent indices and insert 1s in the 'moral' matrix to indicate an edge.
   * 
   * @param moral A matrix that represents an adjacency matrix "in progress" in the sense that it
   *    is continually getting updated each iteration from the "moralize" method.
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

  /** Forms the pproject matrix (graph + identity) used for computing model parameters. */
  def pproject = {
    dag + sparse(IMat(0 until n), IMat(0 until n), ones(1, n))
  }
  
  /**
   * Forms the iproject matrix, which is left-multiplied to send a Pr(X_i | parents) query to its
   * appropriate spot in the cpt via LOCAL offsets for X_i.
   */
  def iproject = {
    var res = (pproject.copy).t
    for (i <- 0 until n) {
      val parents = find(pproject(?, i))
      var cumRes = 1
      val parentsLen = parents.length
      for (j <- 1 until parentsLen) {
        cumRes = cumRes * statesPerNode(parents(parentsLen - j))
        res(i, parents(parentsLen - j - 1)) = cumRes.toFloat
      }
    }
    res
  }
  
  /**
   * Moralize the graph.
   * 
   * This means we convert the graph from directed to undirected and connect parents of nodes in 
   * the directed graph. First, copy the dag to the moral graph because all 1s in the dag matrix
   * are 1s in the moral matrix (these are adjacency matrices). For each node, find its parents,
   * connect them, and update the matrix. Then make it symmetric because the graph is undirected.
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
   * Sequentially colors the moralized graph of the dag so that one can run parallel Gibbs sampling.
   * 
   * Steps: first, moralize the graph. Then iterate through each node, find its neighbors, and apply a
   * "color mask" to ensure current node doesn't have any of those colors. Then find the legal color
   * with least count (a useful heuristic). If that's not possible, then increase "ncolor".
   */
  def color = {
    moralize
    var colorCount = izeros(maxColor, 1)
    colors = -1 * iones(n, 1)
    ncolors = 0
   
    // Access nodes sequentially. Find the color map of its neighbors, then find the legal color w/least count
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

