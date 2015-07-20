package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._

import java.text.NumberFormat
import edu.berkeley.bid.CUMACH._
import scala.collection.mutable._

/**
 * A Bayesian Network implementation with fast parallel Gibbs Sampling (e.g., for MOOC data).
 * 
 * Haoyu Chen and Daniel Seita are building off of Huasha Zhao's original code.
 * 
 * The input needs to be (1) a graph, (2) a sparse data matrix, and (3) a states-per-node file.
 * Make sure the dag and states files are aligned, and that variables are in a topological ordering!
 */
class BayesNet(val dag:Mat, 
               val states:Mat, 
               override val opts:BayesNet.Opts = new BayesNet.Options) extends Model(opts) {

  var mm:Mat = null                         // Local copy of the cpt
  var cptOffset:Mat = null                  // For global variable offsets
  var graph:Graph = null                    // Data structure representing the DAG
  var iproject:Mat = null                   // Local CPT offset matrix
  var pproject:Mat = null                   // Parent tracking matrix
  var statesPerNode:Mat = null              // Variables can have an arbitrary number of states
  var colorInfo:Array[ColorGroup] = null    // Gives us, for each color, a colorStuff class (of arrays)
  var zeroMap:HashMap[(Int,Int),Mat] = null // Map from (nr,nc) -> a zero matrix (to avoid allocation)
  var randMap:HashMap[(Int,Int),Mat] = null // Map from (nr,nc) -> a rand matrix (to avoid allocation)
  var normConstMatrix:Mat = null            // Normalizes the cpt. Do cpt / (cpt.t * nConstMat).t
  var useGPUnow:Boolean = false
  var batchSize:Int = -1
  var counts:Mat = null

  /**
   * Performs a series of initialization steps.
   * 
   * - Builds iproject/pproject for local offsets and computing probabilities, respectively.
   * - For each color group, determine some necessary matrices for uupdate later.
   * - Build the CPT, which is actually counts, not probabilities. I initialize it randomly.
   * 
   * Note that the randomization of the input data to be put back in the data is done in uupdate.
   */
  override def init() = {
    useGPUnow = opts.useGPU && (Mat.hasCUDA > 0)

    // Establish the states per node, the (colored) Graph data structure, and its projection matrices.
    statesPerNode = IMat(states)
    graph = new Graph(dag, opts.dim, statesPerNode)
    graph.color
    iproject = if (useGPUnow) GSMat((graph.iproject).t) else (graph.iproject).t
    pproject = if (useGPUnow) GSMat(graph.pproject) else graph.pproject
    
    // Build the CPT. For now, it stores counts, and to avoid div-by-zero errors, initialize randomly.
    val numSlotsInCpt = IMat(exp(ln(FMat(statesPerNode).t) * SMat(pproject)) + 1e-4)
    cptOffset = izeros(graph.n, 1)
    cptOffset(1 until graph.n) = cumsum(numSlotsInCpt)(0 until graph.n-1)
    cptOffset = convertMat(cptOffset)
    val lengthCPT = sum(numSlotsInCpt).dv.toInt
    val cpt = convertMat(rand(lengthCPT,1))
    
    // To finish building CPT, we normalize it based on the batch size and normalizing constant matrix.
    normConstMatrix = getNormConstMatrix(lengthCPT)
    cpt <-- ( cpt / (cpt.t * normConstMatrix).t )
    val mats = datasource.next
    cpt <-- (cpt * mats(0).ncols)
    datasource.reset
    setmodelmats(new Array[Mat](1))
    modelmats(0) = cpt
    mm = modelmats(0)
    updatemats = new Array[Mat](1)
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols)

    // For each color group, pre-compute most relevant matrices we need later (this does a lot!)
    colorInfo = new Array[ColorGroup](graph.ncolors)
    for (c <- 0 until graph.ncolors) {
      colorInfo(c) = computeAllColorGroupInfo(c)
    }
    zeroMap = new HashMap[(Int,Int),Mat]()
    randMap = new HashMap[(Int,Int),Mat]()

    // Finally, create/convert a few matrices, reset some variables, and add some debugging info
    counts = mm.izeros(mm.length, 1)
    statesPerNode = convertMat(statesPerNode) 
    batchSize = -1
  } 
   
  /** Calls a uupdate/mupdate sequence. Known data is in gmats(0), sampled data is in gmats(1). */
  override def dobatch(gmats:Array[Mat], ipass:Int, here:Long) = {
    uupdate(gmats(0), gmats(1), ipass)
    mupdate(gmats(0), gmats(1), ipass)
  }
  
  /** Calls a uupdate/evalfun sequence. Known data is in gmats(0), sampled data is in gmats(1). */
  override def evalbatch(gmats:Array[Mat], ipass:Int, here:Long):FMat = {
    uupdate(gmats(0), gmats(1), ipass)
    evalfun(gmats(0), gmats(1))
  }
 
  /**
   * Computes an update for the conditional probability table by sampling each variable once (for now).
   * 
   * In the first ipass, it randomizes the user matrix except for those values are already known from
   * sdata. It also establishes various matrices to be put in the colorInfo array or the hash maps (for
   * caching purposes). For each data batch, it iterates through color groups and samples in parallel.
   * 
   * @param sdata The sparse data matrix for this batch (0s = unknowns), which the user matrix shifts by -1.
   * @param user A data matrix with the same dimensions as sdata, and whose columns represent various iid
   *    assignments to all the variables. The known values of sdata are inserted in the same spots in this
   *    matrix, but the unknown values are randomized to be in {0,1,...,k}.
   * @param ipass The current pass over the full data source (not the Gibbs sampling iteration number).
   */
  def uupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    var numGibbsIterations = opts.samplingRate
    
    // For the first pass, we need to create a lot of matrices that rely on knowledge of the batch size.
    if (ipass == 0) {
      numGibbsIterations = numGibbsIterations + opts.numSamplesBurn
      establishMatrices(sdata.ncols)
      val state = convertMat(rand(sdata.nrows, sdata.ncols))
      state <-- float( min( int(statesPerNode ∘ state), statesPerNode-1 ) ) // Need an extra float() outside
      val data = full(sdata)
      val select = data > 0
      user ~ (select ∘ (data-1)) + ((1-select) ∘ state)
    }
    val usertrans = user.t

    for (k <- 0 until numGibbsIterations) {
      for (c <- 0 until graph.ncolors) {

        // Prepare our data by establishing the appropriate offset matrices for the entire CPT blocks
        usertrans(?, colorInfo(c).idsInColor) = zeroMap( (usertrans.nrows, colorInfo(c).numNodes) )
        val offsetMatrix = usertrans * colorInfo(c).iprojectSliced + (colorInfo(c).globalOffsetVector).t
        val replicatedOffsetMatrix = int(offsetMatrix * colorInfo(c).replicationMatrix) + colorInfo(c).strideVector
        val logProbs = ln(mm(replicatedOffsetMatrix))
        val nonExponentiatedProbs = (logProbs * colorInfo(c).combinationMatrix).t
        
        // Establish matrices needed for the multinomial sampling
        val keys = if (user.ncols == batchSize) colorInfo(c).keysMatrix else colorInfo(c).keysMatrixLast
        val bkeys = if (user.ncols == batchSize) colorInfo(c).bkeysMatrix else colorInfo(c).bkeysMatrixLast
        val bkeysOff = if (user.ncols == batchSize) colorInfo(c).bkeysOffsets else colorInfo(c).bkeysOffsetsLast
        val randIndices = if (user.ncols == batchSize) colorInfo(c).randMatrixIndices else colorInfo(c).randMatrixIndicesLast
        val sampleIndices = if (user.ncols == batchSize) colorInfo(c).sampleIDindices else colorInfo(c).sampleIDindicesLast
        
        // Parallel multinomial sampling. Check the colorInfo matrices since they contain a lot of info.
        val maxInGroup = cummaxByKey(nonExponentiatedProbs, keys)(bkeys)
        val probs = exp(nonExponentiatedProbs - maxInGroup)
        val cumprobs = cumsumByKey(probs, keys)
        val normedProbs = cumprobs / cumprobs(bkeys)
        
        // With cumulative probabilities set up in normedProbs matrix, create a random matrix and sample
        val randMatrix = randMap( (colorInfo(c).numNodes, usertrans.nrows) )
        rand(randMatrix)
        randMatrix <-- randMatrix * 0.99999f
        val lessThan = normedProbs < randMatrix(randIndices)
        val sampleIDs = cumsumByKey(lessThan, keys)(sampleIndices)
        usertrans(?, colorInfo(c).idsInColor) = sampleIDs.t
        
        // After we finish sampling with this color group, we override the known values.
        val data = full(sdata)
        val select = data > 0
        usertrans ~ (select *@ (data-1)).t + ((1-select) *@ usertrans.t).t
      }     
    }

    user <-- usertrans.t
  }

  /**
   * After one set of Gibbs sampling iterations, we put the sampled counts in the updatemats(0)
   * value so that it gets "averaged into" the cpt (mm or modelmats(0)), from the IncNorm updater.
   * Also, if the uupdate call involved more than one Gibbs sampling iterations, then the mupdate 
   * effectively "thins" the sampler by only taking results from every n^th^ sample.
   * 
   * @param sdata The sparse data matrix for this batch (0s = unknowns), which we do not use here.
   * @param user A data matrix with the same dimensions as sdata, and whose columns represent various
   *    iid assignments to all the variables. The known values of sdata are inserted in the same spots
   *    in this matrix, but the unknown values are randomized to be in {0,1,...,k}.
   * @param ipass The current pass over the full data source (not the Gibbs sampling iteration number).
   */
  def mupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    val index = int(cptOffset + (user.t * iproject).t)
    val linearIndices = index(?)
    counts <-- accum(linearIndices, 1, counts.length, 1)
    updatemats(0) <-- (float(counts) + opts.alpha)
  }
 
  /**
   * Evaluates the log-likelihood of the data (per column, or per full assignment of all variables).
   * First, we get the index matrix, which indexes into the CPT for each column's variable assignment.
   * Then, using the normalized CPT, we find the log probabilities of the user matrix, and sum
   * vertically (i.e., each variable, valid due to derived rules) and then horizontally (i.e., each
   * sample, which can do since we assume i.i.d.). 
   */
  def evalfun(sdata:Mat, user:Mat):FMat = {  
    val index = int(cptOffset + (user.t * iproject).t)
    val cptNormalized = mm / (mm.t * normConstMatrix).t
    val result = FMat( sum(sum(ln(cptNormalized(index)),1),2) ) / user.ncols
    return result
  }
  
  // -----------------------------------
  // Various debugging or helper methods
  // -----------------------------------

  /** 
   * Determines a variety of information for this color group, and stores it in a ColorGroup object. 
   * First, it establishes some basic information from each color group. Then it computes the more
   * complicated replication matrices, stride vectors, and combination matrices. Check the colorInfo
   * class for details on what the individual matrices represent.
   * 
   * @param c The integer index of the given color group.
   */
  def computeAllColorGroupInfo(c:Int) : ColorGroup = {
    val cg = new ColorGroup 
    cg.idsInColor = find(IMat(graph.colors) == c)
    cg.numNodes = cg.idsInColor.length
    cg.chIdsInColor = find(FMat(sum(SMat(pproject)(cg.idsInColor,?),1)))
    cg.numNodesCh = cg.chIdsInColor.length
    cg.iprojectSliced = SMat(iproject)(?,cg.chIdsInColor)
    cg.globalOffsetVector = convertMat(FMat(cptOffset(cg.chIdsInColor))) // Need FMat to avoid GMat+GIMat
    val startingIndices = izeros(cg.numNodes,1)
    startingIndices(1 until cg.numNodes) = cumsum(IMat(statesPerNode(cg.idsInColor)))(0 until cg.numNodes-1)
    cg.startingIndices = convertMat(startingIndices)

    // Gather useful information for determining the replication, stride, and combination matrices
    var ncols = 0
    val numOnes = izeros(1,cg.numNodesCh)       // Determine how many 1s to have
    val strideFactors = izeros(1,cg.numNodesCh) // Get stride factors for the stride vector
    val parentOf = izeros(1,cg.numNodesCh)      // Get index of parent (or itself) in idsInColor
    val fullIproject = full(iproject)
    for (i <- 0 until cg.numNodesCh) {
      var nodeIndex = cg.chIdsInColor(i).dv.toInt
      if (IMat(cg.idsInColor).data.contains(nodeIndex)) { // This node is in the color group
        numOnes(i) = statesPerNode(nodeIndex)
        ncols = ncols + statesPerNode(nodeIndex).dv.toInt
        strideFactors(i) = 1
        parentOf(i) = IMat(cg.idsInColor).data.indexOf(nodeIndex)
      } else { // This node is a child of a node in the color group
        val parentIndices = find( FMat( sum(SMat(pproject)(?,nodeIndex),2) ) )
        var parentIndex = -1
        var k = 0
        while (parentIndex == -1 && k < parentIndices.length) {
          if (IMat(cg.idsInColor).data.contains(parentIndices(k))) {
            parentIndex = parentIndices(k)
            parentOf(i) = IMat(cg.idsInColor).data.indexOf(parentIndices(k))
          }
          k = k + 1
        }
        if (parentIndex == -1) {
          throw new RuntimeException("Node at index " +nodeIndex+ " is missing a parent in its color group.")
        }
        numOnes(i) = statesPerNode(parentIndex)
        ncols = ncols + statesPerNode(parentIndex).dv.toInt
        strideFactors(i) = fullIproject(parentIndex,IMat(nodeIndex)).dv.toInt
      }
    }
    
    // Form the replication (the dim is (#-of-ch_id-variables x ncols)) and stride matrices
    var col = 0
    val strideVector = izeros(1, ncols)
    val ii = izeros(ncols, 1)
    for (i <- 0 until cg.numNodesCh) {
      val num = numOnes(i)
      ii(col until col+num) = i
      strideVector(col until col+num) = (0 until num)*strideFactors(i)
      col = col + num
    }
    val jj = icol(0 until ncols)
    val vv = ones(ncols, 1)
    cg.strideVector = convertMat(strideVector)
    cg.replicationMatrix = if (useGPUnow) GSMat(sparse(ii,jj,vv)) else sparse(ii,jj,vv) 
    
    // Form keys and ikeys vectors
    val numStatesIds = statesPerNode(cg.idsInColor)
    val ncolsCombo = sum(numStatesIds).dv.toInt
    val keys = izeros(1, ncolsCombo)
    val scaledKeys = izeros(1, ncolsCombo)
    val ikeys = izeros(1, cg.numNodes)
    var keyIndex = 0
    for (i <- 0 until cg.numNodes) {
      val nodeIndex = cg.idsInColor(i)
      val numStates = statesPerNode(nodeIndex).dv.toInt
      keys(keyIndex until keyIndex+numStates) = nodeIndex * iones(1,numStates)
      scaledKeys(keyIndex until keyIndex+numStates) = i * iones(1,numStates)
      keyIndex += numStates
      ikeys(i) = keyIndex-1
    }
    cg.scaledKeys = convertMat(scaledKeys)
    cg.keys = convertMat(keys)
    cg.ikeys = convertMat(ikeys)
    cg.bkeys = cg.ikeys(cg.scaledKeys)

    // Form the combination matrix (# of rows is # of columns of replication matrix)
    val indicesColumns = izeros(1,cg.numNodes)
    indicesColumns(1 until cg.numNodes) = cumsum(numStatesIds.asInstanceOf[IMat])(0 until cg.numNodes-1)
    val nrowsCombo = ncols
    val indicesRows = izeros(1,cg.numNodesCh)
    indicesRows(1 until cg.numNodesCh) = cumsum(numOnes)(0 until numOnes.length-1)
    val iii = izeros(nrowsCombo,1)
    val jjj = izeros(nrowsCombo,1)
    val vvv = ones(nrowsCombo,1)
    for (i <- 0 until cg.numNodesCh) {
      val p = parentOf(i) // Index into the node itself or its parent if it isn't in the color group
      iii(indicesRows(i) until indicesRows(i)+numOnes(i)) = indicesRows(i) until indicesRows(i)+numOnes(i)
      jjj(indicesRows(i) until indicesRows(i)+numOnes(i)) = indicesColumns(p) until indicesColumns(p)+numOnes(i)
    }
    cg.combinationMatrix = if (useGPUnow) {
      GSMat(sparse(iii,jjj,vvv,nrowsCombo,ncolsCombo))
    } else {
      sparse(iii,jjj,vvv,nrowsCombo,ncolsCombo)
    }
    
    cg.idsInColor = convertMat(cg.idsInColor)
    cg.chIdsInColor = convertMat(cg.chIdsInColor)
    if (useGPUnow) {
      cg.iprojectSliced = GSMat(cg.iprojectSliced.asInstanceOf[SMat])
    }
    return cg
  }
  
  /**
   * Called during the first pass over the data to set up matrices for later. These matrices are
   * used in future uupdate calls, and they depend on the batch size, hence why we can only form
   * these during the pass over the data, and not in init().
   * 
   * There are several types of matrices we create:
   * 
   *  - zero matrices to put in zeroMap, for clearing out usertrans
   *  - "rand" matries to put in randMap, for containers to randomize values during sampling
   *  - five colorInfo(c) matrices for the purposes of sampling
   * 
   * In the very likely case that the last batch does not have the same number of columns as the
   * first n-1 batches, then we need to repeat this process for that batch.
   * 
   * @param ncols The number of columns in the current data, or the batch size.
   */
  def establishMatrices(ncols:Int) = {
    if (batchSize == -1) { // Only true if we're on the first mini-batch of ipass = 0.
      batchSize = ncols
      val onesVector = mm.ones(1, ncols)
      val untilVector = convertMat( float(0 until ncols) )
      for (c <- 0 until graph.ncolors) {
        val numVars = colorInfo(c).numNodes
        val randOffsets = int(untilVector * numVars)
        zeroMap += ((ncols,numVars) -> mm.zeros(ncols,numVars))
        randMap += ((numVars,ncols) -> mm.zeros(numVars,ncols))
        colorInfo(c).keysMatrix = (colorInfo(c).keys).t * onesVector
        colorInfo(c).bkeysOffsets = int(untilVector * colorInfo(c).keys.ncols)
        colorInfo(c).bkeysMatrix = int(colorInfo(c).bkeys.t * onesVector) + colorInfo(c).bkeysOffsets
        colorInfo(c).randMatrixIndices = int((colorInfo(c).scaledKeys).t * onesVector) + randOffsets
        colorInfo(c).sampleIDindices = int((colorInfo(c).ikeys).t * onesVector) + colorInfo(c).bkeysOffsets
      }
    } 
    else if (ncols != batchSize) { // On the last batch of ipass = 0 w/different # of columns
      val onesVectorLast = mm.ones(1, ncols)
      val untilVectorLast = convertMat( float(0 until ncols) )
      for (c <- 0 until graph.ncolors) {
        val numVars = colorInfo(c).numNodes
        val randOffsets = int(untilVectorLast * numVars)
        zeroMap += ((ncols,numVars) -> mm.zeros(ncols,numVars))
        randMap += ((numVars,ncols) -> mm.zeros(numVars,ncols))
        colorInfo(c).keysMatrixLast = (colorInfo(c).keys).t * onesVectorLast
        colorInfo(c).bkeysOffsetsLast = int(untilVectorLast * colorInfo(c).keys.ncols)
        colorInfo(c).bkeysMatrixLast = int(colorInfo(c).bkeys.t * onesVectorLast) + colorInfo(c).bkeysOffsetsLast
        colorInfo(c).randMatrixIndicesLast = int((colorInfo(c).scaledKeys).t * onesVectorLast) + randOffsets
        colorInfo(c).sampleIDindicesLast = int((colorInfo(c).ikeys).t * onesVectorLast) + colorInfo(c).bkeysOffsetsLast
      }
    }
  }
 
  /**
   * Creates normalizing matrix N that we can then multiply with the CPT to get a column vector
   * of the same length as the CPT but such that it has normalized probabilties, not counts.
   * 
   * Specific usage: our CPT is a column vector of counts. To normalize and get probabilities, use
   *   CPT / (CPT.t * N).t
   *   
   * Alternatively, one could avoid those two transposes by making CPT a row vector, but since the
   * code assumes it's a column vector, it makes sense to maintain that convention.
   */
  def getNormConstMatrix(cptLength : Int) : Mat = {
    var ii = izeros(1,1)
    var jj = izeros(1,1)
    for (i <- 0 until graph.n-1) {
      var offset = IMat(cptOffset)(i)
      val endOffset = IMat(cptOffset)(i+1)
      val ns = statesPerNode(i).dv.toInt
      var indices = find2(ones(ns,ns))
      while (offset < endOffset) {
        ii = ii on (indices._1 + offset)
        jj = jj on (indices._2 + offset)
        offset = offset + ns
      }
    }
    var offsetLast = IMat(cptOffset)(graph.n-1)
    var indices = find2(ones(statesPerNode.asInstanceOf[IMat](graph.n-1), statesPerNode.asInstanceOf[IMat](graph.n-1)))
    while (offsetLast < cptLength) {
      ii = ii on (indices._1 + offsetLast)
      jj = jj on (indices._2 + offsetLast)
      offsetLast = offsetLast + statesPerNode.asInstanceOf[IMat](graph.n-1)
    }
    val res = sparse(ii(1 until ii.length), jj(1 until jj.length), ones(ii.length-1,1), cptLength, cptLength)
    if (useGPUnow) { // Note that here we have to transpose!
      return GSMat(res.t) 
    } else {
      return res.t
    }
  }
  
  /** A debugging method to print matrices, without being constrained by the command line's cropping. */
  def printMatrix(mat: Mat) = {
    for(i <- 0 until mat.nrows) {
      for (j <- 0 until mat.ncols) {
        print(mat(IMat(i),IMat(j)) + " ")
      }
      println()
    }
  } 
  
  /** A debugging method to print out the CPT of one variable (prettily). */
  def showCpt(nodeID: Int) {
    println("\nCPT for node indexed at " + nodeID)
    val startingOffset = cptOffset(nodeID)
    val numStates = statesPerNode(nodeID).dv.toInt
    val normalizedCPT = ( mm / (mm.t * normConstMatrix).t )
    val parentIndices = find(SMat(graph.dag)(?,nodeID))
    println("Parents: " + parentIndices.t)
    
    if (parentIndices.length == 0) {
      var str = "\t"
      for (j <- 0 until numStates) {
        str += " %.4f".format(normalizedCPT(startingOffset + j).dv)
      }
      println(str)
    } else {
      val totalParentSlots = prod(IMat(statesPerNode)(parentIndices)).dv.toInt
      val parentStates = statesPerNode(parentIndices)
      val statesList = izeros(1,parentIndices.length)
      var currentOffset = startingOffset
      for (i <- 0 until totalParentSlots) {
        if (i > 0) updateStatesString(statesList, parentStates, parentIndices.length-1)
        var str = ""
        for (i <- 0 until statesList.length) {
          str += statesList(i).dv.toInt + " "
        }
        str += "\t"
        for (j <- 0 until numStates) {
          str += " %.4f".format(normalizedCPT(currentOffset + j).dv)
        }
        println(str)
        currentOffset += numStates
      }
    }
  }
  
  /** Recursive, helper method for updating the states list. */
  def updateStatesString(statesList:Mat, parentStates:Mat, j:Int) {
    if (statesList(j).dv.toInt < parentStates(j).dv.toInt-1) {
      statesList(j) += 1
    } else {
      statesList(j) = 0
      updateStatesString(statesList, parentStates, j-1)
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
 * That's it. Other settings, such as the number of Gibbs iterations, are set in "opts".
 */
object BayesNet  {
  
  trait Opts extends Model.Opts {
    var alpha = 0.1f
    var samplingRate = 1
    var numSamplesBurn = 0
  }
  
  class Options extends Opts {}

  /** 
   * A learner with a matrix data source, with states per node, and with a dag prepared. Call this
   * using (assuming proper names):
   * 
   * val (nn,opts) = BayesNet.learner(loadIMat("states.lz4"), loadSMat("dag.lz4"), loadSMat("sdata.lz4"))
   */
  def learner(statesPerNode:Mat, dag:Mat, data:Mat) = {

    class xopts extends Learner.Options with BayesNet.Opts with MatDS.Opts with IncNorm.Opts 
    val opts = new xopts
    opts.dim = dag.ncols
    opts.batchSize = math.min(100000, data.ncols/20 + 1)
    opts.useGPU = false
    opts.npasses = 2 
    opts.isprob = false     // Our CPT should NOT be normalized across their (one) column.
    opts.putBack = 1        // Because this stores samples across ipasses, as required by Gibbs sampling
    val secondMatrix = data.zeros(data.nrows,data.ncols)

    val nn = new Learner(
        new MatDS(Array(data:Mat, secondMatrix), opts),
        new BayesNet(SMat(dag), statesPerNode, opts),
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
class Graph(val dag: Mat, val n: Int, val statesPerNode: Mat) {
 
  var mrf: Mat = null
  var colors: Mat = null
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
    for (i <- 0 until l) {
      for (j <- 0 until l) {
        if (parents(i) != parents(j)) {
          moral(parents(i), parents(j)) = 1f
        }
      }
    }
    moral
  } 

  /** Forms the pproject matrix (dag + identity) used for computing model parameters. */
  def pproject : SMat = {
    return SMat(dag) + sparse(IMat(0 until n), IMat(0 until n), ones(1, n))
  }
  
  /**
   * Forms the iproject matrix, which is left-multiplied to send a Pr(X_i | parents) query to its
   * appropriate spot in the cpt via LOCAL offsets for X_i.
   */
  def iproject : SMat = {
    var res = (pproject.copy).t
    for (i <- 0 until n) {
      val parents = find(SMat(pproject(?, i)))
      var cumRes = 1f
      val parentsLen = parents.length
      for (j <- 1 until parentsLen) {
        cumRes = cumRes * IMat(statesPerNode)(parents(parentsLen - j))
        res.asInstanceOf[SMat](i, parents(parentsLen - j - 1)) = cumRes
      }
    }
    return SMat(res)
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
      var parents = find(SMat(dag(?, i)))
      moral = connectParents(FMat(moral), parents)
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
      var nbs = find(FMat(mrf(?, node)))
      var colorMap = iones(ncolors, 1)
      for (j <- 0 until nbs.length) {
        if (colors(nbs(j)).dv.toInt > -1) {
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


/**
 * This will store a lot of pre-computed variables (mostly matrices) for each color group.
 * 
 * A high-level description of the categories:
 * 
 * - numNodes and numNodesCh are the number of nodes, and the number of nodes and children
 *     in this color group, respectively.
 * - idsInColor and chIdsInColor are indices of the variables in this color group, and in
 *     this color group plus children of those nodes, respectively.
 * - replicationMatrix is a sparse matrix of rows of ones, used to replicate columns
 * - strideVector is a vector where groups are (0 until k)*stride(x) where k is determined
 *     by the node or its parent, and stride(x) is 1 if the node is in the color group.
 * - combinationMatrix is a sparse identity matrix that combines parents with children for
 *     probability computations
 * - keys, scaledKeys, ikeys, and bkeys help us with multinomial sampling
 * - The remaining ten (!) matrices rely on knowledge of the batch size. They are expanded
 *     versions of the previous matrices that use the batch size to increase their elements.
 */
class ColorGroup {
  var numNodes:Int = -1
  var numNodesCh:Int = -1
  var idsInColor:Mat = null
  var chIdsInColor:Mat = null
  var globalOffsetVector:Mat = null
  var iprojectSliced:Mat = null
  var startingIndices:Mat = null
  var replicationMatrix:Mat = null
  var strideVector:Mat = null
  var combinationMatrix:Mat = null
  var keys:Mat = null
  var scaledKeys:Mat = null
  var ikeys:Mat = null
  var bkeys:Mat = null
  var keysMatrix:Mat = null
  var keysMatrixLast:Mat = null
  var bkeysMatrix:Mat = null
  var bkeysMatrixLast:Mat = null
  var bkeysOffsets:Mat = null
  var bkeysOffsetsLast:Mat = null
  var sampleIDindices:Mat = null
  var sampleIDindicesLast:Mat = null
  var randMatrixIndices:Mat = null
  var randMatrixIndicesLast:Mat = null
}
