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
 * The input needs to be (1) a graph, (2) a sparse data matrix, (3) a matrix of equivalence classes
 * (optional, can set to null if needed), and (4) a states-per-node file.
 * Make sure the dag and states files are aligned, and that variables are in a topological ordering!
 */
class BayesNet(val dag:Mat, 
               val states:Mat, 
               val equivClasses:Mat,
               override val opts:BayesNet.Opts = new BayesNet.Options) extends Model(opts) {

  // Miscellaneous, we might want to keep these recorded
  val randSeed:Int = 9999
  
  var mm:Mat = null                         // Copy of the cpt, but be careful of aliasing. We keep this normalized.
  var cptOffset:Mat = null                  // Holds global variable offsets (into the mm = cpt) of each variable.
  var cptOffsetSAME:Mat = null              // A vertically stacked version of cptOffset, for SAME.
  var graph:Graph = null                    // Data structure representing the DAG, "columns = parents."
  var iproject:Mat = null                   // Local CPT offsets; we do "usertrans * iproject" to get the offsets.
  var iprojectBlockedSAME:Mat = null        // A diagonal, blocked version of iproject, for SAME local CPT offsets.
  var pproject:Mat = null                   // Parent tracking matrix, for combining probabilities together.
  var statesPerNode:Mat = null              // Variables can have an arbitrary number of states.
  var statesPerNodeSAME:Mat = null          // A vertically stacked version of statesPerNode, for SAME.
  var colorInfo:Array[ColorGroup] = null    // Gives us, for each color, a colorStuff class (of arrays).
  var zeroMap:HashMap[(Int,Int),Mat] = null // Map from (nr,nc) -> a zero matrix (to avoid allocation).
  var randMap:HashMap[(Int,Int),Mat] = null // Map from (nr,nc) -> a rand matrix (to avoid allocation).
  var normConstMatrix:Mat = null            // Normalizes the cpt. Do cpt / (cpt.t * nConstMat).t.
  var useGPUnow:Boolean = false             // Checks (during initialization only) if we're using GPUs or not.
  var batchSize:Int = -1                    // Holds the batchSize, which we use for some colorInfo matrices.

  var counts1:Mat = null                    // This will accumulate counts that we use for the actual distribution.
  var counts2:Mat = null                    // This will be the counts that we use for the *previous* step that we SUBTRACT.
  var counts3:Mat = null                    // Use this as a copy, because the GPU genericGammaRand doesn't seem to be working.
  var counts4:Mat = null                    // This is like counts1, but WITH Dirichlets!

  var dirichletPrior:Mat = null             // The prior we use to smooth the distribution. If all 1s, SAME will keep it the same.
  var dirichletScale:Mat = null             // The scale we use as part of the prior (typically all 1s).
  var onesSAMEvector:Mat = null             // This the (g)iones(opts.copiesForSAME,1), for certain special uses.
  
  // For equivalence classes (not in the ICLR 2016 paper)
  var equivClassCountMap:Mat = null         // Used to combine counts from same equiv. classes so they get pooled together.
  var equivClassVector:Mat = null           // A vector used to clear out components except leading variables in CPT
  
  // Extra debugging/info gathering
  val real1 = .7 on .3 on .6 on .4 on .95 on .05 on .2 on .8 on .3 on .4 on .3 on .05 on .25 
  val real2 = .7 on .9 on .08 on .02 on .5 on .3 on .2 on .1 on .9 on .4 on .6 on .99 on .01     
  val real = real1 on real2

  // Only for the DLM MOOC data (must tweak for other datasets)
  var previousCPT:Mat = null
  var predictions:Mat = null // Stores the counts we sample during the testing data, depends on SAME
  var predictionsRand:Mat = null // Random matrix for the predictions, again depends on SAME
  var numIterationsPredict = 501 // How many times we iterate. Then prob of a 1 is numIterationsPredict/2 if we have counts.
  var numInitialBurn = 500 // Number of sampling passes we IGNORE, before starting numIterationsPredict
  var areWePredicting:Boolean = false // Indicator for us in the prediction process for uupdate

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
    // Some stuff for experiments and predictions...
    setseed(randSeed)
    println("randSeed = " + randSeed)
    previousCPT = loadFMat("data/ICLR_2016_MOOC_Extras/cpt_iter500_seed0.txt") // MOOC forward sampling experiment
    //predictions = gzeros(334 * opts.copiesForSAME,1367)     // NOTE! When we sample, we actually only use the top block ...
    //predictionsRand = gzeros(334 * opts.copiesForSAME,1367) // NOTE! When we sample, we actually only use the top block ...

    // Now back to normal...
    useGPUnow = opts.useGPU && (Mat.hasCUDA > 0)
    onesSAMEvector = if (useGPUnow) giones(opts.copiesForSAME,1) else iones(opts.copiesForSAME,1)

    // Establish the states per node, the (colored) Graph data structure, and its projection matrices.
    statesPerNode = IMat(states)
    statesPerNodeSAME = kron(onesSAMEvector, IMat(statesPerNode))
    graph = new Graph(dag, opts.dim, statesPerNode)
    graph.color
    iproject = if (useGPUnow) GSMat((graph.iproject).t) else (graph.iproject).t
    pproject = if (useGPUnow) GSMat(graph.pproject) else graph.pproject
    iprojectBlockedSAME = createBlockedDiagonal(iproject)
   
    // Build the CPT. To avoid div-by-zero errors, initialize randomly.
    val numSlotsInCpt = IMat(exp(ln(FMat(statesPerNode).t) * SMat(pproject)) + 1e-4)
    cptOffset = izeros(graph.n, 1)
    cptOffset(1 until graph.n) = cumsum(numSlotsInCpt)(0 until graph.n-1)
    cptOffset = convertMat(cptOffset)
    cptOffsetSAME = kron(onesSAMEvector,cptOffset)
    val lengthCPT = sum(numSlotsInCpt).dv.toInt
    val cpt = convertMat(rand(lengthCPT,1))
    
    // New! If we have equivalence classes, form equiv matrix and use it on the initialized cpt.
    if (equivClasses != null) {
      val result = createEquivClassMatrix(numSlotsInCpt, cptOffset, lengthCPT)
      equivClassCountMap = result._1
      equivClassVector = result._2
      cpt <-- (cpt.t * equivClassCountMap).t // This makes the random values approach 0.5 w/more variables.
    } 
   
    // To finish building CPT, we normalize it based on the batch size and normalizing constant matrix.
    normConstMatrix = getNormConstMatrix(lengthCPT)
    cpt <-- ( cpt / (cpt.t * normConstMatrix).t )
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
    counts1 = mm.zeros(mm.length, 1)
    counts2 = mm.zeros(mm.length, 1)
    counts3 = zeros(mm.length, 1)   // FMat, until the genericGammaRand works on the GPU.
    counts4 = mm.zeros(mm.length, 1)

    dirichletPrior = ones(mm.length, 1) // Change to mm.ones when genericGammaRand works
    dirichletScale = ones(mm.length, 1) // Change to mm.ones when genericGammaRand works
    statesPerNode = convertMat(statesPerNode) 
    batchSize = -1
  } 
   
  /**
   * Calls a uupdate/mupdate sequence. Known data is in gmats(0), sampled data is in gmats(1).
   * If I want to debug, I have several options here:
   *
   * //debugCpt(ipass, here)
   * //computeNormDifference(ipass,here)
   * //computeKL(ipass, here, real)
   * 
   * This is for the MOOC data
   * //println("\nIpass = " + ipass + ", here = " + here)
   * //showCpt(0)
   * //showCpt(186)
   * //showCpt(187)
   * //showCpt(372)
   * //println("")
   * 
   * New (Nov. 2015): if we reach some iteration where we want to do foward sampling to generate data,
   * then do that here after saving the CPT. Then we run this with that generated data.
   */
  override def dobatch(gmats:Array[Mat], ipass:Int, here:Long) = {
    //if (ipass == 500) { // Only for saving CPT and generating data from this CPT via forward/direct sampling.
    //  saveFMat("cpt_iter" + ipass + "_seed" + randSeed + ".txt", FMat(mm))
    //  generateDataFromCPT(ipass)
    //  sys.exit
    //}
    computeKL(ipass, here, previousCPT) // NOTE! Use 'real' for student data, 'previousCPT' for MOOC data

    // Compute counts that we SUBTRACT away later! Call now b/c later, gmats(1)=user gets overrided.
    if (ipass > 0) {
      val index = int(cptOffsetSAME + (gmats(1).t * iprojectBlockedSAME).t)
      val linearIndices = index(?)
      counts2 <-- float(accum(linearIndices, 1, counts2.length, 1))
    }

    uupdate(gmats(0), gmats(1), ipass)
    mupdate(gmats(0), gmats(1), ipass)
  }
  
  /** Calls a uupdate/evalfun sequence. Known data is in gmats(0), sampled data is in gmats(1). */
  override def evalbatch(gmats:Array[Mat], ipass:Int, here:Long):FMat = {
    //areWePredicting = true
    //uupdate(gmats(0), gmats(1), ipass) // For evaluation w/gflops, we don't really need this?
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

    // For SAME, we stack matrices. If kron is missing (type) cases, add them in MatFunctions.scala.
    val stackedData = kron(onesSAMEvector, sdata)
    val select = stackedData > 0

    // For the first pass, we need to create a lot of matrices that rely on knowledge of the batch size.
    var numGibbsIterations = opts.samplingRate
    if (ipass == 0) {
      numGibbsIterations = numGibbsIterations + opts.numSamplesBurn
      establishMatrices(sdata.ncols)
      val state = convertMat(rand(sdata.nrows * opts.copiesForSAME, sdata.ncols))
      state <-- float( min( int(statesPerNodeSAME ∘ state), int(statesPerNodeSAME-1) ) )
      user ~ (select ∘ (stackedData-1)) + ((1-select) ∘ state)
    }
    
    // NEW! If we are doing prediction accuracy, we override "user" so that it is completely random.
    // Applies to all batches; after ipass = 0, we still need to randomize so we ignore putback stuff.
    if (areWePredicting) {
      numGibbsIterations = numIterationsPredict + numInitialBurn // Completely override it with what we use.
      rand(predictionsRand)
      user <-- float( min( int(statesPerNodeSAME ∘ predictionsRand), int(statesPerNodeSAME-1) ) )
      predictions.clear
    }
    
    // Now back to normal from prediction accuracy. usertrans is still user.t
    val usertrans = user.t

    for (k <- 0 until numGibbsIterations) {
      for (c <- 0 until graph.ncolors) {
        
        // Prepare data by establishing appropriate offset matrices for various CPT blocks. First, clear out usertrans.
        usertrans(?, colorInfo(c).idsInColorSAME) = zeroMap( (usertrans.nrows, colorInfo(c).numNodes*opts.copiesForSAME) ) 
        val offsetMatrix = usertrans * colorInfo(c).iprojectSlicedSAME + (colorInfo(c).globalOffsetVectorSAME).t
        val replicatedOffsetMatrix = int(offsetMatrix * colorInfo(c).replicationMatrixSAME) + colorInfo(c).strideVectorSAME
        val logProbs = ln(mm(replicatedOffsetMatrix))
        val nonExponentiatedProbs = (logProbs * colorInfo(c).combinationMatrixSAME).t
        
        // Establish matrices needed for the multinomial sampling
        val keys = if (user.ncols == batchSize) colorInfo(c).keysMatrix else colorInfo(c).keysMatrixLast
        val bkeys = if (user.ncols == batchSize) colorInfo(c).bkeysMatrix else colorInfo(c).bkeysMatrixLast
        val bkeysOff = if (user.ncols == batchSize) colorInfo(c).bkeysOffsets else colorInfo(c).bkeysOffsetsLast
        val randIndices = if (user.ncols == batchSize) colorInfo(c).randMatrixIndices else colorInfo(c).randMatrixIndicesLast
        val sampleIndices = if (user.ncols == batchSize) colorInfo(c).sampleIDindices else colorInfo(c).sampleIDindicesLast
      
        // Parallel multinomial sampling. Check the colorInfo matrices since they contain a lot of info.
        //val maxInGroup = cummaxByKey(nonExponentiatedProbs, keys)(bkeys) // To prevent overflow (if needed).
        //val probs = exp(nonExponentiatedProbs - maxInGroup) // To prevent overflow (if needed).
        val probs = exp(nonExponentiatedProbs)
        probs <-- (probs + 1e-30f) // Had to add this for the DLM MOOC data to prevent 0/(0+0) problems.
        val cumprobs = cumsumByKey(probs, keys)
        val normedProbs = cumprobs / cumprobs(bkeys)    
        
        // With cumulative probabilities set up in normedProbs matrix, create a random matrix and sample
        val randMatrix = randMap( (colorInfo(c).numNodes*opts.copiesForSAME, usertrans.nrows) )
        rand(randMatrix)
        randMatrix <-- randMatrix * 0.99999f
        val lessThan = normedProbs < randMatrix(randIndices)
        val sampleIDs = cumsumByKey(lessThan, keys)(sampleIndices)
        usertrans(?, colorInfo(c).idsInColorSAME) = sampleIDs.t // Note the SAME now...

        // After sampling with this color group over all copies (from SAME), we override the known values.
        // NEW! If we are predicting, we have to AVOID the overriding data thing!
        if (areWePredicting) {
          // Do nothing
        } else { // This was the old part of the code.
          usertrans ~ (select ∘ (stackedData-1)).t + ((1-select) ∘ usertrans.t).t
        }
      }
      
      // NEW! After *all* nodes sampled *without* overriding known values, we increment predictions.
      // Then if we have another Gibbs iteration, it continues from these values, and we again increment.
      if (areWePredicting && k > numInitialBurn) {
        predictions ~ predictions + usertrans.t
      }

    }
    user <-- usertrans.t
  }

  /**
   * After one set of Gibbs sampling iterations, we have a set of counts for each slot in the cpt.
   * We add values from the dirichletPrior, then sample all the parameters independently from a Gamma
   * distribution Gamma(shape,scale=1), where the shape is the count they have. Then the values are
   * put in updatemats(0) to be "averaged into" the cpt based on IncNorm.
   * 
   * @param sdata The sparse data matrix for this batch (0s = unknowns), which we do not use here.
   * @param user A data matrix with the same dimensions as sdata, and whose columns represent various
   *    iid assignments to all the variables. The known values of sdata are inserted in the same spots
   *    in this matrix, but the unknown values are randomized to be in {0,1,...,k}.
   * @param ipass The current pass over the full data source (not the Gibbs sampling iteration number).
   */
  def mupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    val index = int(cptOffsetSAME + (user.t * iprojectBlockedSAME).t)
    val linearIndices = index(?)
    if (ipass > 0) {
      counts1 ~ counts1 - counts2       // Drop the corresponding previous mini-batch
    }
    counts1 ~ counts1 + float(accum(linearIndices, 1, counts1.length, 1)) // Accumulate w/current mini-batch

    // First accumulate raw counts. Then, after setting dirichlet prior to all and sampling,
    // we only take the first per equiv class (thus, equivClassVector), then re-accumulate.
    //if (equivClasses != null) {
    //  counts1 <-- (counts1.t * equivClassCountMap).t
    //}

    // New, test for GPU/CPU differences
    //genericGammaRand(counts1 + dirichletPrior, dirichletScale, counts3) // GPU
    genericGammaRand(FMat(counts1) + dirichletPrior, dirichletScale, counts3) // CPU
    counts4 <-- convertMat(counts3)

    //if (equivClasses != null) {
    //  counts2 <-- (counts2 *@ equivClassVector)
    //  counts2 <-- (counts2.t * equivClassCountMap).t
    //}

    updatemats(0) <-- (counts4 / (counts4.t * normConstMatrix).t) 
  }
 
  /**
   * Evaluates the log-likelihood of the data (per column, or per full assignment of all variables).
   * First, we get the index matrix, which indexes into the CPT for each column's variable assignment.
   * Then, using the normalized CPT, we find the log probabilities of the user matrix, and sum
   * vertically (i.e., each variable, valid due to derived rules) and then horizontally (i.e., each
   * sample, which can do since we assume i.i.d.). 
   */
  def evalfun(sdata:Mat, user:Mat):FMat = {  

    // Take the 'predictions' matrix, iterate through the known indices of sdata, and see if they match.
    // Remember that 'predictions' has counts and we take majority by doing numIterationsPredict / 2.0.
    // Also, sdata is on a 1-2 scale, while predictions assumes we added up a bunch of 0s and 1s.
    if (areWePredicting) {
      areWePredicting = false
      val threshold = numIterationsPredict / 2.0
      var totalOnes = 0f
      var totalTwos = 0f
      //println("threshold = " + threshold)
      var correct = 0f
      val (r, c, v) = find3(SMat(sdata))
      for (i <- 0 until sdata.nnz) {
        val ri = IMat(r(i).toInt)
        val ci = c(i).toInt
        var pred = 1
        if (predictions(ri,ci).dv >= threshold) {
          pred = 2
          totalTwos = totalTwos + 1
        } else {
          totalOnes = totalOnes + 1
        }
        if (pred == v(i).toInt) {
          correct = correct + 1
        }
        //println("ri,ci = " + ri + "," + ci + ", predictions(ri,ci) = " + predictions(ri,ci).dv)
      }
      val res = correct / sdata.nnz
      println("Accuracy: " + res + " = " + correct + " / " + sdata.nnz + ". Total 0/1 predicted = " + totalOnes.toInt + "," + totalTwos.toInt + ", with random seed " + randSeed + ".")
      //sys.exit
    } 

    val index = int(cptOffsetSAME + (user.t * iprojectBlockedSAME).t)
    val result = FMat( sum(sum(ln(mm(index)),1),2) ) / (user.ncols * opts.copiesForSAME)
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
   * Actually, this method name is a bit misleading because some of the color group info relies on
   * knowing the batch size, and we can't do that until we actually see the data.
   * 
   * @param c The integer index of the given color group.
   */
  def computeAllColorGroupInfo(c:Int) : ColorGroup = {
    val cg = new ColorGroup 
    cg.idsInColor = find(IMat(graph.colors) == c)
    cg.numNodes = cg.idsInColor.length
    cg.chIdsInColor = find(FMat(sum(SMat(pproject)(cg.idsInColor,?),1)))
    cg.idsInColorSAME = cg.idsInColor
    for (i <- 1 until opts.copiesForSAME) {
      // Unlike other things where we could use kron, here we change indices b/c we use this
      // for matrix indexing when "clearing out columns" in usertrans when sampling.
      cg.idsInColorSAME = cg.idsInColorSAME on (cg.idsInColor + i*graph.n)
    }
    cg.numNodesCh = cg.chIdsInColor.length
    cg.iprojectSliced = SMat(iproject)(?,cg.chIdsInColor)
    cg.iprojectSlicedSAME = createBlockedDiagonal(cg.iprojectSliced)
    cg.globalOffsetVector = convertMat(FMat(cptOffset(cg.chIdsInColor))) // Need FMat to avoid GMat+GIMat
    cg.globalOffsetVectorSAME = kron(onesSAMEvector, cg.globalOffsetVector)
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
    // A bit confusing, since strideVector is a ROW vector
    cg.strideVectorSAME = kron( onesSAMEvector.t, cg.strideVector)
    cg.replicationMatrix = if (useGPUnow) GSMat(sparse(ii,jj,vv)) else sparse(ii,jj,vv) 
    cg.replicationMatrixSAME = createBlockedDiagonal(cg.replicationMatrix)
    
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
    
    // Now make SAME versions of these! The keys needs to have extra appended at end, 
    // incremented by graph.n just in case we have a color group with just one node.
    cg.keysSAME = keys
    for (i <- 1 until opts.copiesForSAME) {
      cg.keysSAME = cg.keysSAME \ (keys + i*graph.n)
    }
    cg.keysSAME = convertMat(cg.keysSAME)
    cg.bkeysSAME = cg.bkeys
    for (i <- 1 until opts.copiesForSAME) {
      cg.bkeysSAME = cg.bkeysSAME \ (cg.bkeys + i*(cg.bkeys).length)
    }
    cg.scaledKeysSAME = cg.scaledKeys
    for (i <- 1 until opts.copiesForSAME) {
      cg.scaledKeysSAME = cg.scaledKeysSAME \ (cg.scaledKeys + cg.numNodes)
    }
    cg.ikeysSAME = cg.ikeys
    for (i <- 1 until opts.copiesForSAME) {
      cg.ikeysSAME = cg.ikeysSAME \ (cg.ikeys + i*(cg.bkeys).length)
    }

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
    cg.combinationMatrixSAME = createBlockedDiagonal(cg.combinationMatrix)
    
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
   *  - "zero" matrices to put in zeroMap, for clearing out usertrans (must consider opts.copiesForSAME!)
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
        val numVars = colorInfo(c).numNodes * opts.copiesForSAME // SAME!
        val randOffsets = int(untilVector * numVars)
        zeroMap += ((ncols,numVars) -> mm.zeros(ncols,numVars))
        randMap += ((numVars,ncols) -> mm.zeros(numVars,ncols))
        colorInfo(c).keysMatrix = (colorInfo(c).keysSAME).t * onesVector // keys -> keysSAME
        colorInfo(c).bkeysOffsets = int(untilVector * colorInfo(c).keysSAME.ncols) // keys -> keysSAME
        colorInfo(c).bkeysMatrix = int(colorInfo(c).bkeysSAME.t * onesVector) + colorInfo(c).bkeysOffsets // bkeys -> bkeysSAME
        colorInfo(c).randMatrixIndices = int((colorInfo(c).scaledKeysSAME).t * onesVector) + randOffsets // scaledKeys -> scaledKeysSAME
        colorInfo(c).sampleIDindices = int((colorInfo(c).ikeysSAME).t * onesVector) + colorInfo(c).bkeysOffsets // ikeys -> ikeysSAME
      }
    } 
    else if (ncols != batchSize) { // On the last batch of ipass = 0 w/different # of columns
      val onesVectorLast = mm.ones(1, ncols)
      val untilVectorLast = convertMat( float(0 until ncols) )
      for (c <- 0 until graph.ncolors) {
        val numVars = colorInfo(c).numNodes * opts.copiesForSAME // SAME!
        val randOffsets = int(untilVectorLast * numVars)
        zeroMap += ((ncols,numVars) -> mm.zeros(ncols,numVars))
        randMap += ((numVars,ncols) -> mm.zeros(numVars,ncols))
        colorInfo(c).keysMatrixLast = (colorInfo(c).keysSAME).t * onesVectorLast
        colorInfo(c).bkeysOffsetsLast = int(untilVectorLast * colorInfo(c).keysSAME.ncols)
        colorInfo(c).bkeysMatrixLast = int(colorInfo(c).bkeysSAME.t * onesVectorLast) + colorInfo(c).bkeysOffsetsLast
        colorInfo(c).randMatrixIndicesLast = int((colorInfo(c).scaledKeysSAME).t * onesVectorLast) + randOffsets
        colorInfo(c).sampleIDindicesLast = int((colorInfo(c).ikeysSAME).t * onesVectorLast) + colorInfo(c).bkeysOffsetsLast
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
   
  /**
   * Given a matrix as input, we form a diagonal, blocked version of it. So if a is a (sparse) mat, it is
   * like calling kron(mkdiag(ones(1,n)), full(a)), except I think this will be a lot more flexible later. 
   * Places where we use this: user.t * iproject, usertrans * colorInfo(c).iprojectSliced, etc.
   * 
   * @input a A sparse matrix. It does not have to be square!
   */
  def createBlockedDiagonal(a:Mat) : Mat = {
    val (ii,jj,vv) = find3(SMat(a))
    val vvv = iones(opts.copiesForSAME,1) kron vv
    var iii = izeros(1,1)
    var jjj = izeros(1,1)
    for (k <- 0 until opts.copiesForSAME) {
      iii = iii on (ii + k*a.nrows)
      jjj = jjj on (jj + k*a.ncols)
    }
    val res = sparse(iii(1 until iii.length), jjj(1 until jjj.length), vvv, a.nrows*opts.copiesForSAME, a.ncols*opts.copiesForSAME)
    if (useGPUnow) return GSMat(res) else return res         
  }
  
  /** 
   * Creates the sparse, blocked matrix with lots of identities, that can map values in the same
   * equivalence classes together in the appropriate slots (before normalization). We should be able
   * to do: eClassMatrix * cpt to get them to combine appropriately, but due to sparse * dense problems,
   * we actually do (cpt.t * eClassMatrix).t. Note that our matrix here is guaranteed symmetric.
   * Also, we're going to assume that variables are only part of ONE cpt, for now. Let's be nice to myself.
   * This is pretty inefficient, but hey, we only do this once ...
   * 
   * UPDATE: Now we have to return a *second* matrix...
   * 
   * @param numSlots A vector where numSlots(i) = number of total CPT slots w.r.t. variable i.
   * @param cptOffset The usual stuff
   * @param n The length of the cpt
   */ 
  def createEquivClassMatrix(numSlots:Mat, cptOffset:Mat, n:Int) : (Mat,Mat) = {
    var iii = IMat(0 until n).t // B/c we iterate through equiv classes, so need to add cases for non-equiv vars
    var jjj = IMat(0 until n).t // Same reasoning
    var res2 = ones(n,1)

    for (c <- 0 until equivClasses.ncols) {
      // Note: by construction of equivClasses, vars.length >= 2.
      // Also note: we will also require consecutive vars to be here, for simplicity.
      // And we'll use kron for now ... b/c it shouldn't be *that* expensive.
      val vars = find(IMat(equivClasses(?,c)))  // vars(0) = integer index of first variable in this CPT
      val start = cptOffset(vars(0)).dv.toInt
      val num = numSlots(vars(0)).dv.toInt
      val end = cptOffset(vars(vars.length-1)).dv.toInt + (num-1)
      val startNextVar = start + num
      res2(startNextVar to end) = 0 // Ah, use "to" here since I had "num-1" earlier... ;)
      val eye = mkdiag( ones(numSlots(vars(0)).dv.toInt,1) )
      
      // TODO put more cases for kron to handle IMat and FMat (and IMat-IMat pairs)
      // nonShiftedMatrix is the set of identity matrix blocks together, then we add 'start' to rows and cols
      val nonShiftedMatrix = kron(ones(vars.length,vars.length), eye)
      val (ii,jj) = find2(IMat(nonShiftedMatrix))
      val iis = ii + start
      val jjs = jj + start
      iii = iii on iis // (ii + start)
      jjj = jjj on jjs // (jj + start)
    }

    val vvv = iones(iii.length, 1)
    val res1 = sparse(iii, jjj, vvv , n, n) > 0  // Add > 0 b/c some repetition might lead to 2s in diagonal.
    if (useGPUnow) {
      return (GSMat(res1), convertMat(res2))
    } else {
      return (res1, res2)
    }
  }

  // ---------------------------------------------
  // The remaining methods are for debugging only.
  // ---------------------------------------------
  
  /** A debugging method to print matrices, without being constrained by the command line's cropping. */
  def printMatrix(mat: Mat) = {
    for(i <- 0 until mat.nrows) {
      for (j <- 0 until mat.ncols) {
        print(mat(IMat(i),IMat(j)) + " ")
      }
      println()
    }
  }    

  /**
   * A debugging method to compute the norm of difference between normalized real/estimated cpts.
   * Note: this *does* assume our mm is already normalized!
   * Obviously we'll have to replace the real cpt with what we already have...
   */
  def computeNormDifference(ipass:Int, here:Long) = {
    val real = .7 on .3 on .6 on .4 on .95 on .05 on .2 on .8 on
            .3 on .4 on .3 on .05 on .25 on .7 on .9 on .08 on .02 on .5 on .3 on .2 on .1 on .9 on .4 on .6 on .99 on .01 
    val differenceNorm = norm(real - mm)
    println("Currently on ipass = " + ipass + " with here = " + here + "; l-2 norm of (realCpt - mm) is: " + differenceNorm)
  }
  
  /** KL divergence. We assume our mm is normalized. */
  def computeKL(ipass:Int, here:Long, comparisonCPT:Mat) {
    var klDivergence = convertMat(float(0))
    var numDistributions = 0
 
    for (k <- 0 until graph.n) {
      var offset = cptOffset(k).dv.toInt
      val numStates = statesPerNode(k).dv.toInt
      val parentIndices = find(SMat(graph.dag)(?,k))

      // Then split based on no parents (one distribution) or some parents (two or more distributions)
      if (parentIndices.length == 0) {
        var thisKL = convertMat(float(0))
        for (j <- 0 until numStates) {
          thisKL = thisKL + (comparisonCPT(offset+j) * ln( comparisonCPT(offset+j) / mm(offset+j) ))
        }
        klDivergence = klDivergence + thisKL
        numDistributions += 1
      } else {
        val totalParentSlots = prod(IMat(statesPerNode)(parentIndices)).dv.toInt
        numDistributions += totalParentSlots
        for (i <- 0 until totalParentSlots) {       
          var thisKL = convertMat(float(0))
          for (j <- 0 until numStates) {
            thisKL = thisKL + ( comparisonCPT(offset+j) * ln( comparisonCPT(offset+j) / mm(offset+j) ))
          }  
          klDivergence = klDivergence + thisKL
          offset += numStates
        }
      }
    }      
    
    klDivergence = klDivergence / numDistributions
    println(klDivergence + "  " + ipass + " KLDiv")
  }
  
  /** A one-liner that we can insert in a place with ipass and here to debug the cpt. */
  def debugCpt(ipass:Int, here:Long) {
    println("\n\nCurrently on ipass = " + ipass + " with here = " + here + ". This is the CPT:")
    for (k <- 0 until graph.n) {
      showCpt(k)
    }
    println()
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
  
  /** New, do this to experiment with the MOOC data. Do forward sampling (inefficiently, sorry). */
  def generateDataFromCPT(ipass:Int) = {
    val nrows = 4367 // CHANGE AS NEEDED!
    val output = zeros(nrows, graph.n)
    println("\nCurrently generating data from this cpt, via forward sampling, with num samples = " + nrows + ".\n")
    println("\nour cpt:\n" + mm.t)

    for (n <- 0 until graph.n) {
      if (n % 25 == 0) {
        println("Finished with node " + n + " of " + graph.n + ".")
      }
      val startingIndices = FMat(output * full(iproject)(?,n)) + FMat(cptOffset(n)) // Important, we only take the column
      val a = statesPerNode(n).dv.toInt // number of states for this node
      val b = float(0 until a)          // [0,1,...,a-1] ROW vector
      val indices = kron(ones(nrows,1), b)
      val indicesIntoCPT = IMat(startingIndices + indices)
      val cptValues = FMat(mm(indicesIntoCPT)) // Each row is a sample, and must add to one due to normalized probabilities.
      val normedCumSums = cumsumByKey(cptValues.t, ones(cptValues.ncols, cptValues.nrows))
      val randVec = kron(ones(a,1), (rand(1, nrows) * 0.99999f))
      val lessThan = normedCumSums < randVec
      val sampleIDs = sum(lessThan, 1)
      output(?,n) = sampleIDs.t
    }

    saveFMat("newMOOCdata_" + ipass + "ipasses_" + randSeed + ".lz4", output.t) // transposed!
  }

}


/**
 * There are three things the BayesNet needs as input:
 * 
 *  - A states per node array. Each value needs to be an integer that is at least two.
 *  - A DAG array, in which column i represents node i and ones in that column are its parents.
 *  - A sparse data matrix, where 0 indicates an unknown element, and rows are variables.
 * 
 * That's it. Other settings, such as the number of Gibbs iterations, are set in "opts".
 */
object BayesNet {
  
  trait Opts extends Model.Opts {
    var copiesForSAME = 1
    var samplingRate = 1
    var numSamplesBurn = 0
  }
  
  class Options extends Opts {}

  /** 
   * A learner with a matrix data source, with states per node, and with a dag prepared. Call this
   * using (assuming proper names):
   * 
   * val (nn,opts) = BayesNet.learner(loadIMat("states.lz4"), loadSMat("dag.lz4"), loadSMat("sdata.lz4"))
   * 
   * New: we're adding in an eClass, but we can set that to be null if needed.
   */
  def learner(statesPerNode:Mat, dag:Mat, eClasses:Mat, data:Mat) = {

    class xopts extends Learner.Options with BayesNet.Opts with MatDS.Opts with IncNorm.Opts 
    val opts = new xopts
    opts.dim = dag.ncols
    opts.batchSize = math.min(100000, data.ncols/50 + 1)
    opts.useGPU = false
    opts.npasses = 2 
    opts.isprob = false     // Our CPT should NOT be normalized across their (one) column.
    opts.putBack = 1        // Because this stores samples across ipasses, as required by Gibbs sampling
    opts.power = 0.0f       // So that the sampled CPT parameters are exactly what we use next iteration
    val secondMatrix = data.zeros(opts.copiesForSAME*data.nrows,data.ncols)

    val nn = new Learner(
        new MatDS(Array(data:Mat, secondMatrix), opts),
        new BayesNet(SMat(dag), statesPerNode, eClasses, opts),
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
 *  - Oh! Don't forge that we have SAME versions of these!
 */
class ColorGroup {
  var numNodes:Int = -1
  var numNodesCh:Int = -1
  var idsInColor:Mat = null
  var idsInColorSAME:Mat = null
  var chIdsInColor:Mat = null
  var globalOffsetVector:Mat = null
  var globalOffsetVectorSAME:Mat = null
  var iprojectSliced:Mat = null
  var iprojectSlicedSAME:Mat = null
  var startingIndices:Mat = null
  var replicationMatrix:Mat = null
  var replicationMatrixSAME:Mat = null
  var strideVector:Mat = null
  var strideVectorSAME:Mat = null
  var combinationMatrix:Mat = null
  var combinationMatrixSAME:Mat = null

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
  
  var keysSAME:Mat = null
  var bkeysSAME:Mat = null
  var scaledKeysSAME:Mat = null
  var ikeysSAME:Mat = null
}
