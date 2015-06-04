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
 * The input needs to be (1) a graph, (2) a sparse data matrix, and (3) a states-per-node file
 * 
 * This is still a WIP.
 */

// Put general reminders here:
// TODO Check if all these (opts.useGPU && Mat.hasCUDA > 0) tests are necessary.
// TODO Investigate opts.nsampls. For now, have to do batchSize * opts.nsampls or something like that.
// TODO Check if the BayesNet should be re-using the user for now
// TODO Be sure to check if we should be using SMats, GMats, etc. ANYWHERE we use Mats.
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
  var normConstMatrix:Mat = null            // For debugging if needed. Do cpt / (nConstMat * cpt).
  var useGPUnow:Boolean = false             // NOTE: Ideally, we will NOT need to use this at all!
  var batchSize:Int = -1
  var lastBatchSize:Int = -1

  /**
   * Performs a series of initialization steps.
   * 
   * - Builds iproject/pproject for local offsets and computing probabilities, respectively.
   * - For each color group, determine all necessary matrices for use in uupdate later.
   * - Build the CPT, which is actually counts, not probabilities. I initialize it to be ones.
   * 
   * Note that the randomization of the input data to be put back in the data is done in uupdate.
   */
  override def init() = {
    println("At the start of init, GPU memory is " + GPUmem)

    // Establish the states per node, the (colored) Graph data structure, and its projection matrices.
    useGPUnow = opts.useGPU && (Mat.hasCUDA > 0)
    statesPerNode = IMat(states)
    graph = new Graph(dag, opts.dim, statesPerNode)
    graph.color
    iproject = (graph.iproject).t
    pproject = graph.pproject
    
    // For each color group, pre-compute all relevant matrices we need later (this does a lot!)
    colorInfo = new Array[ColorGroup](graph.ncolors)
    for (c <- 0 until graph.ncolors) {
      colorInfo(c) = computeAllColorGroupInfo(c)
    }
    zeroMap = new HashMap[(Int,Int),Mat]()
    
    // Build the CPT. For now, it stores counts, and to avoid div-by-zero errors, initialize w/ones.
    val numSlotsInCpt = IMat(exp(ln(statesPerNode.t) * pproject) + 1e-4)
    cptOffset = izeros(graph.n, 1)
    cptOffset(1 until graph.n) = cumsum(numSlotsInCpt)(0 until graph.n-1)
    cptOffset <-- convertMat(cptOffset)
    val lengthCPT = sum(numSlotsInCpt).dv.toInt
    val cpt = cptOffset.ones(lengthCPT,1)
    setmodelmats(new Array[Mat](1))
    modelmats(0) = cpt
    mm = modelmats(0)
    updatemats = new Array[Mat](1)
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols)

    // To wrap it up, convert a few matrices and add some debugging info
    normConstMatrix = convertMat(getNormConstMatrix(lengthCPT))
    statesPerNode <-- convertMat(statesPerNode) 
    println("At the end of init, GPU memory is " + GPUmem)
  } 
   
  /** Calls a uupdate/mupdate sequence. If ipass == 0, we pre-compute zero matrices for later. */
  override def dobatch(gmats:Array[Mat], ipass:Int, here:Long) = {
    val ncols = gmats(0).ncols
    if (ipass == 0) {
      val first = (here == ncols.toLong)    // We are on the first mini-batch of the first pass
      val last = (ncols != batchSize)       // We are on the last mini-batch of the first pass
      if (first || last) {
        if (first) { 
          batchSize = ncols
        }
        lastBatchSize = ncols
        for (c <- 0 until graph.ncolors) {
          val ncols = colorInfo(c).numNodes
          zeroMap += ((batchSize,ncols) -> mm.zeros(batchSize,ncols))
          zeroMap += ((lastBatchSize,ncols) -> mm.zeros(lastBatchSize,ncols))
        }
      }
    }
    uupdate(gmats(0), gmats(1), ipass)
    mupdate(gmats(0), gmats(1), ipass)
  }
  
  /** Calls a uupdate/evalfun sequence. */
  override def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat = {
    uupdate(gmats(0), gmats(1), ipass)
    evalfun(gmats(0), gmats(1))
  }  
 
  /**
   * TODO describe
   * 
   * @param sdata The sparse data matrix for this batch (0s = unknowns), which the user matrix shifts by -1.
   * @param user
   * @param ipass
   */
  def uupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    var numGibbsIterations = opts.samplingRate
    if (ipass == 0) {
      numGibbsIterations = numGibbsIterations + opts.numSamplesBurn
      val state = convertMat(rand(sdata.nrows, sdata.ncols))
      state <-- min( trunc(statesPerNode *@ state), statesPerNode-1 )
      val data = full(sdata)
      val select = sdata > 0
      user ~ (select *@ (data-1)) + ((1-select) *@ state)
    }
    val usertrans = user.t

    for (k <- 0 until numGibbsIterations) {
      for (c <- 0 until graph.ncolors) {

        // Prepare our data by establishing the appropriate offset matrices for the entire cpt blocks
        usertrans <-- user.t // TODO Temp fix because user gets updated w/known indices, but usertrans doesn't. I think there's a better way to do this.
        val assignment = usertrans.copy
        assignment(?, colorInfo(c).idsInColor) = zeroMap(usertrans.nrows, colorInfo(c).numNodes)
        val offsetMatrix = assignment * colorInfo(c).iprojectSliced + (colorInfo(c).globalOffsetVector).t
        val replicatedOffsetMatrix = offsetMatrix * colorInfo(c).replicationMatrix + colorInfo(c).strideVector
        val probabilities = ln(mm(replicatedOffsetMatrix))
        val combinedProbabilities = exp(probabilities * colorInfo(c).combinationMatrix)

        // Now we can sample for each color in this color group.
        for (i <- 0 until colorInfo(c).numNodes) {
          if (useGPUnow) {
            val start = colorInfo(c).startingIndices(i).dv.toInt
            val numStates = statesPerNode(colorInfo(c).idsInColor(i)).dv.toInt
            val probs = GMat(combinedProbabilities(?, start until start+numStates).t)
            val samples = probs.izeros(probs.nrows, probs.ncols)
            multinomial(probs.nrows, probs.ncols, probs.data, samples.data, sum(probs,1).data, 1)
            val (maxVals, indices) = maxi2( samples ) 
            usertrans(?, colorInfo(c).idsInColor(i)) = indices.t
          } else {
            // TODO for now I'll randomly put samples here because otherwise I'd have to write one
            val indices = IMat(rand(usertrans.nrows,1) * statesPerNode(i) * 0.9999999)
            usertrans(?, colorInfo(c).idsInColor(i)) = FMat(indices)
          }
        }

        // After we finish with this color group, we should override the known values because that affects other parts.
        user ~ usertrans.t
        val data = full(sdata)
        val select = sdata > 0
        user ~ (select *@ (data-1)) + ((1-select) *@ user)
      }     
    }
    
    // After a complete Gibbs iteration (or more, depending on burn-in or thinning), update the CPT.
    updateCPT(user)
  }

  /**
   * After one round of Gibbs sampling iterations, we put the local cpt (mm) into the updatemats(0)
   * value so that it gets "averaged into" the global cpt, modelmats(0). The reason is that it is like
   * "thinning" the samples so we pick every n-th one, where n is an opts parameter.
   * 
   * @param sdata The sparse data matrix for this batch (0s = unknowns), which we do not use here.
   * @param user A data matrix with the same dimensions as sdata, and whose columns represent various
   *    iid assignments to all the variables. The known values of sdata are inserted in the same spots
   *    in this matrix, but the unknown values are randomized to be in {0,1,...,k}.
   * @param ipass The current pass over the full data source (not the Gibbs sampling iteration number).
   */
  def mupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    updatemats(0) <-- mm
  }   
 
  /** TODO This is not finished... but its probably a simple sum over all the cpts? Maybe I do need to normalize? */
  def evalfun(sdata:Mat, user:Mat):FMat = {  
  	row(0f, 0f)
  } 
  
  // -----------------------------------
  // Various debugging or helper methods
  // -----------------------------------
  
  /** 
   * Needs to determine the color group info for color c.
   */
  def computeAllColorGroupInfo(c:Int) : ColorGroup = {
    val cg = new ColorGroup 
    cg.idsInColor = find(IMat(graph.colors) == c)
    cg.numNodes = cg.idsInColor.length
    cg.chIdsInColor = find( FMat( sum(pproject(cg.idsInColor,?),1) ) ) // For some reason I need FMat(...)
    cg.numNodesCh = cg.chIdsInColor.length
    cg.iprojectSliced = iproject(?,cg.chIdsInColor)
    cg.globalOffsetVector = cptOffset(cg.chIdsInColor)
    val startingIndices = izeros(cg.numNodes,1)
    startingIndices(1 until cg.numNodes) = cumsum(IMat(statesPerNode(cg.idsInColor)))(0 until cg.numNodes)
    cg.startingIndices = convertMat(startingIndices)

    // Gather useful information for determining the replication, stride, and combination matrices
    var ncols = 0
    val numOnes = izeros(1,cg.numNodesCh)       // Determine how many 1s to have
    val strideFactors = izeros(1,cg.numNodesCh) // Get stride factors for the stride vector
    val parentOf = izeros(1,cg.numNodesCh)      // Get index of parent (or itself) in idsInColor
    for (i <- 0 until cg.numNodesCh) {
      var nodeIndex = cg.chIdsInColor(i)
      if (IMat(cg.idsInColor).data.contains(nodeIndex)) { // This node is in the color group
        numOnes(i) = statesPerNode(nodeIndex)
        ncols = ncols + statesPerNode(nodeIndex).dv.toInt
        strideFactors(i) = 1
        parentOf(i) = IMat(cg.idsInColor).data.indexOf(nodeIndex)
      } else { // This node is a child of a node in the color group
        val parentIndices = find( FMat( sum(pproject(?,nodeIndex),2) ) )
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
          throw new RuntimeException("For node at index " + nodeIndex + ", it seems to be missing a parent in this color group.")
        }
        numOnes(i) = statesPerNode(parentIndex)
        ncols = ncols + statesPerNode(parentIndex).dv.toInt
        strideFactors(i) = iproject(parentIndex,nodeIndex).dv.toInt
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
      
    // Form the combination matrix (# of rows is # of columns of replication matrix)
    val numStatesIds = statesPerNode(cg.idsInColor)
    val ncolsCombo = sum(numStatesIds).dv.toInt
    val indicesColumns = izeros(1,cg.numNodes)
    indicesColumns(1 until cg.numNodes) = cumsum(numStatesIds.asInstanceOf[IMat])(0 until cg.numNodes)
    val nrowsCombo = ncols
    val indicesRows = izeros(1,cg.numNodesCh)
    indicesRows(1 until cg.numNodesCh) = cumsum(numOnes)(0 until numOnes.length-1)
    val iii = izeros(nrowsCombo,1)
    val jjj = izeros(nrowsCombo,1)
    val vvv = ones(nrowsCombo,1)
    for (i <- 0 until cg.numNodesCh) {
      val p = parentOf(i) // Index into the PARENT of this node, usually different from i, and NOT global system
      iii(indicesRows(i) until indicesRows(i)+numOnes(i)) = indicesRows(i) until indicesRows(i)+numOnes(i)
      jjj(indicesRows(i) until indicesRows(i)+numOnes(i)) = indicesColumns(p) until indicesColumns(p)+numOnes(i)
    }
    cg.combinationMatrix = if (useGPUnow) {
      GSMat(sparse(iii,jjj,vvv,nrowsCombo,ncolsCombo))
    } else {
      sparse(iii,jjj,vvv,nrowsCombo,ncolsCombo)
    }
    
    return cg
  } 
 
  /**
   * Method to update the local cpt table (i.e. mm), called after one or more iterations of Gibbs sampling.
   * This does not update the Learner's cpt, which is modelmats(0).
   * 
   * @param user The state matrix, with all variables updated after sampling. Columns are the batches, and
   *    rows are the variables.
   */
  def updateCPT(user: Mat) : Unit = {
    println("at start of cpt, gpu mem is " + GPUmem)
    /*
    val index = if (useGPUnow) {
      GIMat(cptOffset + (user.t * iproject).t)
    } else {
      IMat(cptOffset + (user.t * iproject).t)
    }
    */
    val index = cptOffset + (user.t * iproject).t
    var counts = mm.izeros(mm.length, 1)
    var tmp = mm.izeros(index(?,0).length,1)
    var ones = mm.iones(index(?,0).length,1)
    println("before loop, GPU mem is " + GPUmem)
    for (i <- 0 until user.ncols) {
      tmp <-- index(?,i)
      tmp <-- counts(tmp)
      tmp ~ tmp + ones
      counts(index(?,i)) = tmp
      // counts(index(?, i)) = counts(index(?, i)) + 1 // The old way
    }
    println("after loop, GPU mem is " + GPUmem)
    mm <-- (counts + opts.alpha)
  } 
  
  /** For debugging the various color group matrices. */
  /*
  def debugColorGroupMatrices = {
    println("Finished with the replication and stride matrices!")
    for (c <- 0 until graph.ncolors) {
      println("Color group with elements\n" + find(graph.colors == c).t)
      printMatrix(FMat(replicationMatrices(c)))
      println("The stride:")
      printMatrix(FMat(strideVectors(c)))
      println()
    }
    println("Finished with the combination matrices!")
    for (c <- 0 until graph.ncolors) {
      println("Color group with elements\n" + find(graph.colors == c).t)
      printMatrix(FMat(combinationMatrices(c)))
      println()
    }
  }
  */
  
  /**
   * Creates normalizing matrix N that we can then multiply with the cpt, i.e., N * cpt, to get a column
   * vector of the same length as the cpt, but such that cpt / (N * cpt) is normalized. I don't actually
   * use this, but it's nice to have it to find the probabilities later for debugging/infomative purposes.
   */
  def getNormConstMatrix(cptLength : Int) : SMat = {
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
    return sparse(ii(1 until ii.length), jj(1 until jj.length), ones(ii.length-1, 1), cptLength, cptLength)
  }
  
  /** For computing Java runtime memory, can be reasonably reliable. */
  def computeMemory = {
    val runtime = Runtime.getRuntime();
    val format = NumberFormat.getInstance(); 
    val sb = new StringBuilder();
    val allocatedMemory = runtime.totalMemory();
    val freeMemory = runtime.freeMemory();
    sb.append("free memory: " + format.format(freeMemory / (1024*1024)) + "M   ");
    sb.append("allocated/total memory: " + format.format(allocatedMemory / (1024*1024)) + "M\n");
    print(sb.toString())
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
 * That's it. Other settings, such as the number of Gibbs iterations, are set in "opts".
 */
object BayesNet  {
  
  trait Opts extends Model.Opts {
    var alpha = 1f
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
   * One issue with this, I guess, is that it assumes the data fits in memory. TODO anything I should change?
   */
  def learner(statesPerNode:Mat, dag:Mat, data:Mat) = {

    class xopts extends Learner.Options with BayesNet.Opts with MatDS.Opts with IncNorm.Opts 
    val opts = new xopts
    opts.dim = dag.ncols
    opts.batchSize = math.min(100000, data.ncols/30 + 1)
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
// TODO investigate all non-Mat matrices (IMat, FMat, etc.) to see if these can be Mats to make them usable on GPUs
// TODO Also see if connectParents, moralize, and color can be converted to GPU friendly code
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
    for(i <- 0 until l) {
      for(j <- 0 until l) {
        if(parents(i) != parents(j)) {
          moral(parents(i), parents(j)) = 1f
        }
      }
    }
    moral
  } 

  /** Forms the pproject matrix (dag + identity) used for computing model parameters. */
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
      val parents = find(SMat(pproject(?, i)))
      var cumRes = 1
      val parentsLen = parents.length
      for (j <- 1 until parentsLen) {
        cumRes = cumRes * IMat(statesPerNode)(parents(parentsLen - j))
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
 * This will store a set of pre-computed variables (usually matrices) for each color group.
 */
class ColorGroup {
  var numNodes:Int
  var numNodesCh:Int
  var idsInColor:Mat
  var chIdsInColor:Mat
  var globalOffsetVector:Mat
  var iprojectSliced:Mat
  var startingIndices:Mat
  var replicationMatrix:Mat
  var strideVector:Mat
  var combinationMatrix:Mat
}

