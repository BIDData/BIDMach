package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._

import java.text.NumberFormat
import edu.berkeley.bid.CUMACH._

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
class BayesNet(val dag:SMat, 
               val states:Mat, 
               override val opts:BayesNet.Opts = new BayesNet.Options) extends Model(opts) {

  var mm:Mat = null
  var cptOffset:Mat = null
  var graph:Graph = null
  var iproject:SMat = null           // Local CPT offset matrix
  var pproject:SMat = null           // Parent tracking matrix
  var statesPerNode:IMat = null
  var useGPUnow:Boolean = false
  var replicationMatrices:Array[Mat] = null
  var strideVectors:Array[Mat] = null
  var combinationMatrices:Array[Mat] = null
 
  /**
   * Performs a series of initialization steps.
   * 
   * - Builds iproject/pproject for local offsets and computing probabilities, respectively.
   * - Build the CPT, which is actually counts, not probabilities. I initialize it to be ones.
   * - Randomize the input data, which is stored in the second matrix
   */
  def init() = {
    computeMemory

    // Establish the states per node, the (colored) Graph data structure, and its projection matrices.
    useGPUnow = opts.useGPU && (Mat.hasCUDA > 0)
    statesPerNode = IMat(states)
    graph = new Graph(dag, opts.dim, statesPerNode)
    graph.color
    iproject = (graph.iproject).t
    pproject = (graph.pproject)
    
    // Form the replication matrices, stride vectors, and combination matrices, for update() later.
    createColorGroupMatrices
    println("Finished creating replication, stride, and combination matrices.")
    //debugColorGroupMatrices
    
    // Build the CPT. For now, it stores counts, and to avoid div-by-zero errors, initialize w/ones.
    val numSlotsInCpt = IMat(exp(ln(FMat(statesPerNode.t)) * pproject) + 1e-4)
    cptOffset = izeros(graph.n, 1)
    cptOffset(1 until graph.n) = cumsum(numSlotsInCpt)(0 until graph.n-1)
    val lengthCPT = sum(numSlotsInCpt).dv.toInt
    val cpt = if(useGPUnow) gones(lengthCPT, 1) else ones(lengthCPT, 1)
    setmodelmats(new Array[Mat](1))
    modelmats(0) = cpt
    mm = modelmats(0)
    updatemats = new Array[Mat](1)
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols)
    
    // Randomize the initial starting states, and store the data in the second matrix source
    if (this.mats.size < 2) {
      throw new RuntimeException("We need at least two matrices in the datasource.")
    }
    while (datasource.hasNext) {
      mats = datasource.next
      val sdata = mats(0)
      val state = mats(1)
      if (sdata.nrows != state.nrows || sdata.ncols != state.ncols) {
        throw new RuntimeException("size(sdata), size(state) do not match: " +size(sdata)+ " and " +size(state))
      }
      state <-- rand(sdata.nrows, sdata.ncols)
      state <-- min( FMat(trunc(statesPerNode *@ state)) , statesPerNode-1)
      val nonzeroIndices = find(SMat(sdata))
      state(nonzeroIndices) = 0f
      state(nonzeroIndices) = (full(sdata)(nonzeroIndices) - 1)
      datasource.putBack(mats,1)
    }
  } 
   
  // TODO
  def dobatch(gmats:Array[Mat], ipass:Int, here:Long) = {
    /*
    println("Yes, I know not to put too much stock into this, but here is memory anyway:")
    computeMemory()
    if ((ipass+1) % 25 == 0) {
      println("Also, with ipass = " + ipass + ", here is our modelmats(0).t:")
      println(modelmats(0).t)
    } 
    */
    //println("Inside dobatch with ipass = " + ipass + ", user =\n")
    //printMatrix(FMat(gmats(1)))
    computeMemory
    uupdate(gmats(0), gmats(1), ipass)
    mupdate(gmats(0), gmats(1), ipass)
  }
  
  // TODO
  def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat = {
    uupdate(gmats(0), gmats(1), ipass)
    evalfun(gmats(0), gmats(1))
  }  
 
  // Note: sdata is a sparse matrix that has 1s, 2s, ..., ks. The user matrix shifts those over by -1.
  // Thus, when updating stuff, we only want to add the stuff in sdata that exceeds one, and shift that.
  // Of course, for what we use in our code, called input, we need user.t, not user.
  /**
   * TODO
   */
  def uupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    val usertrans = user.t
    var numGibbsIterations = opts.samplingRate
    if (ipass == 0) {
      numGibbsIterations = numGibbsIterations + opts.numSamplesBurn
    }
    
    for (k <- 0 until numGibbsIterations) {
      println("In uupdate(), Gibbs iteration " + (k+1) + " of " + numGibbsIterations)
      for (c <- 0 until graph.ncolors) {

        // Several steps. First, establish local offset matrix for the START of cpt blocks
        val idsInColor = find(graph.colors == c)
        val chIdsInColor = find(FMat( sum(pproject(idsInColor,?),1) ))
        val numStates = statesPerNode(idsInColor)
        val assignment = usertrans.copy
        assignment(?,idsInColor) = if (useGPUnow) {
          gzeros(usertrans.nrows, idsInColor.length)
        } else {
          zeros(usertrans.nrows, idsInColor.length)
        }
        val offsetMatrix = if (useGPUnow) {
          assignment * GSMat(iproject(?,chIdsInColor))
        } else {
          assignment * SMat(iproject(?,chIdsInColor))
        }
        val globalOffsetVector = cptOffset(chIdsInColor)
        offsetMatrix <-- (offsetMatrix + globalOffsetVector.t)

        // Then expand our matrix to find indices for each possible state within the cpt blocks
        val replicatedOffsetMatrix = if (useGPU) {
          GIMat(offsetMatrix * replicationMatrices(c))
        } else {
          IMat(offsetMatrix * replicationMatrices(c))
        }
        replicatedOffsetMatrix <-- (replicatedOffsetMatrix + strideVectors(c))
        val probabilities = ln(mm(replicatedOffsetMatrix))
        val combinedProbabilities = exp(probabilities * combinationMatrices(c))

        // Now we can actually sample for each color in this color group
        val startingIndices = izeros(idsInColor.length,1)
        startingIndices(1 until idsInColor.length) = cumsum(numStates)(0 until idsInColor.length-1)
        for (i <- 0 until idsInColor.length) {
          if (useGPUnow) {
            val probs = GMat(combinedProbabilities(?, startingIndices(i) until startingIndices(i)+statesPerNode(idsInColor(i))).t)
            val samples = probs.izeros(probs.nrows, probs.ncols)
            multinomial(probs.nrows, probs.ncols, probs.data, samples.data, sum(probs,1).data, 1)
            val (maxVals, indices) = maxi2(GMat( samples ));  // maxVals = (1, 1, ..., 1)
            usertrans(?, idsInColor(i)) = GMat(indices.t)
          } else {
            // TODO maybe for now just randomly put samples here? Otherwise I have to write a multinomial sampler.
            val probs = combinedProbabilities(?, startingIndices(i) until startingIndices(i)+statesPerNode(idsInColor(i))).t
            println("Sorry, we can't test with non-GPU matrices now. But here are the probabilities for this node:")
            printMatrix(FMat(probs))
            sys.exit
          }
        }
      }
    }

    // TODO be sure to override the value of user with known values here! Can do it outside of the for loop, by the way.
    user ~ usertrans.t
    computeMemory
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
 
  // TODO This is not finished...
  def evalfun(sdata:Mat, user:Mat):FMat = {  
  	row(0f, 0f)
  } 
  
  // -----------------------------------
  // Various debugging or helper methods
  // -----------------------------------

  /**
   * A little expensive, especially the process of looking up parents, but right now I just want it to work.
   */
  def createColorGroupMatrices = {
    replicationMatrices = new Array[Mat](graph.ncolors)
    strideVectors = new Array[Mat](graph.ncolors)
    combinationMatrices = new Array[Mat](graph.ncolors)

    // Iterate through each color group and add one matrix to each of the three arrays we created
    // Be careful of global indices (e.g., idsInColor, statesPerNode) and local indices (e.g., numOnes, parentOf)
    for (c <- 0 until graph.ncolors) {
      val idsInColor = find(graph.colors == c)
      val chIdsInColor = find(FMat( sum(pproject(idsInColor,?),1) ))
      var ncols = 0
      val numOnes = izeros(1, chIdsInColor.length) // So we can iterate through and determine how many 1s to have
      val strideFactors = izeros(1, chIdsInColor.length) // So we can get stride factors for our strideVectors
      val parentOf = izeros(1, chIdsInColor.length) // So for each variable, we can extract index into chIdsInColor

      for (i <- 0 until chIdsInColor.length) {
        var nodeIndex = chIdsInColor(i)
        if (idsInColor.data.contains(nodeIndex)) {
          numOnes(i) = statesPerNode(nodeIndex)
          ncols = ncols + statesPerNode(nodeIndex)
          strideFactors(i) = 1
          parentOf(i) = idsInColor.data.indexOf(nodeIndex)
        } else {
          // Have to find the ONE parent of this node that IS in the color group
          val parentIndices = find(FMat( sum(pproject(?,nodeIndex),2) ))
          var parentIndex = -1
          var k = 0
          while (parentIndex == -1 && k < parentIndices.length) {
            if (idsInColor.data.contains(parentIndices(k))) {
              parentIndex = parentIndices(k)
              parentOf(i) = idsInColor.data.indexOf(parentIndices(k))
            }
            k = k + 1
          }
          if (parentIndex == -1) {
            throw new RuntimeException("For node at index " + nodeIndex + ", it seems to be missing a parent in this color group.")
          }
          numOnes(i) = statesPerNode(parentIndex)
          ncols = ncols + statesPerNode(parentIndex)
          strideFactors(i) = full(iproject(parentIndex,IMat(nodeIndex))).dv.toInt
        }
      }

      // Form the replication and stride matrices
      var col = 0
      val strideVector = izeros(1, ncols)
      val ii = izeros(ncols, 1)
      for (i <- 0 until chIdsInColor.length) {
        val num = numOnes(i)
        ii(col until col+num) = i
        strideVector(col until col+num) = (0 until num)*strideFactors(i)
        col = col + num
      }
      val jj = icol(0 until ncols)
      val vv = ones(ncols, 1)
      val replicationMatrix = sparse(ii, jj, vv) // dims are (#-of-ch_id-variables x ncols)
      replicationMatrices(c) = if (useGPUnow) GSMat(replicationMatrix) else replicationMatrix
      strideVectors(c) = if (useGPUnow) GIMat(strideVector) else strideVector
      
      // Form the combination matrix
      val numStatesIds = statesPerNode(idsInColor)
      val ncolsCombo = sum(numStatesIds).dv.toInt
      val indicesColumns = izeros(1, idsInColor.length)
      indicesColumns(1 until idsInColor.length) = cumsum(numStatesIds)(0 until idsInColor.length-1)
      val nrowsCombo = ncols
      val indicesRows = izeros(1,chIdsInColor.length)
      indicesRows(1 until chIdsInColor.length) = cumsum(numOnes)(0 until numOnes.length-1)
      val iii = izeros(nrowsCombo,1)
      val jjj = izeros(nrowsCombo,1)
      val vvv = ones(nrowsCombo,1)
      for (i <- 0 until chIdsInColor.length) {
        val p = parentOf(i) // Index into the PARENT of this node, usually different from i, and NOT global system
        iii(indicesRows(i) until indicesRows(i)+numOnes(i)) = indicesRows(i) until indicesRows(i)+numOnes(i)
        jjj(indicesRows(i) until indicesRows(i)+numOnes(i)) = indicesColumns(p) until indicesColumns(p)+numOnes(i)
      }
      val combinationMatrix = sparse(iii,jjj,vvv,nrowsCombo,ncolsCombo) // # rows is # columns of replicationMatrix
      combinationMatrices(c) = if (useGPUnow) GSMat(combinationMatrix) else combinationMatrix
    }
  }
  
  /** For debugging the various color group matrices. */
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
    var numSamplesBurn = 100
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
class Graph(val dag: SMat, val n: Int, val statesPerNode: IMat) {
 
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

