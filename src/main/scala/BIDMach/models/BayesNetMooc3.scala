package BIDMach.models

import BIDMat.{ CMat, CSMat, DMat, Dict, IDict, FMat, GMat, GIMat, GSMat, HMat, IMat, Mat, SMat, SBMat, SDMat }
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import java.io._
import scala.util.Random

/**
 * A Bayesian Network implementation with fast parallel Gibbs Sampling (e.g., for MOOC data).
 * 
 * Haoyu Chen and Daniel Seita are building off of Huasha Zhao's original code.
 */

// Put general reminders here:
// TODO Check if all these (opts.useGPU && Mat.hasCUDA > 0) tests are necessary.
// TODO Investigate opts.nsampls. For now, have to do batchSize * opts.nsampls or something like that.
class BayesNetMooc3(val dag:Mat, 
                    val states:Mat, 
                    override val opts:BayesNetMooc3.Opts = new BayesNetMooc3.Options) extends Model(opts) {

  var graph:Graph1 = null
  var mm:Mat = null         // the cpt in our code
  var iproject:Mat = null
  var pproject:Mat = null
  var cptOffset:IMat = null
  var statesPerNode:IMat = IMat(states)
  var normConstMatrix:SMat = null
  
  /* this is the init method; it does the same tasks in our setup method, but it will also init the state by randomly sample
   * some numbers. The filling unknown elements parts is the same as the "initState()" method in the old code.
  */
  def init() = {
    
    // build graph and the iproject/pproject matrices from it
    graph = dag match {
      case dd:SMat => new Graph1(dd, opts.dim, statesPerNode)
      case _ => throw new RuntimeException("dag not SMat")
    }
    graph.color
    iproject = if (opts.useGPU && Mat.hasCUDA > 0) GMat(graph.iproject) else graph.iproject
    pproject = if (opts.useGPU && Mat.hasCUDA > 0) GMat(graph.pproject) else graph.pproject

    // build the cpt (which is the modelmats) and the cptoffset vectors
    val numSlotsInCpt = if (opts.useGPU && Mat.hasCUDA > 0) {
      GIMat(exp(GMat((pproject.t)) * ln(GMat(statesPerNode))) + 1e-3) 
    } else {
      IMat(exp(DMat(full(pproject.t)) * ln(DMat(statesPerNode))) + 1e-3)     
    }
    var cptOffset = izeros(graph.n, 1)
    if (opts.useGPU && Mat.hasCUDA > 0) {
      cptOffset(1 until graph.n) = cumsum(GMat(numSlotsInCpt))(0 until graph.n-1)
    } else {
      cptOffset(1 until graph.n) = cumsum(IMat(numSlotsInCpt))(0 until graph.n-1)
    }
    val lengthCPT = sum(numSlotsInCpt).dv.toInt
    var cpt = rand(lengthCPT,1)
    normConstMatrix = getNormConstMatrix(cpt)
    cpt = cpt / (normConstMatrix * cpt)
    setmodelmats(new Array[Mat](1))
    modelmats(0) = if (opts.useGPU && Mat.hasCUDA > 0) GMat(cpt) else cpt
        
    // TODO Well, this is only for if we have a matrix of length > 1 ... but we are changing the size of 
    // state, which is confusing because I assumed that the columns had to be consistent among the sources.
    // init the data here, i.e. revise the data into the our format, i.e. randomnize the unknown elements
    // I think it will read each matrix and init the state, and putback to the datarescource. 
    // it would be similar as what we wrote in initState()
    // here it has one more variable: opts.nsampls, it's like the number of traning samples
    if (mats.size > 1) {
      while (datasource.hasNext) {
        mats = datasource.next
        val sdata = mats(0)
        var state = mats(1)
        state = rand(state.nrows, state.ncols * opts.nsampls)
        if (opts.useGPU && Mat.hasCUDA > 0) {
          state = min( GMat(trunc(statesPerNode *@ state)) , statesPerNode-1)
        } else {
          state = min( FMat(trunc(statesPerNode *@ state)) , statesPerNode-1)
        }
        val innz = sdata match { 
          case ss: SMat => find(ss >= 0)
          case ss: GSMat => find(SMat(ss) >= 0)
          case _ => throw new RuntimeException("sdata not SMat/GSMat")
        }
        for(i <- 0 until opts.nsampls){
          state.asInstanceOf[FMat](innz + i * sdata.ncols *  graph.n) = 0f
          state(?, i*sdata.ncols until (i+1)*sdata.ncols) = state(?, i*sdata.ncols until (i+1)*sdata.ncols) + (sdata.asInstanceOf[SMat](innz))
        }
        datasource.putBack(mats,1)
      }
    }

    mm = modelmats(0)
    updatemats = new Array[Mat](1)
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols)
  }

  /**
   * Performs parallel sampling over the color groups, for opts.uiter iterations.
   * 
   * @param sdata
   * @param user
   * @param ipass The current Gibbs sampling iteration index.
   */
  def uupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    if (putBack < 0 || ipass == 0) user.set(1f) // If no info on 'state' matrix, set all elements to be 1.
    for (k <- 0 until opts.uiter) {
      for(c <- 0 until graph.ncolors){
        val idInColor = find(graph.colors == c)
        val numState = IMat(maxi(maxi(statesPerNode(idInColor),1),2)).v
        var stateSet = new Array[Mat](numState)
        var pSet = new Array[Mat](numState)
        var pMatrix = zeros(idInColor.length, sdata.ncols * opts.nsampls)
        for (i <- 0 until numState) {
          val saveID = find(statesPerNode(idInColor) > i)
          val ids = idInColor(saveID)
          val pids = find(FMat(sum(pproject(ids, ?), 1)))
          initStateColor(sdata, ids, i, stateSet, user)
          computeP(ids, pids, i, pSet, pMatrix, stateSet(i), saveID, idInColor.length)
        }
        sampleColor(sdata, numState, idInColor, pSet, pMatrix, user)
      } 
    } 
  }
  
  /**
   * I think this method is equivalent to our method: "updateCpt"
   */
  def mupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    println("Not yet implemented")
  }
  
  def dobatch(mats:Array[Mat], ipass:Int, here:Long) = {
    println("Not yet implemented")
  }

  def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat = {
    println("Not yet implemented")
    return null
  }

  /**
   * Initializes the statei matrix for this particular color group and for this particular value.
   * It fills in the unknown values at the ids locations with i, then we can use it in computeP.
   * 
   * @param fdata Training data matrix, with unknowns of -1 and known values in {0,1,...,k}.
   * @param ids Indices of nodes in this color group that can also attain value/state i.
   * @param i An integer representing a value/state (we use these terms interchangeably).
   * @param stateSet An array of statei matrices, each of which has "i" in the unknowns of "ids".
   * @param user
   */
  def initStateColor(fdata: Mat, ids: IMat, i: Int, stateSet: Array[Mat], user:Mat) = {
    // TODO Why do we need to cast statei to FMats, and use that test for innz?
    var statei = user.copy
    statei(ids,?) = i
    if (!checkState(statei)) {
      println("problem with initStateColor(), max elem is " + maxi(maxi(statei,1),2).dv)
    }
    val innz = find(FMat(fdata) >= 0)
    /*
    val innz = fdata match {
      case ss: SMat => find(fdata >= 0)
      case ss: GMat => find(SMat(fdata) >= 0)
    }
    val innz = find(fdata >= 0)
    statei.asInstanceOf[FMat](innz) = 0f
    statei(innz) <-- statei(innz) + fdata(innz)
     * 
     */
    for (i <- 0 until opts.nsampls) {
      statei.asInstanceOf[FMat](innz + i * fdata.ncols * graph.n) <--  fdata.asInstanceOf[SMat](innz)
    }
    if (!checkState(statei)) {
      println("problem with end of initStateColor(), max elem is " + maxi(maxi(statei,1),2).dv)
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

   // the changes I made here are 1) use the opts.eps to replace the 1e-10, change the dim of the pSet,
   // i.e. the ncols of pSet is batchsize * nsampls

  def computeP(ids: IMat, pids: IMat, i: Int, pSet: Array[Mat], pMatrix: Mat, statei: Mat, saveID: IMat, numPi: Int) = {
    val a = cptOffset(pids) + IMat(iproject(pids, ?) * statei)
    val b = maxi(maxi(a,1),2).dv
    if (b >= mm.length) {
      println("ERROR! In computeP(), we have max index " + b + ", but cpt.length = " + mm.length)
    }
    val nodei = ln(getCpt(cptOffset(pids) + IMat(iproject(pids, ?) * statei)) + opts.eps)
    var pii = zeros(numPi, statei.ncols)
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
   * @param user
   */
   // changes I made in this part: 1) delete the globalPMatrices parts; 2) use the user to replace the state matrix
   // 3) add the for loop for the nsampls; 4) fdata change type to be Mat

  def sampleColor(fdata: Mat, numState: Int, idInColor: IMat, pSet: Array[Mat], pMatrix: Mat, user: Mat) = {
    val sampleMatrix = rand(idInColor.length, fdata.ncols * opts.nsampls)
    pSet(0) = pSet(0) / pMatrix
    user(idInColor,?) <-- 0 * user(idInColor,?)
    
    // Each time, we check to make sure it's <= pSet(i), but ALSO exceeds the previous \sum (pSet(j)).
    for (i <- 1 until numState) {
      val saveID = find(statesPerNode(idInColor) > i)
      val ids = idInColor(saveID)
      val pids = find(FMat(sum(pproject(ids, ?), 1)))
      pSet(i) = (pSet(i) / pMatrix) + pSet(i-1) // Normalize and get the cumulative prob
      // Use Hadamard product to ensure that both requirements are held.
      user(ids, ?) = user(ids,?) + i * ((sampleMatrix(saveID, ?) <= pSet(i)(saveID, ?)) *@ (sampleMatrix(saveID, ?) >= pSet(i - 1)(saveID, ?)))
      if (!checkState(user)) {
        println("problem with loop in sampleColor(), max elem is " + maxi(maxi(user,1),2).dv)
      }
    }

    // Finally, re-write the known state into the state matrix
    val saveIndex = find(FMat(fdata) >= 0)
    for (j <- 0 until opts.nsampls) {
      user.asInstanceOf[FMat](saveIndex + j * fdata.ncols * graph.n) <--  fdata.asInstanceOf[SMat](saveIndex)
    }
    if (!checkState(user)) {
      println("problem with end of sampleColor(), max elem is " + maxi(maxi(user,1),2).dv)
    }
  }


  /**
   * Creates normalizing matrix N that we can then multiply with the cpt, i.e., N * cpt, to get a column
   * vector of the same length as the cpt, but such that cpt / (N * cpt) is normalized. Use SMat to save
   * on memory, I think.
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

  /** Returns a conditional probability table specified by indices from the "index" matrix. */
  def getCpt(index: Mat) = {
    var cptindex = mm.zeros(index.nrows, index.ncols)
    for(i <-0 until index.ncols){
      cptindex(?, i) = mm(IMat(index(?, i)))
    }
    cptindex
  }

  /** Returns FALSE if there's an element at least size 2, which is BAD. */
  def checkState(state: Mat) : Boolean = {
    val a = maxi(maxi(state,2),1).dv
    if (a >= 2) {
      return false
    }
    return true
  }

}



// TODO Work in progress on the options, need to establish learners
object BayesNetMooc3  {
  trait Opts extends Model.Opts {
    var nsampls = 1
    var alpha = 0.01f
    var uiter = 1
    var eps = 1e-9
  }
  
  class Options extends Opts {}
  
  /*
  def mkBayesNetmodel(dag: Mat, fopts:Model.Opts) = {
  	new BayesNet(dag, fopts.asInstanceOf[BayesNet.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
  	new IncNorm(nopts.asInstanceOf[IncNorm.Opts])
  } 
   
  def learner(dag0:Mat, mat0:Mat) = {
    class xopts extends Learner.Options with BayesNet.Opts with MatDS.Opts with IncNorm.Opts
    val opts = new xopts
    //opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    opts.batchSize = mat0.ncols
    opts.dim = dag0.ncols
  	val nn = new Learner(
  	    new MatDS(Array(mat0:Mat), opts), 
  	    new BayesNet(dag0, opts), 
  	    null,
  	    new IncNorm(opts), 
  	    opts)
    (nn, opts)
  }
  
  def reuseuser(a:Mat, dim:Int, ival:Float):Mat = {
    val out = a match {
      case aa:SMat => FMat.newOrCheckFMat(dim, a.ncols, null, a.GUID, "SMat.reuseuser".##)
      case aa:FMat => FMat.newOrCheckFMat(dim, a.ncols, null, a.GUID, "FMat.reuseuser".##)
      case aa:GSMat => GMat.newOrCheckGMat(dim, a.ncols, null, a.GUID, "GSMat.reuseuser".##)
      case aa:GMat => GMat.newOrCheckGMat(dim, a.ncols, null, a.GUID, "GMat.reuseuser".##)
    }
    out.set(ival)
    out
  }
  *
  */
     
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

