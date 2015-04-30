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

/**
 * A Bayesian Network implementation with fast parallel Gibbs Sampling (e.g., for MOOC data).
 * 
 * Haoyu Chen and Daniel Seita are building off of Huasha Zhao's original code.
 */

// Put general reminders here:
// TODO Check if all these (opts.useGPU && Mat.hasCUDA > 0) tests are necessary.
// TODO Investigate opts.nsampls. For now, have to do batchSize * opts.nsampls or something like that.
// TODO Check if the BayesNet should be re-using the user for now
// To be honest, right now it's far easier to assume the datasource is providing us with a LENGTH ONE matrix each step
// TODO Be sure to check if we should be using SMats, GMats, etc. ANYWHERE we use Mats.
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
  
  /**
   * Performs a series of initialization steps, such as building the iproject/pproject matrices,
   * setting up a randomized (but normalized) CPT, and randomly sampling users if our datasource
   * provides us with an array of two or more matrices.
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
   * Performs parallel sampling over the color groups, for opts.uiter iterations (i.e., it does multiple
   * Gibbs sampling iterations), over the current batch of data.
   * 
   * @param sdata
   * @param user
   * @param ipass The current pass over the data source (not the Gibbs sampling iteration number).
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
    // TODO I think we need to assign to "user" here in order to get evalfun() to work.
  }
  
  /**
   * After one full round of Gibbs sampling iterations, update and normalize a new CPT that then gets
   * passed in as an updater to update the true model matrix (i.e., the actual CPT we get as output).
   * 
   * @param sdata
   * @param user
   * @param ipass The current pass over the data source (not the Gibbs sampling iteration number).
   */
  def mupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    val numCols = size(user, 2)
    val index = IMat(cptOffset + iproject * user)
    var counts = zeros(mm.length, 1)
    for (i <- 0 until numCols) {
      counts(index(?, i)) = counts(index(?, i)) + 1
    }
    counts = counts + opts.alpha
    updatemats(0) <-- counts / (normConstMatrix * counts)
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
    val a = cptOffset + IMat(iproject * user)
    val b = maxi(maxi(a,1),2).dv
    if (b >= mm.length) {
      println("ERROR! In eval(), we have max index " + b + ", but cpt.length = " + mm.length)
    }
    val index = IMat(cptOffset + iproject * user)
    val ll = sum(sum(ln(getCpt(index))))
    return ll.dv
  }
  
  /**
   * Does a uupdate/mupdate on a datasource segment, called from dobatchg() in Model.scala.
   * 
   * @param mats An array of matrices representing a segment of the original data.
   * @param ipass The current pass over the data source (not the Gibbs sampling iteration number).
   * @param here The total number of elements seen so far, including the ones in this current batch.
   */
  def dobatch(mats:Array[Mat], ipass:Int, here:Long) = {
    val sdata = mats(0)
    val user = if (mats.length > 1) mats(1) else BayesNetMooc3.reuseuser(mats(0), opts.dim, 1f)
    uupdate(sdata, user, ipass)
    mupdate(sdata, user, ipass)
  }

  /**
   * Does a uupdate/evalfun on a datasource segment, called from evalbatchg() in Model.scala.
   * 
   * @param mats An array of matrices representing a segment of the original data.
   * @param ipass The current pass over the data source (not the Gibbs sampling iteration number).
   * @param here The total number of elements seen so far, including the ones in this current batch.
   * @return The log likelihood of the data on this datasource segment (i.e., sdata).
   */
  def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat = {
    val sdata = mats(0)
    val user = if (mats.length > 1) mats(1) else BayesNetMooc3.reuseuser(mats(0), opts.dim, 1f)
    uupdate(sdata, user, ipass)
    evalfun(sdata, user)
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



// TODO We need to describe how we want the data to be formatted
object BayesNetMooc3  {

  trait Opts extends Model.Opts {
    var nsampls = 1 // Can we assume this is always 1 please?
    var alpha = 0.1f
    var uiter = 1
    var eps = 1e-9
  }
  
  class Options extends Opts {}

  /** 
   * A learner with file paths to a matrix data source file, a states per node file, and a dag file.
   * This will need to form the actual data elements! This has not been tested yet. 
   */
  def learner(dataFile:String, statesPerNodeFile:String, dagFile:String) = {
    var data: SMat = null
    var statesPerNode: IMat = null  
    var dag: SMat = null
    // Then copy the code from the next learner, unless we can figure out a way to directly call it
  } 

  /** 
   * A learner with a matrix data source, with states per node, and with a dag prepared for us.
   * This has not been tested yet.
   */
  def learner(data:Mat, statesPerNode:Mat, dag:Mat) = {
    class xopts extends Learner.Options with BayesNetMooc3.Opts with MatDS.Opts with IncNorm.Opts 
    val opts = new xopts
    opts.dim = dag.ncols
    opts.batchSize = data.ncols // Easiest for debugging to start it as data.ncols
    val nn = new Learner(
        new MatDS(Array(data:Mat), opts),
        new BayesNetMooc3(dag, statesPerNode, opts),
        null,
        new IncNorm(opts),
        opts)
    (nn, opts)
  }
  
  /**
   * A learner with a files data source, with states per node, and with a dag prepared for us.
   * This has not been tested yet.
   */
  def learner(fnames:List[(Int)=>String], statesPerNode:Mat, dag:Mat) {
    class xopts extends Learner.Options with BayesNetMooc3.Opts with FilesDS.Opts with IncNorm.Opts 
    val opts = new xopts
    opts.dim = dag.ncols
    opts.batchSize = 10000 // Just a "random" number for now TODO change obviously!
    opts.fnames = fnames
    implicit val threads = threadPool(4)
    val nn = new Learner(
        new FilesDS(opts),
        new BayesNetMooc3(dag, statesPerNode, opts),
        null,
        new IncNorm(opts),
        opts)
    (nn, opts)   
  }
  
  // ---------------------------------------------------------------------- //
  // Other non-learner methods, such as those that manage data input/output //
  
  // TODO Check if this is what we want/need. We may need something like this if we're sticking with 'user'.
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
    /*
    var tempNodeMap = new scala.collection.mutable.HashMap[String, Int]()
    var lines = scala.io.Source.fromFile(path).getLines
    for (l <- lines) {
      var t = l.split(",")
      tempNodeMap += (t(0) -> (t(1).toInt - 1))
    }
    nodeMap = tempNodeMap
    */
  }

  /**
   * Loads the size of state to create state size array: statesPerNode. In the input file, each
   * line looks like N1, 3 (i.e. means there are 3 states for N1 node, which are 0, 1, 2) 
   * 
   * @param path The state size file
   * @param n The number of the nodes
   */
  def loadStateSize(path: String, n: Int) = {
    /*
    statesPerNode = izeros(n, 1)
    var lines = scala.io.Source.fromFile(path).getLines
    for (l <- lines) {
      var t = l.split(",")
      statesPerNode(nodeMap(t(0))) = t(1).toInt
    }
    */
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
    /*
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
    */
  }

  /**
   * Loads the data and stores the training samples in 'sdata'. 
   *
   * The path refers to a text file that consists of five columns: the first is the line's hash, the
   * second is the student number, the third is the question number, the fourth is a 0 or a 1, and
   * the fifth is the concept ID. We should be able to directly split the lines to put data in "row", 
   * "col" and "v", which are then put into an (nq x ns) sparse matrix.
   * 
   * Note: with new data, the fourth column should probably be an arbitrary value in {0,1,...,k}. In
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
    /*
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
      if (t(5) == "1") {
        // NEW! Only add this if we have never seen the pair...
        val a = sMap(shash)
        val b = nodeMap("I"+t(2))
        if (!(coordinatesMap contains (a,b))) {
          coordinatesMap += ((a,b) -> 1)
          row(ptr) = a
          col(ptr) = b
          // Originally for binary data, this was: v(ptr) = (t(3).toFloat - 0.5) * 2
          v(ptr) = t(3).toFloat+1
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
    */
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
    /*
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
      if (t(5) == "0") {
        val a = sMap(shash)
        val b = nodeMap("I"+t(2))
        if (!(coordinatesMap contains (a,b))) {
          coordinatesMap += ((a,b) -> 1)
          row(ptr) = a
          col(ptr) = b
          // Originally for binary data, this was: v(ptr) = (t(3).toFloat - 0.5) * 2
          v(ptr) = t(3).toFloat+1
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
    */
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

