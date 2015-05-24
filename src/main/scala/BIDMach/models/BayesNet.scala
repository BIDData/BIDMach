package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._

/**
 * WIP: Bayes network using cooled Gibbs parameter estimation
 */
class BayesNet(val dag:SMat, override val opts:BayesNet.Opts = new BayesNet.Options) extends Model(opts) {

  var mm:Mat = null
  var offset:Mat = null
  var graph:Graph = null
  var iproject:Mat = null
  var pproject:Mat = null
  
  def init() = {
    graph = new Graph(dag, opts.dim)
    graph.color
    iproject = if (opts.useGPU && Mat.hasCUDA > 0) GSMat(graph.iproject) else graph.iproject
    pproject = if (opts.useGPU && Mat.hasCUDA > 0) GSMat(graph.pproject) else graph.pproject
    
    // prepare cpt 
    val np = sum(dag)
    val ns = IMat(pow(2*ones(1, graph.n), np+1))
    val l = sum(ns).v
    val modelmat = rand(l, 1)
    modelmat(1 until l by 2) = 1 - modelmat(0 until l-1 by 2)
    setmodelmats(new Array[Mat](1))
    modelmats(0) = if (opts.useGPU && Mat.hasCUDA > 0) GMat(modelmat) else modelmat
    
    // prepare cpt offset 
    offset = izeros(graph.n, 1)
    offset(1 until graph.n) = cumsum(ns)(0 until graph.n-1)
    
    if (mats.size > 1) {
      while (datasource.hasNext) {
        mats = datasource.next
        val sdata = mats(0)
        val state = mats(1)
        state <-- rand(state.nrows, state.ncols * opts.nsampls)
        state <-- (state >= 0.5)
        val innz = sdata match { 
          case ss: SMat => find(ss)
          case ss: GSMat => find(SMat(ss))
          case _ => throw new RuntimeException("sdata not SMat/GSMat")
        }
        for(i <- 0 until opts.nsampls){
        	state.asInstanceOf[FMat](innz + i * sdata.ncols *  graph.n) = 0f
        	state(?, i*sdata.ncols until (i+1)*sdata.ncols) = state(?, i*sdata.ncols until (i+1)*sdata.ncols) + (sdata.asInstanceOf[SMat] > 0)
        }
        datasource.putBack(mats,1)
      }
    }
    
    mm = modelmats(0)
    updatemats = new Array[Mat](1)
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols)
  }
  
  def getCpt(index: Mat) = {
    var cptindex = mm.zeros(index.nrows, index.ncols)
    for(i <-0 until index.ncols){
      cptindex(?, i) = mm(IMat(index(?, i)))
    }
    cptindex
  }
  
  def uupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    if (putBack < 0 || ipass == 0) user.set(1f)
    println("Inside uupdate with user:\n" + user)

    for (k <- 0 until opts.uiter) {
      for (c <- 0 until graph.ncolors) {
    	val ids = find(graph.colors == c)
    	val innz = sdata match {
    	  case ss: SMat => find(ss)
    	  case ss: GSMat => find(SMat(ss)) // This makes no sense
    	}
    	
    	// Get probabilities for getting 0
    	val user0 = user.copy
    	user0.asInstanceOf[FMat](ids, ?) = 0f
    	for (i <- 0 until opts.nsampls) {
    	  user0.asInstanceOf[FMat](innz + i * sdata.ncols * graph.n) = 0f
    	  //user0(?, i*sdata.ncols until (i+1)*sdata.ncols) <-- user0(?, i*sdata.ncols until (i+1)*sdata.ncols) + (sdata.asInstanceOf[SMat] > 0)
    	  user0(?, i*sdata.ncols until (i+1)*sdata.ncols) = user0(?, i*sdata.ncols until (i+1)*sdata.ncols) + (sdata.asInstanceOf[SMat] > 0)
    	}
    	val nodep0 = ln(getCpt(offset + SMat(iproject) * FMat(user0)) + opts.eps)        
    	val p0 = exp(SMat(pproject(ids, ?)) * FMat(nodep0))
    	
    	// Get probabilities for getting 1
    	val user1 = user.copy
    	user1.asInstanceOf[FMat](ids, ?) = 1f
    	for (i <- 0 until opts.nsampls) {
    	  user1.asInstanceOf[FMat](innz + i * sdata.ncols * graph.n) = 0f
    	  //user1(?, i*sdata.ncols until (i+1)*sdata.ncols) <-- user1(?, i*sdata.ncols until (i+1)*sdata.ncols) + (sdata.asInstanceOf[SMat] > 0)
    	  user1(?, i*sdata.ncols until (i+1)*sdata.ncols) = user1(?, i*sdata.ncols until (i+1)*sdata.ncols) + (sdata.asInstanceOf[SMat] > 0)
    	}
    	val nodep1 = ln(getCpt(offset + SMat(iproject) * FMat(user1)) + opts.eps)        
    	val p1 = exp(SMat(pproject(ids, ?)) * FMat(nodep1))
    	
    	// Now sample
    	val p = p1/(p0+p1)
    	val sample = rand(ids.length, sdata.ncols)
    	sample <-- (sample <= p) // This works because it's a full size matrix
    	println("For color group " + c + ", with sample matrix size " + size(sample) + ", and ids.t = " + ids.t + ", we just finished sampling:\n" + sample)
    	//user(ids, ?) <-- sample
    	user(ids,?) = sample
    	for (i <- 0 until opts.nsampls) {
    	  user.asInstanceOf[FMat](innz + i * sdata.ncols * graph.n) = 0f
    	  //user(?, i*sdata.ncols until (i+1)*sdata.ncols) <-- user(?, i*sdata.ncols until (i+1)*sdata.ncols) + (sdata.asInstanceOf[SMat] > 0)
    	  user(?, i*sdata.ncols until (i+1)*sdata.ncols) = user(?, i*sdata.ncols until (i+1)*sdata.ncols) + (sdata.asInstanceOf[SMat] > 0)
    	}
      } 
    }	
    println("Done with uupdate, user:\n" + user)
  }
  
  def mupdate(sdata:Mat, user:Mat, ipass:Int):Unit = {
    println("inside mupdate with user:\n" + user)
  	val um = updatemats(0) 
  	um.set(0)
	val index = IMat(offset + SMat(iproject) * FMat(user > 0))
	for (i <- 0 until user.ncols) {
	  um(index(?, i)) = um(index(?, i)) + 1f
	}
    um <-- um + opts.alpha
    	
	// normalize count matrix
	val norm = um.zeros(um.nrows, um.ncols)
	val l = um.length
    norm(0 until l-1 by 2) = um(1 until l by 2) 
    norm(1 until l by 2) = um(0 until l-1 by 2) 
    norm <-- norm + um
    um <-- um / norm
  }
  
  def evalfun(sdata:Mat, user:Mat):FMat = {  
  	row(0f, 0f)
  }
  
  def dobatch(gmats:Array[Mat], ipass:Int, here:Long) = {
    println("\n\n\nInside dobatch with here = " + here)
    println("also, our modelmats(0).t is:\n" + modelmats(0).t)
    val sdata = gmats(0)
    val user = if (gmats.length > 1) gmats(1) else BayesNet.reuseuser(gmats(0), opts.dim, 1f)
    uupdate(sdata, user, ipass)
    mupdate(sdata, user, ipass)
  }
  
  def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat = {
    val sdata = gmats(0)
    val user = if (gmats.length > 1) gmats(1) else BayesNet.reuseuser(gmats(0), opts.dim, 1f)
    uupdate(sdata, user, ipass)
    evalfun(sdata, user)
  } 
}

object BayesNet  {
  trait Opts extends Model.Opts {
    var nsampls = 1
    var alpha = 0.01f
    var uiter = 1
    var eps = 1e-9
  }
  
  class Options extends Opts {}
  
  def mkBayesNetmodel(dag: SMat, fopts:Model.Opts) = {
  	new BayesNet(dag, fopts.asInstanceOf[BayesNet.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
  	new IncNorm(nopts.asInstanceOf[IncNorm.Opts])
  } 
   
  def learner(dag0:SMat, mat0:Mat) = {
    class xopts extends Learner.Options with BayesNet.Opts with MatDS.Opts with IncNorm.Opts
    val opts = new xopts
    //opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    opts.batchSize = mat0.ncols
    opts.dim = dag0.ncols
    opts.isprob = false // Important! Otherwise it normalizes across columns!
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
     
}

/**
 * A graph structure for Bayesian Networks. Includes features for:
 * 
 *   (1) moralizing graphs, 'moral' matrix must be (i,j) = 1 means node i is connected to node j
 *   (2) coloring moralized graphs, not sure why there is a maxColor here, though...
 *
 * @param dag An adjacency matrix with a 1 at (i,j) if node i has an edge TOWARDS node j.
 * @param n The number of vertices in the graph. 
 */
class Graph(val dag: SMat, val n: Int) {
 
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
  
  /**
   * Computes the local offset matrix.
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
   * Computes the dag + identity matrix.
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
