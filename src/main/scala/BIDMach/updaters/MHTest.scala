package BIDMach.updaters
 
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._
import edu.berkeley.bid.CUMACH

/**
 * Our fast MH test. See:
 *
 *  An Efficient Minibatch Acceptance Test for Metropolis-Hastings, arXiv 2016
 *  Haoyu Chen, Daniel Seita, Xinlei Pan, John Canny
 *
 * This is a work in progress. A couple of important notes:
 * 
 * - It will not do any sort of minibatch size incrementing here. That's for
 * other parts of the code. This keeps track of necessary statistics for our
 * test. If we accept, then we clear those stats. Otherwise, we update them as
 * needed according to our minibatch.
 * 
 * - An alterantive idea to utilize updatemats and to avoid a second pass
 * through the evaluation would be to take a Taylor series expansion (see my
 * other notes). However, let's focus on the direct method from the paper for
 * now, but this will ignore updatemats.
 * 
 * - Why do `IncNorm.scala` and `IncMult.scala` create new modelmats in the
 * update method? Doesn't that allocate memory each time?
 * 
 * - Potental issue: assumes we can input N and T perfectly. If not it might be
 * better to provide a generic parameter `K` for a constant that we multiply,
 * and an estimate of the number of training data points.
 */
class MHTest(override val opts:MHTest.Opts = new MHTest.Options) extends Updater {

  var n2ld:DMat = null                 // X_c values for pre-selected \sigma.
  var deltaTheta:Array[Mat] = null     // Container for Gaussian noise from proposer.
  var tmpTheta:Array[Mat] = null       // Backup container to hold current theta.
  var proposedTheta:Array[Mat] = null  // The proposed theta.
  var modelmats:Array[Mat] = null      // The current theta.
  var updatemats:Array[Mat] = null     // Contains gradients (not currently used).
  var scores0:FMat = null              // Container for (N/T)*log p(x_i*|theta) terms.
  var scores1:FMat = null              // Container for (N/T)*log p(x_i*|theta') terms.
  var lRatios:FMat = null              // Holds (N/T)*log(ratio) across FULL minibatch.

  var newMinibatch:Boolean = true      // Whether we should run the proposer.
  var b:Long = 0                       // Current minibatch size (also `b` in the paper).
  var N:Long = 0                       // Maximum minibatch size (i.e. all training data).
  var n:Float = 0f                     // The *number* of MBs we are using.
  var logu:Float = 0f                  // log u, since we assume a symmetric proposer.
  var avgLogLL0:Float = 0f             // (N/(bT)) \sum_{i=1}^b log p(x_i*|theta).
  var avgLogLL1:Float = 0f             // (N/(bT)) \sum_{i=1}^b log p(x_i*|theta').
  var T:Int = opts.T                   // The temperature of the distribution.
  var t:Int = 0                        // The current total number of samples.


  /** 
   * Standard initialization. We have:
   * 
   * - n2ld loads the pre-computed X_c variable distribution
   * - {delta,proposed,tmp}Theta initialized to zeros to start
   * 
   * Note that the file for the norm2logdata should be in the correct directory.
   */
  override def init(model0:Model) = {
    model = model0;
    modelmats = model.modelmats
    updatemats = model.updatemats
    scores0 = zeros(1,model.datasource.opts.batchSize)
    scores1 = zeros(1,model.datasource.opts.batchSize)
    lRatios = zeros(1,opts.N)

    if (opts.Nknown) {
      N = opts.N
    } else {
      println("WARNING: opts.Nknown=false. (For now it should be true.)")
      throw new RuntimeException("Aborting now.")
    }
    
    val norm2logdata = loadDMat("data/MHTestCorrections/norm2log%d_20_%2.1f.txt" format 
        (opts.nn2l, opts.n2lsigma))
    n2ld = norm2logdata(?,0) \ cumsum(norm2logdata(?,1))

    val nmats = modelmats.length;
    deltaTheta = new Array[Mat](nmats)
    proposedTheta = new Array[Mat](nmats)
    tmpTheta = new Array[Mat](nmats)

    for (i <- 0 until nmats) {
      deltaTheta(i) = modelmats(i).zeros(modelmats(i).nrows, modelmats(i).ncols)
      proposedTheta(i) = modelmats(i).zeros(modelmats(i).nrows, modelmats(i).ncols)
      tmpTheta(i) = modelmats(i).zeros(modelmats(i).nrows, modelmats(i).ncols)
    }
  }


  /**
   * This performs the update and the MH test based on a minibatch of data. The
   * original data is split up into equal-sized minibatches in the Learner code.
   * (The last minibatch is ignored since it generally has a different size.)
   * 
   * Compute \Delta* (deltaStar) for our MH test using the current batch. Doing
   * so requires getting scores of omats, hence model.evalbatchg. And this also
   * has to be done twice, one for theta, one for theta'. We require the
   * variance, hence we need a vector, and will exit if not. For variance, the
   * workaround is to store individuals in `lRatios`. Don't forget to divide by
   * b due to the CLT assumption of \Delta*. NOTE: scores0 and scores1 are now
   * assigned using <-- to avoid alias.
   * 
   * @param ipass The current pass over the full (training) data.
   * @param step Progress within the current minibatch, indicated as a numerical
   * 		index representing the starting column of this minibatch.
   * @param gprogress The percentage progress overall, when considering the
   *    datasource size and the number of (training) passes.
   */
  override def update(ipass:Int, step:Long, gprogress:Float):Unit = {
    
    // Do some data collection and reset values.
    if (newMinibatch) beforeEachMinibatch()
    b += model.datasource.opts.batchSize
    n += 1.0f

    // Now compute scores; *should* be log p(...) stuff, if not we better check.
    scores0 <-- model.evalbatchg(model.datasource.omats, ipass, step) * (N/T.dv)
    if (scores0.length == 1) {
      throw new RuntimeException("Need individual scores, but getting a scalar.")
    }
    for (i <- 0 until modelmats.length) {
      modelmats(i) <-- proposedTheta(i)
    }
    scores1 <-- model.evalbatchg(model.datasource.omats, ipass, step) * (N/T.dv)

    avgLogLL0 = ((n-1)/n)*avgLogLL0 + mean(scores0,2).v/n
    avgLogLL1 = ((n-1)/n)*avgLogLL1 + mean(scores1,2).v/n
    val deltaStar = (avgLogLL1 - avgLogLL0) - logu
    val indices:IMat = irow((b - model.datasource.opts.batchSize).toInt until b.toInt)
    lRatios(indices) = (scores1 - scores0)
    val sampleVariance = variance(lRatios(0 until b.toInt)).v / b.toFloat
    
    // Now proceed with test. First, a few important values:
    val targetVariance = opts.n2lsigma * opts.n2lsigma
    val numStd = deltaStar / math.sqrt(sampleVariance)
    var accept = false
    
    if (opts.verboseMH) {
      println("b="+b+", n="+n+", logu="+logu+ ", b-mbSize="+(b - model.datasource.opts.batchSize).toInt)
      println("mean(scores0) = "+mean(scores0,2).dv+", mean(scores1) = "+mean(scores1,2).dv)
      println("avgLogLL0 = "+avgLogLL0+", avgLogLL1 = "+avgLogLL1)
      println("sampleVar = " +sampleVariance)
      println("delta* = " + deltaStar)
    }
    
    // Take care of abnormally good or bad minibatches (can probably be deleted).
    if (math.abs(numStd) > 5.0) {
      if (opts.verboseMH) {
        println("\tCASE 1: math.abs(numStd) = " +math.abs(numStd))
      }
      newMinibatch = true
      if (numStd > 0) {
        accept = true
      }
    }
    // If sample variance is too large, we cannot run the test.
    else if (sampleVariance >= targetVariance) {
      if (opts.verboseMH) {
        println("\tCASE 2: sample >= target = "+targetVariance)
      }
      if (ipass > 0 && b == N) {
        println("WARNING: test used entire dataset but variance is still too high.")
        println("  sample variance: %f, num std = %f" format (sampleVariance, numStd))
        if (opts.continueDespiteFull) {
          println("Nonetheless, we will accept/reject this sample based on Delta*") 
          newMinibatch = true
          if (deltaStar > 0) accept = true
        } else {
          throw new RuntimeException("Aborting program!")
        }
      }
    } 
    // Run the test by sampling a Gaussian and the X_c.
    else {
      newMinibatch = true
      val Xn = dnormrnd(0, math.sqrt(targetVariance-sampleVariance), 1, 1).dv
      val Xc = normlogrnd(1,1).dv
      val testStat = deltaStar + Xn + Xc
      if (opts.verboseMH) {
        println("\tCASE 3; with testStat = "+testStat)
      }
      if (testStat > 0) {
        accept = true
      }
    }

    // Reset parameters appropriately, using <-- to avoid alias issues.
    if (accept) {
      for (i <- 0 until modelmats.length) {
        tmpTheta(i) <-- modelmats(i) // Now tmpTheta has proposed theta.
      }     
    } else {
      for (i <- 0 until modelmats.length) {
        modelmats(i) <-- tmpTheta(i) // Now modelmats back to old theta.
      }
    }
    
    if (newMinibatch) {
      if (accept) {
        afterEachMinibatch(avgLogLL1)
      } else {
        afterEachMinibatch(avgLogLL0)
      }
    }
  }
  
   
  /** Stuff we should do before each minibatch. */
  def beforeEachMinibatch() {
    if (opts.verboseMH) println("\n\tNew minibatch!")
    randomWalkProposer()
    logu = ln(rand(1,1)).v
    newMinibatch = false
    b = 0
    n = 0
    avgLogLL0 = 0f
    avgLogLL1 = 0f
    for (i <- 0 until modelmats.length) {
      tmpTheta(i) <-- modelmats(i)
    }
    lRatios.clear
  }

 
  /**
   * Stuff we should do after each minibatch. 
   * 
   * @param scores The avg log likelihood of the current parameter (depends on
   * 		whether we accepted or rejected).
   */
  def afterEachMinibatch(scores:Float) {
    t += 1
    if (opts.collectData) {
      for (i <- 0 until modelmats.length) {
        saveFMat(opts.collectDataDir+ "theta_%d_%04d.fmat.lz4" format (i,t), FMat(modelmats(i)))
      }
      val a = row(b, scores)
      saveFMat(opts.collectDataDir+ "data_%04d.fmat.lz4" format (t), FMat(a))
    }
  }

 
  /**
   * A random walk proposer, but we should try and see if we can do something
   * fancier. Having the proposer as a simple \sigma*I (for identity matrix I),
   * however, means we can safely iterate through model matrices and update
   * independently. Doing it this way avoids excess memory allocation.
   */
  def randomWalkProposer() = {
    for (i <- 0 until modelmats.length) {
      normrnd(0, opts.sigmaProposer, deltaTheta(i))
      proposedTheta(i) <-- modelmats(i)
      proposedTheta(i) ~ proposedTheta(i) + deltaTheta(i)
    }
  }
 
  
  /**
   * Randomly generate sample(s) from the correction distribution X_c. It
   * samples values in (0,1) and then finds the x-positions (in some bounded
   * range such as [-10,10]) corresponding to those CDF values in X_c. This is
   * unchanged from John Canny's original implementation.
   * 
   * @param m Number of rows of random samples.
   * @param n Number of columns of random samples.
   */
  def normlogrnd(m:Int, n:Int):DMat = {
    val rr = drand(m, n)
    var i = 0
    while (i < rr.length) {
      val rv = rr.data(i)
      var top = n2ld.nrows
      var bottom = 0
      while (top - bottom > 1) {
        val mid = (top + bottom) / 2
        if (rv > n2ld(mid, 1)) {
          bottom = mid;
        } else {
          top = mid
        }
      }
      val y0 = n2ld(bottom, 1)
      val y1 = n2ld(math.min(top, n2ld.nrows-1), 1)
      val alpha = if (y1 != y0) ((rv - y0) / (y1 - y0)) else 0.0
      val x0 = n2ld(bottom, 0)
      val x1 = n2ld(math.min(top, n2ld.nrows-1), 0)
      val newx = alpha * x1 + (1-alpha) * x0
      rr.data(i) = newx
      i += 1
    }
    rr
  }
 
}


object MHTest {

  trait Opts extends Updater.Opts {
    var N = 100000
    var T = 1000
    var Nknown = true
    var n2lsigma = 1.0f
    var nn2l = 4000
    var sigmaProposer = 0.05f
    var continueDespiteFull = true
    var verboseMH = true
    var collectData = false
    var collectDataDir = "tmp/"
  }
 
  class Options extends Opts {}
}
