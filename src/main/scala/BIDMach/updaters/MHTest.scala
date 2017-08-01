package BIDMach.updaters
 
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._
import edu.berkeley.bid.CUMACH

/**
 * Our fast MH test. See:
 *
 *  	An Efficient Minibatch Acceptance Test for Metropolis-Hastings, UAI 2017
 *  	Daniel Seita, Xinlei Pan, Haoyu Chen, John Canny
 *
 * This should be working for logistic regression. Here are a couple of
 * highlights for John Canny:
 * 
 * - It will not do any minibatch size incrementing here. That's for other parts
 * of the code. This keeps track of necessary statistics for our test. If we
 * accept, then we reset those stats, otherwise we update them.
 * 
 * - In particular, we need \Delta* and Var(\Delta*). Since \Delta* is of the form:
 * 
 * 		\Delta* = -\psi + (1/b)\sum_{i=1}^b Y_i 
 * 
 * for IID random variables Y_i, which represent a log of a probability ratio,
 * it suffices to compute the statistics as follows:
 * 
 * 		- \Delta* can be determined by keeping a running sum of all the Y_i terms.
 * 
 * 		- Due to the Central Limit Theorem assumption, the estimated variance of
 * 		\Delta* is the variance of the {Y_1, ... , Y_b} terms, *divided* by b.
 * 		Thus, we need Var(Y_i) which is estimated here as
 * 	
 * 				[ (1/b) \sum_{i=1}^b Y_i^2 ] - [ (1/b) \sum_{i=1}^b Y_i]^2
 * 
 * 		and this requires keeping a running sum of the Y_i^2 terms.
 * 
 * - An alternative idea to utilize updatemats and to avoid a second pass
 * through the evaluation would be to take a Taylor series expansion. However,
 * we do not support this now.
 * 
 * - POTENTIAL ISSUE: this code assumes we can input N and T perfectly, as we
 * did in the logistic regression example. If not, it might be better to provide
 * a generic parameter `K` for a constant that we multiply to the scores. For
 * instance, K could be our estimate of N/T.
 */
class MHTest(override val opts:MHTest.Opts = new MHTest.Options) extends Updater {

  var n2ld:DMat = null                 // X_c values for pre-selected \sigma.
  var deltaTheta:Array[Mat] = null     // Container for Gaussian noise from proposer.
  var tmpTheta:Array[Mat] = null       // Backup container to hold current theta.
  var proposedTheta:Array[Mat] = null  // The proposed theta (in our paper, it's theta').
  var modelmats:Array[Mat] = null      // The current theta.
  var updatemats:Array[Mat] = null     // Contains gradients (currently ignored!).
  var scores0:FMat = null              // Container for (N/T)*log p(x_i*|theta) terms.
  var scores1:FMat = null              // Container for (N/T)*log p(x_i*|theta') terms.
  var diff:FMat = null                 // Container for scores1-scores0.

  var newMinibatch:Boolean = true      // If true, need to run the proposer to get theta'.
  var b:Long = 0                       // Current minibatch size (also `b` in the paper).
  var N:Long = 0                       // Maximum minibatch size (i.e. all training data).
  var n:Float = 0f                     // The *number* of minibatches we are using.
  var psi:Float = 0f                   // \psi = log (1 * prop_ratio * prior_ratio)
  var T:Int = 1                        // The temperature of the distribution.
  var t:Int = 0                        // Current number of samples of theta.
  var sumOfValues:Float = 0f           // \sum_{i=1}^b (N/T)*log(p(x_i|theta')/p(x_i|theta)).
  var sumOfSquares:Float = 0f          // \sum_{i=1}^b ((N/T)*log(p(x_i|theta')/p(x_i|theta)))^2.
  var targetVariance:Float = 0f        // The target variance (so we only need one X_corr).


  /** 
   * Standard initialization. We have:
   * 
   * - n2ld loads the pre-computed X_c variable distribution.
   * - {delta,proposed,tmp}Theta initialized to zeros with correct dimensions.
   * - If desired, initialize modelmats with small values to break symmetry.
   * 
   * Note that the file for the norm2logdata should be in the correct directory.
   */
  override def init(model0:Model) = {
    model = model0;
    modelmats = model.modelmats
    updatemats = model.updatemats
    scores0 = zeros(1,model.datasource.opts.batchSize)
    scores1 = zeros(1,model.datasource.opts.batchSize)
    diff = zeros(1,model.datasource.opts.batchSize)
    T = opts.temp

    if (opts.Nknown) {
      N = opts.N
    } else {
      println("WARNING: opts.Nknown=false. (For now it should be true.)")
      throw new RuntimeException("Aborting now.")
    }
    
    val norm2logdata = loadDMat("data/MHTestCorrections/norm2log%d_20_%2.1f.txt" format 
        (opts.nn2l, opts.n2lsigma))
    n2ld = norm2logdata(?,0) \ cumsum(norm2logdata(?,1))
    targetVariance = opts.n2lsigma * opts.n2lsigma

    val nmats = modelmats.length;
    deltaTheta = new Array[Mat](nmats)
    proposedTheta = new Array[Mat](nmats)
    tmpTheta = new Array[Mat](nmats)

    for (i <- 0 until nmats) {
      deltaTheta(i) = modelmats(i).zeros(modelmats(i).nrows, modelmats(i).ncols)
      proposedTheta(i) = modelmats(i).zeros(modelmats(i).nrows, modelmats(i).ncols)
      tmpTheta(i) = modelmats(i).zeros(modelmats(i).nrows, modelmats(i).ncols)
    }

    if (opts.initThetaHere) {
      for (i <- 0 until nmats) {
        modelmats(i) <-- normrnd(0, 0.03f, modelmats(i).nrows, modelmats(i).ncols)
      }
    }
  }


  /**
   * This performs the update and the MH test based on a minibatch of data. The
   * original data is split up into equal-sized minibatches in the Learner code.
   * (The last minibatch is ignored since it generally has a different size.)
   * 
   * @param ipass The current pass over the full (training) data.
   * @param step Progress within the current minibatch, indicated as a numerical
   * 		index representing the starting column of this minibatch.
   * @param gprogress The percentage progress overall, when considering the
   *    datasource size and the number of (training) passes.
   */
  override def update(ipass:Int, step:Long, gprogress:Float):Unit = {
    if (newMinibatch) beforeEachMinibatch()
    b += model.datasource.opts.batchSize
    n += 1.0f

    // (Part 1) Compute scores for theta and theta', scaled by N/T.
    scores0 <-- (model.evalbatchg(model.datasource.omats, ipass, step) * (N/T.dv))
    if (scores0.length == 1) {
      throw new RuntimeException("Need individual scores, but getting a scalar.")
    }
    for (i <- 0 until modelmats.length) {
      modelmats(i) <-- proposedTheta(i)
    }
    scores1 <-- (model.evalbatchg(model.datasource.omats, ipass, step) * (N/T.dv))
    diff ~ scores1 - scores0

    // (Part 2) Update our \Delta* and sample variance of \Delta*.
    sumOfSquares += sum((diff)*@(diff)).v
    sumOfValues += sum(diff).v
    val deltaStar = sumOfValues/b.v - psi
    val sampleVariance = (sumOfSquares/b.v - ((sumOfValues/b.v)*(sumOfValues/b.v))) / b.v
    val numStd = deltaStar / math.sqrt(sampleVariance)
    var accept = false
    if (opts.verboseMH) debugPrints(sampleVariance, deltaStar)

    // (Part 3) Run our test! 
    // (Part 3.1) Take care of the full data case; this usually indicates a problem.
    if (ipass > 0 && b == N) {
      println("WARNING: test used entire dataset but variance is still too high.")
      println("  sample variance: %f, num std = %f" format (sampleVariance, numStd))
      if (opts.continueDespiteFull) {
        println("Nonetheless, we will accept/reject this sample based on Delta*") 
        newMinibatch = true
        if (deltaStar > 0) {
          accept = true
        }
      } else {
        throw new RuntimeException("Aborting program!")
      }
    }
    // (Part 3.2) Abnormally good or bad minibatches.
    else if (math.abs(numStd) > 5.0) {
      if (opts.verboseMH) {
        println("\tCASE 1: math.abs(numStd) = " +math.abs(numStd))
      }
      newMinibatch = true
      if (numStd > 0) {
        accept = true
      }
    }
    // (Part 3.3) If sample variance is too large, don't do anything.
    else if (sampleVariance >= targetVariance) {
      if (opts.verboseMH) {
        println("\tCASE 2: sample >= target = "+targetVariance)
      }
    } 
    // (Part 3.4) Finally, we can run our test by sampling a Gaussian and X_corr.
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

    // (Part 4) Reset parameters and use <-- to avoid alias problems.
    if (accept) {
      for (i <- 0 until modelmats.length) {
        tmpTheta(i) <-- modelmats(i) // Now tmpTheta has proposed theta.
      }     
    } else {
      for (i <- 0 until modelmats.length) {
        modelmats(i) <-- tmpTheta(i) // Now modelmats back to old theta.
      }
    }
    if (newMinibatch) afterEachMinibatch()
  }
  
   
  /**
   * Stuff we should do before each minibatch. This involves calling the
   * proposer, resetting some values, and saving the current model matrix into
   * `tmpTheta` so we can restore it later when needed.
   */
  def beforeEachMinibatch() {
    if (opts.verboseMH) println("\n\tNew minibatch!")
    randomWalkProposer()
    psi = ln(1).v // WARNING, symmetric proposals ONLY, since \psi(1,\theta,theta')=0.
    newMinibatch = false
    b = 0
    n = 0
    sumOfValues = 0f
    sumOfSquares = 0f
    for (i <- 0 until modelmats.length) {
      tmpTheta(i) <-- modelmats(i)
    }
  }

 
  /**
   * Stuff we should do after each minibatch. If desired, We repeatedly save the
   * model matrices and the minibatch size for analysis later. We also deal with
   * logic about the burn-in period, and also exit the program if we reach the
   * desired number of samples.
   */
  def afterEachMinibatch() {
    t += 1
    if (opts.collectData) {
      for (i <- 0 until modelmats.length) {
        saveFMat(opts.collectDataDir+ "theta_%d_%05d.fmat.lz4" format (i,t), FMat(modelmats(i)))
      }
      saveFMat(opts.collectDataDir+ "data_%05d.fmat.lz4" format (t), FMat(b))
    }
    if (t == opts.exitThetaAmount && opts.exitTheta) {
      println("Exiting code now since t=" +t)
      sys.exit
    }
    if (t == opts.burnIn) {
      println("ALERT: Past burn-in period. Now change temperature, proposer, etc.")
      T = opts.tempAfterBurnin
      opts.sigmaProposer = opts.sigmaProposerAfterBurnin
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
 

  /** This is for debugging. */
  def debugPrints(sampleVariance:Float, deltaStar:Float) {
    println("b="+b+", n="+n+", psi="+psi+ ", b-mbSize="+(b - model.datasource.opts.batchSize).toInt)
    println("mean(scores0) = "+mean(scores0,2).dv+", mean(scores1) = "+mean(scores1,2).dv)
    println("sampleVar = " +sampleVariance)
    println("delta* = " + deltaStar)
  }
  
}


object MHTest {

  trait Opts extends Updater.Opts {
    var N = 100000
    var temp = 1
    var tempAfterBurnin = 1
    var Nknown = true
    var n2lsigma = 1.0f
    var nn2l = 4000
    var sigmaProposer = 0.05f
    var sigmaProposerAfterBurnin = 0.05f
    var continueDespiteFull = true
    var verboseMH = false
    var collectData = false
    var collectDataDir = "tmp/"
    var exitTheta = false
    var exitThetaAmount = 3000
    var initThetaHere = false
    var burnIn = -1
  }
 
  class Options extends Opts {}
}
