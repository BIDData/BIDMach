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
 * - TODO later: temperature info, GPU usage, etc.
 */
class MHTest(override val opts:MHTest.Opts = new MHTest.Options) extends Updater {

  var n2ld:DMat = null                 // X_c values for pre-selected \sigma.
  var deltaTheta:Array[Mat] = null     // Container for Gaussian noise from proposer.
  var proposedTheta:Array[Mat] = null  // The proposed theta.
  var tmpTheta:Array[Mat] = null       // Backup container to hold current theta.
  var modelmats:Array[Mat] = null      // The current theta.
  var updatemats:Array[Mat] = null     // Contains gradients (not currently used).

  var newMinibatch:Boolean = true      // Whether we should run the proposer.
  var mbSize:Long = 0                  // Current minibatch size.
  var N:Long = 0                       // Maximum minibatch size (i.e. all training data).
  var n:Float = 0f                     // The number of MBs we are using.
  var logu:Float = 0f                  // log u, since we assume a symmetric proposer.
  var avgLogLLCurrent:Float = 0f       // (N/b) \sum_{i=1}^b log p(x_i* | theta)
  var avgLogLLProposed:Float = 0f      // (N/b) \sum_{i=1}^b log p(x_i* | theta')


  /**
   * Standard initialization.
   */
  override def init(model0:Model) = {
    model = model0;
    modelmats = model.modelmats
    updatemats = model.updatemats

    val norm2logdata = loadDMat("norm2log%d_20_%2.1f.txt" format 
        (opts.nn2l, opts.n2lsigma))
    n2ld = norm2logdata(?,0) \ cumsum(norm2logdata(?,1))

    for (i <- 0 until modelmats.length) {
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
   * @param ipass The current pass over the full (training) data.
   * @param step Progress within the current minibatch, indicated as a numerical
   * 		index representing the starting column of this minibatch.
   * @param gprogress The percentage progress overall, when considering the
   *    datasource size and the number of (training) passes.
   */
  override def update(ipass:Int, step:Long, gprogress:Float):Unit = {
    
    // Preliminary steps and data clearing/preparation.
    if (ipass == 0) N = step 
    if (newMinibatch) {
      randomWalkProposer()
      logu = ln(rand(1,1)).v
      newMinibatch = false
      mbSize = 0
      n = 0
      avgLogLLCurrent = 0f
      avgLogLLProposed = 0f
      for (i <- 0 until modelmats.length) {
        tmpTheta(i) = modelmats(i)
      }
    }
    mbSize += model.datasource.opts.batchSize
    n += 1

    // TODO this is not completely correct.
    // if ipass == 0, N is not yet correct. How to fix?
    // Also, check to make sure datasource.omats is the correct one to use.

    // Compute \Delta* (deltaStar) for the acceptance test.
    val newC = N * (model.evalbatchg(model.datasource.omats, ipass, step)).v
    avgLogLLCurrent = ((n-1)/n)*avgLogLLCurrent + newC/n
    for (i <- 0 until modelmats.length) {
      modelmats(i) = proposedTheta(i)
    }
    val newP = N * (model.evalbatchg(model.datasource.omats, ipass, step)).v
    avgLogLLProposed = ((n-1)/n)*avgLogLLProposed + newP/n
    val deltaStar = (avgLogLLProposed - avgLogLLCurrent) - logu

    // TODO HOW DO I COMPUTE THIS (EFFICIENTLY)? This requires individual
    // scores, but we are only given the averages in model.evalbatchg
    val sampleVariance = 0f 

    val targetVariance = opts.n2lsigma * opts.n2lsigma
    val numStd = deltaStar / math.sqrt(sampleVariance)
    var accept = false
    
    // Take care of abnormally good or bad minibatches (can probably be deleted).
    if (math.abs(numStd) > 5.0) {
      newMinibatch = true
      if (numStd > 0) accept = true
    }
    // If sample variance is too large, we cannot run the test.
    else if (sampleVariance >= targetVariance) {
      if (ipass > 0 && mbSize == N) {
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
      if (deltaStar + Xn + Xc > 0) accept = true
    }

    // Reset parameters if the proposal was rejected or if we need more data.
    if (!accept) {
      for (i <- 0 until modelmats.length) {
        modelmats(i) = tmpTheta(i)
      }
    }
  }
 

  /**
   * A random walk proposer, but we should try and see if we can do something
   * fancier. Having the proposer as a simple \sigma*I (for identity matrix I),
   * however, means we can safely iterate through model matrices and update
   * independently.
   */
  def randomWalkProposer() = {
    for (i <- 0 until modelmats.length) {
      normrnd(0, opts.sigmaProposer, deltaTheta(i))
      proposedTheta(i) ~ modelmats(i) + deltaTheta(i)
    }
  }
 
  
  /**
   * Randomly generate sample(s) from the correction distribution X_c. It
   * samples values in (0,1) and then finds the x-positions (in some bounded
   * range such as [-10,10]) corresponding to those CDF values in X_c.
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
    val n2lsigma = 0.9f
    val nn2l = 2000
    val sigmaProposer = 0.05f
    val continueDespiteFull = true
  }
 
  class Options extends Opts {}
}