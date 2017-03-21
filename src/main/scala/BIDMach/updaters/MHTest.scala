package BIDMach.updaters
 
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._
import BIDMach.models.Model._
import edu.berkeley.bid.CUMACH
import BIDMach.Learner

/**
 * Our fast MH test. See:
 *
 *  	An Efficient Minibatch Acceptance Test for Metropolis-Hastings, arXiv 2016
 *  	Haoyu Chen, Daniel Seita, Xinlei Pan, John Canny
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
 * 		\Delta* = - log(u) + (1/b)\sum_{i=1}^b Y_i 
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
  var logu:Float = 0f                  // log u, since we assume a symmetric proposer.
  var T:Int = 1                        // The temperature of the distribution.
  var _t:Int = 0                       // Current number of samples of theta.
  var sumOfValues:Float = 0f           // \sum_{i=1}^b (N/T)*log(p(x_i|theta')/p(x_i|theta)).
  var sumOfSquares:Float = 0f          // \sum_{i=1}^b ((N/T)*log(p(x_i|theta')/p(x_i|theta)))^2.
  var targetVariance:Float = 0f        // The target variance (so we only need one X_corr).

  // Daniel: experimental, for the SMF.
  var currentSizeSMF:Int = -1;
  var adagrad:ADAGrad = null;
  var tmpMomentum:Array[Mat] = null
  var acceptanceRate:Mat = null

  /** 
   * Standard initialization. We have:
   * 
   * - n2ld loads the pre-computed X_c variable distribution.
   * - {delta,proposed,tmp}Theta initialized to zeros with correct dimensions.
   * - If desired, initialize modelmats with small values to break symmetry.
   * - If desired, initialize an internal ADAGrad updater.
   * 
   * Note that the file for the norm2logdata should be in the correct directory.
   */
  override def init(model0:Model) = {
    model = model0;
    modelmats = model.modelmats
    updatemats = model.updatemats
    acceptanceRate = zeros(1, opts.exitThetaAmount * 10)
    if (opts.smf) {
      val numnnz = model.asInstanceOf[SMF].getNonzeros()
      if (numnnz < 0) {
        println("Something wrong happened, numnnz="+numnnz)
        sys.exit
      }
      scores0 = zeros(1, numnnz*10)
      scores1 = zeros(1, numnnz*10)
      diff = zeros(1, numnnz*10)
    } else {
      scores0 = zeros(1, model.datasource.opts.batchSize)
      scores1 = zeros(1, model.datasource.opts.batchSize)
      diff = zeros(1, model.datasource.opts.batchSize)
    }
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
    
    if (opts.smf) {
      // This should force adagrad.momentum(i) = momentum(i) in the rest of this code.
      adagrad = new ADAGrad(opts.asInstanceOf[ADAGrad.Opts])
      adagrad.init(model)
      tmpMomentum = new Array[Mat](nmats)
      for (i <- 0 until nmats) {
        tmpMomentum(i) = modelmats(i).zeros(modelmats(i).nrows, modelmats(i).ncols)
      }
    }
  }
  

  /**
   * This performs the update and the MH test based on a minibatch of data. The
   * original data is split up into equal-sized minibatches in the Learner code.
   * (The last minibatch is ignored since it generally has a different size.)
   * 
   * SMF.scala necessitates extra cases to handle the varying batch sizes. They
   * differ across "minibatches" so the scores at "the end" have to be cleared
   * (but since scores are only for current MB, just call "clear"), the number
   * of nonzeros have to be computed, and then the scores are copied to the
   * appropriate interval. EDIT: ugh, never mind, doesn't work ...
   * 
   * @param ipass The current pass over the full (training) data.
   * @param step Progress within the current minibatch, indicated as a numerical
   * 		index representing the starting column of this minibatch.
   * @param gprogress The percentage progress overall, when considering the
   *    datasource size and the number of (training) passes.
   */
  override def update(ipass:Int, step:Long, gprogress:Float):Unit = {
    if (newMinibatch) beforeEachMinibatch(ipass, step, gprogress)
    n += 1.0f

    // (Part 1) Compute scores for theta and theta', scaled by N/T.
    if (opts.smf) {
      currentSizeSMF = model.datasource.omats(0).nnz
      b += currentSizeSMF
      scores0.clear
      scores0(0 -> currentSizeSMF) = (model.evalbatchg(model.datasource.omats, ipass, step) * (N/T.dv)).t
    } else {
      scores0 <-- (model.evalbatchg(model.datasource.omats, ipass, step) * (N/T.dv))
      b += scores0.length
    }
    if (scores0.length == 1) {
      throw new RuntimeException("Need individual scores, but getting a scalar.")
    }

    for (i <- 0 until modelmats.length) {
      modelmats(i) <-- proposedTheta(i)
    }
    if (opts.smf) {
      scores1.clear
      scores1(0 -> currentSizeSMF) = (model.evalbatchg(model.datasource.omats, ipass, step) * (N/T.dv)).t
      diff.clear
      diff(0 -> currentSizeSMF) = scores1(0 -> currentSizeSMF) - scores0(0 -> currentSizeSMF)
    } else {
      scores1 <-- (model.evalbatchg(model.datasource.omats, ipass, step) * (N/T.dv))
      diff ~ scores1 - scores0
    }

    // (Part 2) Update our \Delta* and sample variance of \Delta*.
    if (opts.smf) {
      sumOfSquares += sum((diff(0 -> currentSizeSMF)) *@ (diff(0 -> currentSizeSMF))).v
      sumOfValues += sum(diff(0 -> currentSizeSMF)).v
    } else {
      sumOfSquares += sum((diff)*@(diff)).v
      sumOfValues += sum(diff).v
    }
    val deltaStar = sumOfValues/b.v - logu
    val sampleVariance = (sumOfSquares/b.v - ((sumOfValues/b.v)*(sumOfValues/b.v))) / b.v    
    val numStd = deltaStar / math.sqrt(sampleVariance)
    var accept = false
    if (opts.verboseMH) debugPrints(sampleVariance, deltaStar, numStd, sumOfValues/b.v)

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
    else if (math.abs(numStd) > 10.0) {
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
        println("\tCASE 3; with testStat = %1.4f (Xn = %1.4f, Xc = %1.4f)" format (testStat, Xn, Xc))
      }
      if (testStat > 0) {
        accept = true
      }
    }

    // (Part 4) Reset parameters and use <-- to avoid alias problems.
    if (accept) {
      if (opts.verboseMH) println("\tACCEPT")
      for (i <- 0 until modelmats.length) {
        tmpTheta(i) <-- modelmats(i) // Now tmpTheta has proposed theta.
      }
      acceptanceRate(_t) = 1
    } else {
      if (opts.verboseMH) println("\treject")
      for (i <- 0 until modelmats.length) {
        modelmats(i) <-- tmpTheta(i) // Now modelmats back to old theta.
        if (opts.smf) {
          adagrad.momentum(i) <-- tmpMomentum(i) // Revert ADAGrad momentum.
        }
      }
      acceptanceRate(_t) = 0
    }
    
    if (newMinibatch) afterEachMinibatch(ipass, gprogress)
  }
  
   
  /**
   * Stuff we should do before each minibatch. This involves calling the
   * proposer, resetting some values, and saving the current model matrix into
   * `tmpTheta` so we can restore it later when needed. Here, we want to set the
   * proposer matrices, so that when we continue in uupdate, we have the current
   * and proposed model matrices stored in modelmats and proposedTheta,
   * respectively.
   * 
   * Also, we have a different (i.e. better!) proposer with ADAGrad. The update
   * *should* affect all of the modelmats(i) due to aliasing (since it changes
   * adagrad.modelmats(i)). However, this doesn't put it in proposedTheta, so
   * here's a workaround: get the modelmats stored into tmpTheta. Then do the
   * update, which will update modelmats to the proposed matrices. Then copy
   * those into propsoedTheta, and then get current modelmats back to tmpTheta
   * (i.e. so modelmats remains the same before and after, and it's just the
   * proposedTheta which changes). With momentum, fortunately it's simpler, we
   * have that in adagrad.momentum and simply copy the old state into
   * tmpMomentum.
   */
  def beforeEachMinibatch(ipass:Int, step:Long, gprogress:Float) {
    if (opts.verboseMH) println("\n\tNew minibatch!")

    for (i <- 0 until modelmats.length) {
      tmpTheta(i) <-- modelmats(i)
      if (opts.smf) {
        tmpMomentum(i) <-- adagrad.momentum(i)
      }
    } 

    if (opts.smf) {
      adagrad.update(ipass, step, gprogress)
      for (i <- 0 until modelmats.length) {
        proposedTheta(i) <-- adagrad.modelmats(i) // adagrad.modelmats(i) = modelmats(i)
        modelmats(i) <-- tmpTheta(i) // Should make adagrad.modelmats(i) back to what it was before.
      } 
    } else {
      randomWalkProposer()
    }

    logu = ln(rand(1,1)).v
    newMinibatch = false
    b = 0
    n = 0
    sumOfValues = 0f
    sumOfSquares = 0f
  }

 
  /**
   * Stuff we should do after each minibatch. If desired, We repeatedly save the
   * model matrices and the minibatch size for analysis later. We also deal with
   * logic about the burn-in period, and also exit the program if we reach the
   * desired number of samples.
   */
  def afterEachMinibatch(ipass:Int, gprogress:Float) {
    _t += 1
    if (opts.collectData) {
      for (i <- 0 until modelmats.length) {
        saveFMat(opts.collectDataDir+ "theta_%d_%04d.fmat.lz4" format (i,_t), FMat(modelmats(i)))
      }
      saveFMat(opts.collectDataDir+ "data_%04d.fmat.lz4" format (_t), FMat(b))
    }
    if (_t == opts.exitThetaAmount && opts.exitTheta) {
      println("Exiting code now since t=" +_t)
      sys.exit
    }
    if (_t == opts.burnIn) {
      println("ALERT: Past burn-in period. Now change temperature, proposer, etc.")
      T = opts.tempAfterBurnin
      opts.sigmaProposer = opts.sigmaProposerAfterBurnin
    }
    if (opts.smf) {
      if (opts.saveAcceptRate && (ipass+1) == opts.asInstanceOf[Learner.Options].npasses
          && gprogress > 0.99) {
        val mom = adagrad.opts.momentum(0).v
        val lr = adagrad.opts.lrate.v
        val lang = adagrad.opts.langevin(0).v
        saveMat(opts.acceptRateDir+"arate_%1.3f_%1.4f_%1.3f.mat.lz4" format (mom,lr,lang), 
            acceptanceRate(0 -> _t))
      }
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
  def debugPrints(sampleVariance:Float, deltaStar:Float, numStd:Double, sumDivB:Float) {
    val s1 = mean(scores1(0 -> b.toInt)).dv
    val s0 = mean(scores0(0 -> b.toInt)).dv
    println("b = %d, n = %d, logu = %1.4f" format (b, n.toInt, logu))
    println("mean(scores1) (%1.4f) - mean(scores0) (%1.4f) = %1.4f" format (s1, s0, s1-s0))
    println("maxi(scores1) = "+maxi(scores1(0 -> b.toInt))+", maxi(scores0) = "+maxi(scores0(0 -> b.toInt)))
    println("mini(scores1) = "+mini(scores1(0 -> b.toInt))+", mini(scores0) = "+mini(scores0(0 -> b.toInt)))
    println("delta^* (%1.4f) = sumDivB (%1.4f) - logu (%1.4f)" format (deltaStar, sumDivB, logu))
    println("sampleVar = %1.4f, numStd = %1.4f" format (sampleVariance, numStd))
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
    var smf = false
    var saveAcceptRate = false
    var acceptRateDir = "tmp/"
  }
 
  class Options extends Opts {}
}
