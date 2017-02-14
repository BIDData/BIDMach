package BIDMach.updaters
 
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._
import edu.berkeley.bid.CUMACH

/**
 * This is an approximate MH test based on the algorithm described in:
 *
 *  	Austerity in MCMC Land: Cutting the Metropolis-Hastings Budget, ICML 2014
 *  	Anoop Korattikara, Yutian Chen, Max Welling
 *
 * TODO This is a major work in progress.
 * 
 * - It assumes a fixed `epsilon` within (0,1).
 */
class AustereMH(override val opts:AustereMH.Opts = new AustereMH.Options) extends Updater {

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
  var mu0:Float = 0f                   // log(u)/N, assuming symmetric  proposer.
  var T:Int = 1                        // The temperature of the distribution.
  var t:Int = 0                        // Current number of samples of theta.
  var sumOfValues:Float = 0f           // \sum_{i=1}^b (N/T)*log(p(x_i|theta')/p(x_i|theta)).
  var sumOfSquares:Float = 0f          // \sum_{i=1}^b ((N/T)*log(p(x_i|theta')/p(x_i|theta)))^2.


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

    // (Part 2) Update \bar{l} and \bar{l^2} (using the paper's notation).
    sumOfSquares += sum((diff)*@(diff)).v
    sumOfValues += sum(diff).v
    val lBar = sumOfValues/b.v
    val l2Bar = sumOfSquares/b.v
    val sampleStDev = sqrt((l2Bar - lBar*lBar) * (b/(b-1)))    
    val stDevOfLBar = (sampleStDev/sqrt(b)) * sqrt(1 - (b.v-1)/(N-1))
    val testStat = (lBar - mu0) / stDevOfLBar
    if (opts.verbose) debugPrints()
    
    // (Part 3) Now run their test.
    var accept = false
    val delta = (1.0 - normcdf(abs(testStat))).dv
    if (delta < opts.epsi) {
      newMinibatch = true
      if (lBar > mu0) {
        accept = true
      } else {
        accept = false
      }
    } else {
      newMinibatch = false
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
    if (newMinibatch && accept) afterEachMinibatch()
  }
  
   
  /**
   * Stuff we should do before each minibatch. This involves calling the
   * proposer, resetting some values, and saving the current model matrix into
   * `tmpTheta` so we can restore it later when needed.
   */
  def beforeEachMinibatch() {
    if (opts.verbose) println("\n\tNew minibatch!")
    randomWalkProposer()
    mu0 = ln(rand(1,1)).v / N
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
        saveFMat(opts.collectDataDir+ "theta_%d_%04d.fmat.lz4" format (i,t), FMat(modelmats(i)))
      }
      saveFMat(opts.collectDataDir+ "data_%04d.fmat.lz4" format (t), FMat(b))
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
 
  
  /** This is for debugging. */
  def debugPrints() {
    println("b="+b+", n="+n+", mu0="+mu0+ ", b-mbSize="+(b - model.datasource.opts.batchSize).toInt)
    println("mean(scores0) = "+mean(scores0,2).dv+", mean(scores1) = "+mean(scores1,2).dv)
  }
  
}


object AustereMH {

  trait Opts extends Updater.Opts {
    var epsi = 0.05
    var fixedEpsi = true
    var N = 100000
    var temp = 1
    var tempAfterBurnin = 1
    var Nknown = true
    var sigmaProposer = 0.05f
    var sigmaProposerAfterBurnin = 0.05f
    var continueDespiteFull = true
    var verbose = false
    var collectData = false
    var collectDataDir = "tmp/"
    var exitTheta = false
    var exitThetaAmount = 3000
    var initThetaHere = false
    var burnIn = -1
  }
 
  class Options extends Opts {}
}
