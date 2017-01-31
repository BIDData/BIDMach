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
 * This is a work in progress. For now assume a simple matrix as the data
 * source, or row vectors if dealing with vectors. Current issues:
 * 
 * 1. Since the update relies on minibatches, I assume we can start with the
 * minibatch provided by BIDMach on a given iteration. But how to get the next
 * one? If we can peek ahead and add the next datasource's minibatch, won't that
 * added minibatch also be what BIDMach provides as the starting minibatch for
 * the next iteration?
 * 
 * 2. Where can we get information about the log likelihood of the elements? We
 * need `log p(x_i*|theta)` and `log p(x_i*|theta)` for all minibatch elements.
 * Is this model.evalfun?
 * 
 * 3. Do we ignore updatemats?
 * 
 * 4. Where can we extract information about temperature?
 * 
 * 5. Later, how to extend this to the GPU case and save on memory?
 */
class MHTest(override val opts:MHTest.Opts = new MHTest.Options) extends Updater {

  // Put null stuff here that we initialize later. I'm not sure how much of this we need.
  var modelmats:Array[Mat] = null
  var updatemats:Array[Mat] = null
  var minibatch:Array[Mat] = null
  //var dsource:Datasource = null

  var theta:Mat = null
  var ttheta:Mat = null
  var thetaLogL:Mat = null
  var tthetaLogL:Mat = null
  var diff:Mat = null
  var thetaArrayMat:Array[Mat] = null

  var delta:Mat = null
  var n2ld:DMat = null


  /**
   * I think we need the model to get minibatches of data.
   */
  override def init(model0:Model) = {
    model = model0;
    modelmats = model.modelmats
    updatemats = model.updatemats
    val mm = modelmats(0)

    theta = mm.zeros(mm.nrows, mm.ncols)
    ttheta = mm.zeros(mm.nrows, mm.ncols)
    delta = mm.zeros(mm.nrows, mm.ncols)
    thetaArrayMat = new Array[Mat](1)

    val norm2logdata = loadDMat("norm2log%d_20_%2.1f.txt" format 
        (opts.nn2l, opts.n2lsigma))
    n2ld = norm2logdata(?,0) \ cumsum(norm2logdata(?,1))
    
    //data = model.datasource
  }


  /**
   * This performs the update and the MH test.
   *
   * @param ipass The current pass over the full (training) data.
   * @param step Progress within the current minibatch, indicated as a numerical
   * 		index representing the starting column of this minibatch.
   * @param gprogress The percentage progress overall, when considering the
   *    datasource size and the number of (training) passes.
   */
  override def update(ipass:Int, step:Long, gprogress:Float):Unit = {
    ttheta = proposer(theta, delta)
    var logu = ln(rand(1,1)).v;
    var done = false
    var mbStep = 100 // indicates how many elements in current minibatch
    var mbSize = 100 // indicates amount to increment minibatch size

    do {
      // Extract minibatch of data somehow. It has type Array[Mat], not Mat.
      // extract from column indices `step` to `step+mbStep`
      minibatch = null // TODO
      
      // Evaluate log likelihoods. I think `step` as third argument is OK.
      thetaLogL = model.evalbatch(minibatch, ipass, step)
      thetaArrayMat(0) = ttheta
      model.setmodelmats(thetaArrayMat)
      tthetaLogL = model.evalbatch(minibatch, ipass, step)

      // multiply diff by (N/T) to scale for data size and temperature?
      diff = thetaLogL - tthetaLogL

      // Then run the test with our minibatch
      val (moved,takeStep) = testFunction(diff, logu, false)
      done = moved
      
      if (done) {
        // TODO Record some statistics. Note that we already set the model to
        // have the new theta, so if we don't move, revert to the old theta.
        if (!takeStep) {
          thetaArrayMat(0) = theta
          model.setmodelmats(thetaArrayMat)
        }
      } else {
        // Increase the minibatch size 
        if (opts.fasterIncreaseUse) {
          mbStep = (mbStep*opts.fasterIncreaseFactor).toInt
        } else {
          mbStep += mbSize
        }
      }

    } while (!done)
  }
  

  /**
   * This will run the actual test function.
   * 
   * @param diff A vector containing scaled log likelihood ratios for each
   *    element in the minibatch
   * @param logu The \psi(u,theta,theta') term, which is log(u) for symmetric
	 *    proposals that we use.
	 * @param full A boolean indicating if this minibatch is equivalent to the
	 * 		full data. This should ideally be false always.
	 * @return A boolean (b1,b2) where b1=true means done, otherwise increase
	 * 		minibatch size, and b2=true means we moved to the new theta, otherwise
	 * 		stick with old theta.
   */
  def testFunction(diff:Mat, logu:Double, full:Boolean):(Boolean,Boolean) = {
    val tvar = variance(diff).dv/diff.length;
    val targvar = opts.n2lsigma * opts.n2lsigma;
    val x = mean(diff).dv;
    val ns = x / math.sqrt(tvar);
    
    if (math.abs(ns) > 5) {
      // Get abnormally large |ns| values out of the way.
      if (ns > 0) (true, true) else (true, false)
    } else {
      if (tvar >= targvar) {
        // If tvar >= targvar, we need to decrease sample variance of data.
        if (full) { 
          if (opts.continueDespiteFull) {
            println("Warning: test failed variance condition, var=%f nstd = %f" 
                format (tvar, ns));
            if (x > 0) (true, true) else (true, false)
          } else {
            throw new RuntimeException("Test failed variance condition, var=%f, nstd = %f" 
                format (tvar, ns))
          }
        } else {
          (false, false)
        }
      } else {
        // Otherwise, we can run our test.
        val xn = dnormrnd(0, math.sqrt(targvar-tvar), 1, 1).dv
        val xc = normlogrnd(1,1).dv
        if ((x + xn + xc) > 0) {
          (true, true)
        } else {
          (true, false)
        }
      }
    }
  }
  
   
  /**
   * A proposer, which is currently the simple random walk but we should try and
   * see if we can do something fancier.
   * 
   * @param t Our current theta.
   * @param d Container to hold sampled noise.
   */
  def proposer(t:Mat, d:Mat) = {
    normrnd(0, opts.sigmaProposer, d)
    t+d
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
    val fasterIncreaseUse = false
    val fasterIncreaseFactor = 2.0
    val n2lsigma = 0.9
    val nn2l = 2000
    val sigmaProposer = 0.05
    val continueDespiteFull = true
  }
 
  class Options extends Opts {}
}