package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMach._
import BIDMach.datasources._
import BIDMach.updaters._
import java.lang.ref._
import jcuda.NativePointerObject
import java.lang.Math;

/**
 * Independent Component Analysis, using FastICA. It has the ability to center and whiten data. It is 
 * based on the method presented in:
 * 
 * A. Hyvärinen and E. Oja. Independent Component Analysis: Algorithms and Applications. 
 * Neural Networks, 13(4-5):411-430, 2000.
 * 
 * In particular, we provide the logcosh, exponential, and kurtosis "G" functions.
 * 
 * This algorithm computes the following modelmats array:
 *   - modelmats(0) stores the inverse of the mixing matrix. If X = A*S represents the data, then it's the 
 *     estimated A^-1^, which we assume is square and invertible for now.
 *   - modelmats(1) stores the mean vector of the data, which is computed entirely on the first pass. This
 *     means once we estimate A^-1^ in modelmats(0), we need to first shift the data by this amount, and
 *     then multiply to recover the (centered) sources. Example:
 * {{{
 * modelmats(0) * (data - modelmats(1))
 * }}}
 *     Here, data is an n x N matrix, whereas modelmats(1) is an n x 1 matrix. For efficiency reasons, we
 *     assume a constant batch size for each block of data so we take the mean across all batches. This is 
 *     true except for (usually) the last batch, but this almost always isn't enough to make a difference.
 *     
 * Thus, modelmats(1) helps to center the data. The whitening in this algorithm happens during the updates
 * to W in both the orthogonalization and the fixed point steps. The former uses the computed covariance 
 * matrix and the latter relies on an approximation of W^T^*W to the inverse covariance matrix. It is fine
 * if the data is already pre-whitened before being passed to BIDMach.
 *     
 * Currently, we are thinking about the following extensions:
 *   - Allowing ICA to handle non-square mixing matrices. Most research about ICA assumes that A is n x n.
 *   - Improving the way we handle the computation of the mean, so it doesn't rely on the last batch being
 *     of similar size to all prior batches. Again, this is minor, especially for large data sets.
 *   - Thinking of ways to make this scale better to a large variety of datasets
 * 
 * For additional references, see Aapo Hyvärinen's other papers, and visit:
 *   http://research.ics.aalto.fi/ica/fastica/
 */
class ICA(override val opts:ICA.Opts = new ICA.Options) extends FactorModel(opts) {
  
  // Some temp variables. The most important one is mm, which is our W = A^{-1}.
  var mm:Mat = null
  var batchIteration = 0.0f
  var G_fun: Mat=>Mat = null
  var g_fun: Mat=>Mat = null
  var g_d_fun: Mat=>Mat = null
  var stdNorm:FMat = null

  var debug = false
  
  override def init() {
    super.init()
    if (refresh) {
      mm = modelmats(0)
      setmodelmats(Array(mm, mm.zeros(mm.nrows,1)))
    }
    updatemats = new Array[Mat](2)
    updatemats(0) = mm.zeros(mm.nrows, mm.nrows)
    updatemats(1) = mm.zeros(mm.nrows,1) // Keep to avoid null pointer exceptions, but we don't use it
    opts.G_function match {
      case "logcosh" => {
        G_fun = G_logcosh; g_fun = g_logcosh; g_d_fun = g_d_logcosh; 
        stdNorm = FMat(0.375);
      }
      case "exponent" => {
        G_fun = G_exponent; g_fun = g_exponent; g_d_fun = g_d_exponent; 
        stdNorm = FMat(-1.0 / sqrt(2.0));
      }
      case "kurtosis" => {
        G_fun = G_kurtosis; g_fun = g_kurtosis; g_d_fun = g_d_kurtosis;
        stdNorm = FMat(0.75);
      }
      case _ => throw new RuntimeException("opts.G_function is not a valid value: " + opts.G_function)
    }
  }
    
  /** 
   * Store data in "user" for use in the next mupdate() call, and updates the moving average if necessary.
   * Also "orthogonalizes" the model matrix after each update, as required by the algorithm.
   * 
   * First, it checks if this is the first pass over the data, and if so, updates the moving average assuming
   * that the number of data samples in each block is the same for all blocks. After the first pass, the data 
   * mean vector is fixed in modelmats(1). Then the data gets centered via: "data ~ data - modelmats(1)".
   * 
   * We also use "user ~ mm * data" to store all (w_j^T^) * (x^i^) values, where w_j^T^ is the j^th^ row of
   * our estimated W = A^-1^, and x^i^ is the i^th^ sample in this block of data. These values are later used
   * as part of fixed point updates.
   * 
   * @param data An n x batchSize matrix, where each column corresponds to a data sample.
   * @param user An intermediate matrix that stores (w_j^T^) * (x^i^) values.
   * @param ipass The current pass through the data.
   */
  def uupdate(data : Mat, user : Mat, ipass : Int, pos:Long) {
    if (ipass == 0) {
      batchIteration = batchIteration + 1.0f
      modelmats(1) <-- (modelmats(1)*(batchIteration-1) + mean(data,2)) / batchIteration 
    }
    data ~ data - modelmats(1)
    mm <-- orthogonalize(mm,data)
    user ~ mm * data
  }
  
  /**
   * This performs the matrix fixed point update to the estimated W = A^{-1}:
   * 
   *   W^+^ = W + diag(alpha,,i,,) * [ diag(beta,,i,,) - Expec[g(Wx)*(Wx)^T^] ] * W,
   * 
   * where g = G', beta,,i,, = -Expec[(Wx),,i,,g(Wx),,i,,], and alpha,,i,, = -1/(beta,,i,, - Expec[g'(Wx),,i,,]). 
   * We need to be careful to take expectations of the appropriate items. The gwtx and g_wtx terms are matrices 
   * with useful intermediate values that represent the full data matrix X rather than a single column/element x.
   * The above update for W^+^ goes in updatemats(0), except the additive W since that should be taken care of by
   * the ADAGrad updater.
   * 
   * I don't think anything here changes if the data is not white, since one of Hyvärinen's papers implied
   * that the update here includes an approximation to the inverse covariance matrix.
   * 
   * @param data An n x batchSize matrix, where each column corresponds to a data sample.
   * @param user An intermediate matrix that stores (w_j^T^) * (x^i^) values.
   * @param ipass The current pass through the data.
   */
  def mupdate(data : Mat, user : Mat, ipass : Int, pos:Long) {
    val gwtx = g_fun(user)
    val g_wtx = g_d_fun(user)
    val termBeta = mkdiag( -mean(user *@ gwtx, 2) )
    val termAlpha = mkdiag( -1.0f / (getdiag(termBeta) - (mean(g_wtx,2))) )
    val termExpec = (gwtx *^ user) / data.ncols
    updatemats(0) <-- termAlpha * (termBeta + termExpec) * mm
  }
    
  /**
   * Currently, this computes the approximation of negentropy, which is the objective function to maximize.
   * 
   * To understand this, let w be a single row vector of W, let x be a single data vector, and let v be a
   * standard normal random variable. To find this one independent component, we maximize
   * 
   *   J(w^T^x) \approx ( Expec[G(w^T^x)] - Expec[G(v)] )^2^,
   * 
   * where G is the function set at opts.G_function. So long as the W matrix (capital "W") is orthogonal, 
   * which we do enforce, then w^T^x satisfies the requirement that the variance be one. To extend this to
   * the whole matrix W, take the sum over all the rows, so the problem is: maximize{ \sum,,w,, J(w^T^x) }.
   * 
   * On the other hand, the batchSize should be much greater than one, so "data" consists of many columns.
   * Denoting the data matrix as X, we can obtain the expectations by taking the sample means. In other words,
   * we take the previous "user" matrix, W*X, apply the function G to the data, and THEN take the mean across
   * rows, so mean(G(W*X),2). The mean across rows gives what we want since it's applying the same row of W
   * to different x (column) vectors in our data.
   * 
   * @param data An n x batchSize matrix, where each column corresponds to a data sample.
   * @param user An intermediate matrix that stores (w_j^T^) * (x^i^) values.
   * @param ipass The current pass through the data.
   */
  def evalfun(data : Mat, user : Mat, ipass : Int, pos:Long) : FMat = {
    val big_gwtx = G_fun(user)
    val rowMean = FMat(mean(big_gwtx,2)) - stdNorm
    return sum(rowMean *@ rowMean)
  }
  
  /** Assumes G(x) = log(cosh(x)), a good general-purpose contrast function. */
  private def G_logcosh(m : Mat) : Mat = {
    return ln(cosh(m))
  }
  
  /** Assumes g(x) = d/dx log(cosh(x)) = tanh(x). */
  private def g_logcosh(m : Mat) : Mat = {
    return tanh(m)
  }
  
  /** Assumes g'(x) = d/dx tanh(x). This is pretty complicated; see WolframAlpha for confirmation. */
  private def g_d_logcosh(m : Mat) : Mat = {
    val a = (2*cosh(m))/(cosh(2*m)+1)
    a ~ a *@ a
    return a
  }
  
  /** Assumes G(x) = -exp(-x^2/2), good if data is super-Gaussian or robustness is needed. */
  private def G_exponent(m : Mat) : Mat = {
    return -exp(-0.5f * (m *@ m))
  }
  
  /** Assumes g(x) = d/dx -exp(-x^2/2) = x*exp(-x^2/2). */
  private def g_exponent(m : Mat) : Mat = {
    return m *@ exp(-0.5f * (m *@ m))
  }
  
  /** Assumes g'(x) = d/dx x*exp(-x^2/2) = (1-x^2)*exp(-x^2/2). */
  private def g_d_exponent(m : Mat) : Mat = {
    return (1 - (m *@ m)) *@ exp(-0.5f * (m *@ m))
  }

  /** Assumes G(x) = x^4/4, a weak contrast function, but OK for sub-Gaussian data w/no outliers. */
  private def G_kurtosis(m: Mat) : Mat = {
    val c = m *@ m
    c ~ c *@ c 
    return c / 4.0f
  }
  
  /** Assumes g(x) = d/dx x^4/4 = x^3. */
  private def g_kurtosis(m : Mat) : Mat = {
    return m *@ m *@ m
  }
  
  /** Assumes g'(x) = d/dx x^3 = 3x^2. */
  private def g_d_kurtosis(m : Mat) : Mat = {
    return 3 * (m *@ m)
  }
  
  /** 
   * Takes in the model matrix and returns an orthogonal version of it, so WW^T = identity. We use a method 
   * from A. Hyvärinen and E. Oja (2000): an iterative algorithm that uses a norm that is NOT the Frobenius
   * norm, and then iterate a W = 1.5*W - 0.5*W*^W*W update until convergence (it's quadratic in convergence).
   * This involves no eigendecompositions and should be fast. We use the maximum absolute row sum norm, so we
   * take the absolute value of elements, sum over rows, and pick the largest of the values. The above assumes 
   * that the covariance matrix of the data is the identity, i.e., C = I. If not, plug in C.
   * 
   * @param w The model matrix that we want to transform to be orthogonal (often referred to as "mm" here).
   * @param dat The data matrix, used to compute the covariance matrices if necessary.
   */
  private def orthogonalize(w : Mat, dat : Mat) : Mat = {
    var C:Mat = null
    if (opts.preWhitened) {
      C = mkdiag(ones(dat.nrows,1))
    } else {
      C = getSampleCovariance(dat)
    }
    val WWT = w * C *^ w
    val result = w / sqrt(maxi(sum(abs(WWT), 2)))
    if (sum(sum(result)).dv.isNaN) {
      println("Error: sum(sum(result)) = NaN, indicating issues wiht sqrt(maxi(sum(abs(WWT),2))).")
    }
    var a = 0
    while (a < opts.numOrthogIter) { // Can result in NaNs, be careful.
      val newResult = ((1.5f * result) - 0.5f * (result * C *^ result * result))
      result <-- newResult
      if (sum(sum(result)).dv.isNaN) {
        println("Error: sum(sum(result)) = NaN, indicating that NaNs are appearing.")
      }
      a = a + 1
    }
    return result
  }

  /** Gets sample covariance matrix (one column of m is one sample). See Wikipedia for matrix formulation. */
  private def getSampleCovariance(m : Mat) : Mat = {
    val F = m - mean(m,2)
    return (F *^ F) / (m.ncols - 1)
  }
}


object ICA {

  trait Opts extends FactorModel.Opts {
    var G_function:String = "logcosh"
    var numOrthogIter:Int = 10
    var preWhitened:Boolean = false
  }

  class Options extends Opts {}
  
  /** ICA with a single matrix datasource. The dimension is based on the input matrix. */
  def learner(mat0:Mat) = {
    class xopts extends Learner.Options with MatDS.Opts with ICA.Opts with ADAGrad.Opts
    val opts = new xopts
    opts.dim = size(mat0)(0)
    opts.npasses = 10
    opts.batchSize = math.min(250000, mat0.ncols/15 + 1) // Just a heuristic
    opts.numOrthogIter = math.min(10, 5+math.sqrt(opts.dim).toInt)
    val nn = new Learner(
        new MatDS(Array(mat0:Mat), opts), 
        new ICA(opts), 
        null,
        new ADAGrad(opts), 
        opts)
    (nn, opts)
  }
  
  /** ICA with a files dataSource. */
  def learner(fnames:List[(Int)=>String], d:Int) = {
    class xopts extends Learner.Options with FilesDS.Opts with ICA.Opts with ADAGrad.Opts
    val opts = new xopts
    opts.dim = d
    opts.fnames = fnames
    opts.batchSize = 25000;
    implicit val threads = threadPool(4)
    val nn = new Learner(
        new FilesDS(opts), 
        new ICA(opts), 
        null,
        new ADAGrad(opts), 
        opts)
    (nn, opts)
  }

  /** Ranks the independent components by their contribution to the original data. */
  def rankComponents() = {
    println("rankComponents() not yet implemented.")
  }
   
}
