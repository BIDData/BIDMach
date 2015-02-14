package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMach._
import java.lang.ref._
import jcuda.NativePointerObject
import java.lang.Math;

/**
 * Independent Component Analysis, using FastICA. It has the ability to center and whiten data. See below
 * for features and possible limitations.
 * 
 * Our algorithm is based on the method presented in:
 * 
 * A. Hyvärinen and E. Oja. Independent Component Analysis: Algorithms and Applications. 
 * Neural Networks, 13(4-5):411-430, 2000.
 * 
 * In particular, we provide the logcosh, exponential, and kurtosis "G" functions.
 * 
 * This algorithm computes the following modelmats array:
 * 
 *   > modelmats(0) stores the inverse of the mixing matrix. If X = A*S represents the data, then it's the 
 *     estimated A^{-1}, which we assume is square and invertible for now.
 *   > modelmats(1) stores the mean vector of the data, which is computed entirely on the first pass. This
 *     means once we estimate A^{-1} in modelmats(0), we need to first shift the data by this amount, and
 *     then multiply to recover the (centered) sources. Example:
 * {{{
 * modelmats(0) * (data - modelmats(1))
 * }}}
 *     Here, data is an n x N matrix, whereas modelmats(1) is an n x 1 matrix. For efficiency reasons, we
 *     assume a constant batch size for each block of data so we take the mean across all batches. This is 
 *     true except for (usually) the last batch, but this often isn't enough to make a difference.
 *     
 * Thus, modelmats(1) helps to center the data. The whitening in this algorithm happens during the updates
 * to W in both the orthogonalization and the fixed point steps. The former uses the computed convariance 
 * matrix and the latter relies on an approximation of W^T*W to the inverse covariance matrix. It is fine
 * if the data is already pre-whitened before being passed to BIDMach.
 *     
 * Currently, we are working on the following extensions:
 * 
 *   > Allowing ICA to handle non-square mixing matrices. Most research about ICA assumes that A is n x n.
 *   > Improving the way we handle the computation of the mean, so it doesn't rely on the last batch being
 *     of similar size to all prior batches. Again, this is minor, especially for large data sets.
 * 
 * For additional references, see Aapo Hyvärinen's other papers, and visit:
 *   http://research.ics.aalto.fi/ica/fastica/
 */
class ICA(override val opts:ICA.Opts = new ICA.Options) extends FactorModel(opts) {
  
  var mm:Mat = null         // Our W = A^{-1}.
  var batchIteration = 0.0  // Used to keep track of the mean

  override def init() {
    super.init()
    if (refresh) {
      mm = modelmats(0)
      setmodelmats(Array(mm, mm.zeros(mm.nrows,1)))
    }
    updatemats = new Array[Mat](2)
    updatemats(0) = mm.zeros(mm.nrows, mm.nrows)
    updatemats(1) = mm.zeros(mm.nrows,1) // Keep to avoid null pointer exceptions, but we don't use it
  }
    
  /** 
   * Store data in "user" for use in the next mupdate() call, and updates the moving average if necessary.
   * Also "orthogonalizes" the model matrix.
   * 
   * First, it checks if this is the first pass over the data, and if so, updates the moving average assuming
   * that the number of data samples in each block is the same for all blocks. After the first pass, the data 
   * mean vector is fixed in modelmats(1). Then the data gets centered via: "data ~ data - modelmats(1)".
   * 
   * We also use "user ~ mm * data" to store all (w_j^T) * (x^{i}) values, where w_j^T is the j^th row of our
   * estimated W = A^{-1}, and x^{i} is the i^{th} sample in this block of data. These values are later used
   * as part of fixed point updates.
   * 
   * @param data An n x batchSize matrix, where each column corresponds to a data sample.
   * @param user An intermediate matrix that stores (w_j^T) * (x^{i}) values.
   * @param ipass The current pass through the data.
   */
  def uupdate(data : Mat, user : Mat, ipass : Int) {
    if (ipass == 0) {
      batchIteration = batchIteration + 1.0
      modelmats(1) = (modelmats(1)*(batchIteration-1) + mean(data,2)) / batchIteration 
    }
    data ~ data - modelmats(1)
    if (useGPU) { // After each update in mupdate(), mm must be orthogonalized.
      mm <-- GMat(orthogonalize(mm, data))
    } else {
      mm <-- orthogonalize(mm, data)
    }
    user ~ mm * data
  }
  
  /**
   * This performs the matrix fixed point update to the estimated W = A^{-1}:
   * 
   *   W^+ = W + diag(alpha_i) * [ diag(beta_i) - Expec[g(Wx)*(Wx)^T] ] * W,
   * 
   * where g = G', beta_i = -Expec[(Wx)_ig(Wx)_i], and alpha_i = -1/(beta_i - Expec[g'(Wx)_i]). We need to
   * be careful to take expectations of the appropriate items. The gwtx and g_wtx terms are matrices with
   * useful intermediate values that represent the full data matrix X rather than a single column/element x.
   * The above update for W^+ goes in updatemats(0), except the additive W since that should be taken care
   * of by the ADAGrad updater.
   * 
   * I don't THINK anything here changes if the data is not white, since one of Hyvärinen's papers implied
   * that the update here includes an approximation to the inverse covariance matrix.
   * 
   * @param data An n x batchSize matrix, where each column corresponds to a data sample.
   * @param user An intermediate matrix that stores (w_j^T) * (x^{i}) values.
   * @param ipass The current pass through the data.
   */
  def mupdate(data : Mat, user : Mat, ipass : Int) {
    val m = data.ncols
    val n = mm.ncols
    val B = mm * user
    val (gwtx,g_wtx) = opts.G_function match {
      case "logcosh" => (g_logcosh(user), g_d_logcosh(user))
      case "exponent" => (g_exponent(user), g_d_exponent(user))
      case "kurtosis" => (g_kurtosis(user), g_d_kurtosis(user))
      case _ => throw new RuntimeException("opts.G_function is not a valid value: " + opts.G_function)
    }
    val termBeta = mkdiag( -mean(user *@ gwtx, 2) )
    val termAlpha = mkdiag( -1.0 / (getdiag(termBeta) - (mean(g_wtx,2))) )
    val termExpec = (gwtx *^ user) / m
    updatemats(0) <-- termAlpha * (termBeta + termExpec) * mm
  }
    
  /**
   * Currently, this computes the approximation of negentropy, which is the objective function to maximize.
   * 
   * To understand this, let w be a single row vector of W, let x be a single data vector, and let v be a
   * standard normal random variable. To find this one independent component, we maximize
   * 
   *   J(w^Tx) \approx ( Expec[G(w^Tx)] - Expec[G(v)] )^2,
   * 
   * where G is the function set at opts.G_function. So long as the W matrix (capital "W") is orthogonal, 
   * which we do enforce, then w^Tx satisfies the requirement that the variance be one. To extend this to
   * the whole matrix W, take the sum over all the rows, so the problem is: maxmize{ \sum_w J(w^Tx) }.
   * 
   * On the other hand, the batchSize should be much greater than one, so "data" consists of many columns.
   * Denoting the data matrix as X, we can obtain the expectations by taking the sample means. In other words,
   * we take the previous "user" matrix, W*X, apply the function G to the data, and THEN take the mean across
   * rows, so mean(G(W*X),2). The mean across rows gives what we want since it's applying the same row of W to
   * different x (column) vectors in our data.
   * 
   * @param data An n x batchSize matrix, where each column corresponds to a data sample.
   * @param user An intermediate matrix that stores (w_j^T) * (x^{i}) values.
   * @param ipass The current pass through the data.
   */
  def evalfun(data : Mat, user : Mat, ipass : Int) : FMat = {
    val (big_gwtx, stdNorm) = opts.G_function match {
      case "logcosh" => (G_logcosh(user), FMat(0.375)) // 0.375 obtained from Numpy sampling
      case "exponent" => (G_exponent(user), FMat(-1.0 / sqrt(2.0)))
      case "kurtosis" => (G_kurtosis(user), FMat(0.75))
      case _ => throw new RuntimeException("opts.G_function is not a valid value: " + opts.G_function)
    }
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
    return ( (2*cosh(m))/(cosh(2*m)+1) ) *@ ( (2*cosh(m))/(cosh(2*m)+1) )
  }
  
  /** Assumes G(x) = -exp(-x^2/2), good if data is super-Gaussian or robustness is needed. */
  private def G_exponent(m : Mat) : Mat = {
    return -exp(-0.5 * (m *@ m))
  }
  
  /** Assumes g(x) = d/dx -exp(-x^2/2) = x*exp(-x^2/2). */
  private def g_exponent(m : Mat) : Mat = {
    return m *@ exp(-0.5 * (m *@ m))
  }
  
  /** Assumes g'(x) = d/dx x*exp(-x^2/2) = (1-x^2)*exp(-x^2/2). */
  private def g_d_exponent(m : Mat) : Mat = {
    return (1 - (m *@ m)) *@ exp(-0.5 * (m *@ m))
  }

  /** Assumes G(x) = x^4/4, a weak contrast function, but OK for sub-Gaussian data w/no outliers. */
  private def G_kurtosis(m: Mat) : Mat = {
    return (m *@ m *@ m *@ m) / 4.0
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
   * Takes in the model matrix and returns an orthogonal version of it, so WW^T = identity.
   * 
   * We use one of two methods depending on opts.orthogMethod. Both are from A. Hyvärinen and E. Oja (2000):
   * 
   *   (1) The preferred way is to use an iterative algorithm that uses a norm that is NOT the Frobenius norm,
   *     and then iterate a W = 1.5*W - 0.5*W*^W*W update until convergence (it's quadratic in convergence).
   *     This involves no eigendecompositions and should be fast. We use the maximum absolute row sum norm, so
   *     we take the absolute value of elements, sum over the rows, and pick the largest of those values.
   *   (2) A second way that might be slower is to use W = (W*^W)^{-1/2}*W, which requires the eigendecomposition
   *     of the W*^W (this is the same as W*W^T) matrix.
   *     
   * The above assume that the covariance matrix of the data is the identity, i.e., C = I. If not, plug in C.
   * 
   * @param w The model matrix that we want to transform to be orthogonal (often referred to as "mm" here).
   * @param dat The data matrix, used to compute the covariance matrices if necessary.
   */
  private def orthogonalize(w : Mat, dat : Mat) : FMat = {
    if (opts.orthogMethod != "iterative" && opts.orthogMethod != "symmetric") {
      throw new RuntimeException("opts.orthogMethod is not a valid value: " + opts.orthogMethod)
    }
    if (opts.orthogMethod == "iterative") {
      val C = FMat(getSampleCovariance(FMat(dat)))
      val WWT = w * C *^ w
      var result = FMat(w / sqrt(maxi(sum(abs(WWT), 2)))) // Maximum ABSOLUTE, ROW, SUM, norm
      var a = 0
      while (a < opts.dim) { // Quadratic in convergence, so perhaps opts.dim is good?
        val newMatrix = FMat((1.5 * result) - 0.5 * (result * C *^ result * result))
        result = newMatrix
        a = a + 1
      }
      return result
    }
    else {
      val fmm = FMat(w)
      val eigDecomp = seig(fmm * FMat(getSampleCovariance(FMat(dat))) *^ fmm)
      val DnegSqrt = mkdiag(sqrt(1.0 / eigDecomp._1))
      val Q = eigDecomp._2
      return Q * DnegSqrt *^ Q * fmm 
    }
  }

  /** Gets sample covariance matrix (one column of m is one sample). See Wikipedia for matrix formulation. */
  private def getSampleCovariance(m : Mat) : Mat = {
    val meanVec = mean(m, 2)
    val onesRow = ones(1, m.ncols)
    val F = m - (meanVec * onesRow)
    return (F *^ F) / (m.ncols - 1)
  }
}


object ICA {

  /**
   * Provides some settings for the ICA model.
   * 
   * > G_function: possible values are "logcosh", "exponent", "kurtosis". Using "logcosh" is recommended.
   * > orthogMethod: possible values are "iterative", which corresponds to the W <- 1.5*W + .5*(W*^W*W)
   *   case (recommended), or "symmetric" which is W <- (W*^W)^{-1/2}*W, which uses eigendecompositions.
   */
  trait Opts extends FactorModel.Opts {
    val G_function:String = "logcosh"
    val orthogMethod:String = "iterative"
  }

  class Options extends Opts {}
}