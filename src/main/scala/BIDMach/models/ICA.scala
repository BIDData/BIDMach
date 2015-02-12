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
 * Independent Component Analysis, using FastICA with logcosh, exponents, or kurtosis G-fucntions.
 * 
 * Some links with other implementations:
 * https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/fastica_.py#L60
 * More here: http://research.ics.aalto.fi/ica/fastica/
 */
class ICA(override val opts:ICA.Opts = new ICA.Options) extends FactorModel(opts) {
  
  var currentWhiteningMatrix:Mat = null // We DO NOT want this but keep for now
    /* // For whitening...
    val sampleCov = getSampleCovariance(data)
    val eigDecomp = seig(FMat(sampleCov))
    val DnegSqrt = mkdiag(sqrt(1.0 / eigDecomp._1))
    val Q = eigDecomp._2
    currentWhiteningMatrix = Q * DnegSqrt *^ Q
    */

  var mm:Mat = null                     // Our W = A^{-1}.
  var dataMean:Mat = null
  var batchIteration = 0.0              // Used to keep track of the mean

  override def init() {
    super.init()
    if (refresh) {
      mm = modelmats(0)
      modelmats = new Array[Mat](2)
      modelmats(0) = mm
      modelmats(1) = mm.zeros(mm.nrows, 1)
      //modelmats(2) = mkdiag(mm.ones(mm.nrows, 1)) // Has to start out like this, right?
    }
    updatemats = new Array[Mat](2)
    updatemats(0) = mm.zeros(mm.nrows, mm.nrows)
    updatemats(1) = mm.zeros(mm.nrows,1) // Keep to avoid null pointer exceptions, but we don't use it
    //updatemats(2) = mm.zeros(mm.nrows, mm.nrows)
    println("Initial modelmats(0) =\n" + modelmats(0))
  }
    
  /** 
   * Store data in "user" for use in the next mupdate() call. This produces all (w_j^T)*(x^{i}) values,
   * which can speed up operations such as log likelihood computations that use those dot products. We
   * also scale the data to be zero-mean by subtracting away the estimated data mean, modelmats(1).
   * 
   * Whitening:
   *  (1) For given data block, estimate Exp[X *^ T], the covariance matrix.
   *  (2) Find ED^{-1/2}E^T from the eigendecomposition of the estimated sample covariance matrix
   *  (3) Store that as our current whitening matrix and left-multiply the current data
   *  
   * Note: new code to compute the mean here on the first pass, rather than have it converge to something.
   * It does assume that all the data comes in equal-sized batches, with the exception of the last one.
   */
  def uupdate(data : Mat, user : Mat, ipass : Int) {
    //println("\n\nINSIDE UUPDATE()")
    //println("Line 2: modelmats(0) =\n" + modelmats(0))
    //println("Line 3: mm =\n" + mm)
    //println("Line 4: modelmats(2) =\n" + modelmats(2))
    if (ipass == 0) {
      batchIteration = batchIteration + 1.0
      modelmats(1) = (modelmats(1)*(batchIteration-1) + mean(data,2)) / batchIteration 
      println("Inside ipass = 0, batchIteration = " + batchIteration + " and modelmats(1).t = " + modelmats(1).t)
    }
    data ~ data - modelmats(1)
    // Well, we DO want this part so keep that here. AFTER this, we "assume" the data is white.
    // data <-- modelmats(2) * data // This is why we need modelmats(2) to start out as a NON-zero matrix.
    user ~ mm * data // w^Tx, with whitened x (supposedly)
  }
  
  /**
   * Update A^{-1} = W = modelmat(0) = mm, varies based on the ICA algorithm. Here, "data" is zero-mean
   * for this particular block.
   */
  def mupdate(data : Mat, user : Mat, ipass : Int) {
    val m = data.ncols
    val n = mm.ncols
    val B = mm * user
    if (useGPU) {
      mm <-- GMat(orthogonalize(mm, data))
    } else {
      mm <-- orthogonalize(mm, data)
    }
    val (gwtx,g_wtx) = opts.G_function match {
      case "logcosh" => (g_logcosh(user), g_d_logcosh(user))
      case "exponent" => (g_exponent(user), g_d_exponent(user))
      case "kurtosis" => (g_kurtosis(user), g_d_kurtosis(user))
    }
    val termBeta = mkdiag( -mean(user *@ gwtx, 2))
    val termAlpha = mkdiag( -1.0 / (getdiag(termBeta) - (mean(g_wtx,2))) )
    val termExpec = (gwtx *^ user) / m
    updatemats(0) <-- termAlpha * (termBeta + termExpec) * mm
  }
    
  /**
   * Computes objective function to minimize or maximize (I think both ways work). Just as in mupdate(),
   * summations and other operations are done using matrix operations so it can be tricky.
   */
  def evalfun(data : Mat, user : Mat) : FMat = {
    val m = data.ncols
    val n = mm.ncols
    val (big_gwtx, stdNorm) = opts.G_function match {
      case "logcosh" => (G_logcosh(user), FMat(0.375)) // 0.375 obtained from Numpy sampling
      case "exponent" => (G_exponent(user), FMat(-1.0 / sqrt(2.0)))
      case "kurtosis" => (G_kurtosis(user), FMat(0.75))
    }
    val expUser = FMat(mean(mean(big_gwtx)))
    return (expUser - stdNorm) * (expUser - stdNorm)
  }
  
  // We have nine total functions for different choices of G(x), g(x), and g'(x), where g(x) = G'(x).
  
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
  
  // Other helper functions

  /** For "orthogonalizing" W using the (WW^T)^{-1/2} * W technique; uses eigendecomposition of WW^T. */
  private def orthogonalize(matrix : Mat, dat : Mat) : FMat = {
    val fmm = FMat(matrix) // For FMats only, not GMats.
    val fdata = FMat(dat)
    //val eigDecomp = seig(fmm * FMat(getSampleCovariance(fdata)) *^ fmm)
    val eigDecomp = seig(fmm *^ fmm)
    val DnegSqrt = mkdiag(sqrt(1.0 / eigDecomp._1))
    val Q = eigDecomp._2
    val result = Q * DnegSqrt *^ Q * fmm
    return result
  }

  /** Gets sample covariance matrix (one column of m is one sample). See Wikipedia for matrix formulation. */
  private def getSampleCovariance(m : Mat) : Mat = {
    val meanVec = mean(m, 2)
    val onesRow = ones(1, m.ncols)
    val F = m - (meanVec * onesRow)
    return (F *^ F) / (m.ncols - 1)
  }
  
  /** Gets the log of the absolute value of the determinant of the input, using QR decomposition. */
  private def logAbsValDeterminantQR(matrix : Mat) : FMat = {
    val (q,r) = matrix match {
      case matrix : GMat => QRdecomp(FMat(matrix))
      case _ => QRdecomp(matrix)
    }
    val x = sum(ln(abs(getdiag(r))))
    return x.dv
  }
}


object ICA {

  /**
   * G_function: possible values are "logcosh", "exponent", "kurtosis"
   * dataAlreadyWhtie: true if data has been pre-whitened (ideal), if not, need to do expensive whitening.
   */
  trait Opts extends FactorModel.Opts {
    val G_function:String = "logcosh"
    val dataAlreadyWhte:Boolean = true
  }

  class Options extends Opts {}
}