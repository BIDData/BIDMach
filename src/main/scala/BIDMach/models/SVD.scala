package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMach.datasources._
import BIDMach.datasinks._
import BIDMach.updaters._
import BIDMach._

/**
 * A scalable approximate SVD (Singular Value Decomposition) using subspace iteration
 * 
 * '''Parameters'''
 - dim(256): Model dimension
 *
 * Other key parameters inherited from the learner, datasource and updater:
 - blockSize: the number of samples processed in a block
 - npasses(10): number of complete passes over the dataset
 *
 */

class SVD(opts:SVD.Opts = new SVD.Options) extends Model(opts) {
 
  var Q:Mat = null;                                        // (Left) Singular vectors
  var SV:Mat = null;                                       // Singular values
  var P:Mat = null;
  var R:Mat = null
  
  def init() = {
  	val nfeats = mats(0).nrows;  	
  	Q = normrnd(0, 1, nfeats, opts.dim);                   // Randomly initialize Q
  	QRdecompt(Q, Q, null);                                 // Orthonormalize it
  	Q = convertMat(Q);                                     // Move to GPU or double if needed
  	P = Q.zeros(Q.nrows, Q.ncols);                         // Zero P
  	SV = Q.zeros(1, opts.dim);                             // Holder for Singular values
    R = Q.zeros(opts.dim, opts.dim)
  	setmodelmats(Array(Q, SV));                      
  	updatemats = Array(P);
  }
  
  def dobatch(mats:Array[Mat], ipass:Int, pos:Long):Unit = {
    val M = mats(0);
    val PP = (Q.t * M *^ M).t                              // Compute P = M * M^t * Q efficiently
    if (ipass < opts.miniBatchPasses) {
      P = PP;
      subspaceIter; 
    } else {
      P ~ P + PP;
    }
  }
  
  def evalbatch(mat:Array[Mat], ipass:Int, pos:Long):FMat = {
	  val M = mats(0);
    SV ~ P ∙ Q;                                            // Estimate the singular values
    max(SV, 1e-6f, SV)
    if (ogmats != null) ogmats(0) = Q.t * M;               // Save right singular vectors
    val diff = (P / SV)  - Q;                              // residual
    row(-(math.sqrt(norm(diff) / diff.length)));           // return the norm of the residual
  }
  
  override def updatePass(ipass:Int) = {
    if (ipass >= opts.miniBatchPasses) {
      if (ipass % 2 == 1)
      	RayleighRitz;
      else
        subspaceIter;
    }
    P.clear;
  }


  def RayleighRitz = {
    R ~ P ^* Q;
    val (evals, evecs) = feig(cpu(R));
    R <-- evecs(?, irow((R.ncols-1) to 0 by -1));
    Q <-- Q * R;
    P <-- P * R;
  }
  
  def subspaceIter = {
	  QRdecompt(P, Q, null);
  }
}

object SVD  {
  trait Opts extends Model.Opts {
    var miniBatchPasses = 1;
  }
  
  class Options extends Opts {}
      
  class MatOptions extends Learner.Options with SVD.Opts with MatSource.Opts with Batch.Opts
  
  def learner(mat:Mat):(Learner, MatOptions) = { 
    val opts = new MatOptions;
    opts.batchSize = math.min(100000, mat.ncols/30 + 1);
    opts.updateAll = true;
  	val nn = new Learner(
  	    new MatSource(Array(mat), opts), 
  			new SVD(opts), 
  			null,
  			new Batch(opts), 
  			null,
  			opts)
    (nn, opts)
  }
  
  class FileOptions extends Learner.Options with SVD.Opts with FileSource.Opts with Batch.Opts
  
  def learner(fnames:String):(Learner, FileOptions) = { 
    val opts = new FileOptions;
    opts.batchSize = 10000;
    opts.fnames = List(FileSource.simpleEnum(fnames, 1, 0));
    opts.updateAll = true;
    implicit val threads = threadPool(4);
  	val nn = new Learner(
  	    new FileSource(opts), 
  			new SVD(opts), 
  			null,
  			new Batch(opts), 
  			null,
  			opts)
    (nn, opts)
  }
  
  class PredOptions extends Learner.Options with SVD.Opts with MatSource.Opts with MatSink.Opts;
  
  // This function constructs a predictor from an existing model 
  def predictor(model:Model, mat1:Mat):(Learner, PredOptions) = {
    val nopts = new PredOptions;
    nopts.batchSize = math.min(10000, mat1.ncols/30 + 1)
    nopts.dim = model.opts.dim;
    val newmod = new SVD(nopts);
    newmod.refresh = false
    model.copyTo(newmod)
    val nn = new Learner(
        new MatSource(Array(mat1), nopts), 
        newmod, 
        null,
        null,
        new MatSink(nopts),
        nopts)
    (nn, nopts)
  }
 
} 



