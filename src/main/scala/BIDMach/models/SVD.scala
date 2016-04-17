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
  var R:Mat = null;
  var Mean:Mat = null;
  var batchCount = 0;
  var batchStep = 0;
  var batchSize = 0;
  var meanCount = 0;
  var alpha:Mat = null;
  
  def init() = {
  	val nfeats = mats(0).nrows;
  	batchSize = mats(0).ncols;
  	if (refresh) {
  		Q = normrnd(0, 1, nfeats, opts.dim);                 // Randomly initialize Q
//  		QRdecompt(Q, Q, null);                               // Orthonormalize it
		Q ~ Q / sqrt(Q dot Q);
  		SV = Q.zeros(1, opts.dim);                           // Holder for Singular values
  		if (opts.subMean) Mean = Q.zeros(nfeats, 1)
  	} else {
  	  Q = modelmats(0);
  	  SV = modelmats(1);
  	  if (opts.subMean) Mean = modelmats(2);
  	}
  	Q = convertMat(Q);                                     // Move to GPU or double if needed
  	SV = convertMat(SV);
  	if (opts.subMean) {
  		Mean = convertMat(Mean);
  	  setmodelmats(Array(Q, SV, Mean));
  	} else {
  	  setmodelmats(Array(Q, SV));
  	}
  	Mean.clear;
  	P = Q.zeros(Q.nrows, Q.ncols);                         // Zero P
    R = Q.zeros(opts.dim, opts.dim);
    alpha = Q.zeros(1,1);
                      
  	updatemats = Array(P);
    batchCount = 0;
    batchStep = opts.batchesPerUpdate;
  }
  
  def dobatch(mats:Array[Mat], ipass:Int, pos:Long):Unit = {
    val M = mats(0);
    if (opts.subMean && ipass == 0) {
      meanCount += 1;
      alpha.set(1f/meanCount);
      val mn = mean(M, 2);
      Mean ~ Mean + alpha * (mn - Mean);      
    }
    val Qt = Q.t;                                           // Compute P = M * M^t * Q efficiently
    val QtM = Qt * M;
    if (opts.subMean) QtM ~ QtM - (Qt * Mean);
    val PPt = QtM *^ M;
    if (opts.subMean) PPt ~ PPt - (sum(QtM,2) *^ Mean);
    val PP = PPt.t                    
    if (ipass < opts.miniBatchPasses) {
      if (batchCount >= batchStep) {
        subspaceIter;                                        // Do minibatch subspace iterations 
        batchCount = 0;
        batchStep *= 2;
        P.clear;
      }
    }
    P ~ P + PP;
    batchCount += 1;
  }
  
  def evalbatch(mat:Array[Mat], ipass:Int, pos:Long):FMat = {
	  val M = mats(0);
	  if (ogmats != null) {
	  	val Qt = Q.t; 
	  	val QtM = Qt * M;
	  	if (opts.subMean) QtM ~ QtM - Qt * Mean;
	    ogmats(0) = QtM;                                     // Save right singular vectors
	    val PPt = QtM *^ M;
	    if (opts.subMean) PPt ~ PPt - QtM *^ Mean;
	    P <-- PPt.t
	  }
	  SV ~ (P ∙ Q) / P.nrows;                                            // Estimate the singular values
	  max(SV, 1e-6f, SV);
	  val diff = if (opts.evalType == 0) {
	    P - (SV ∘ Q);	                                       // residual
	  } else {
	  	(P / SV)  - Q;      
	  }
    row(-(math.sqrt(norm(diff) / (1L * diff.length * batchSize * batchCount))));           // return the norm of the residual
  } 
  
  override def updatePass(ipass:Int) = {
    if (ipass < opts.asInstanceOf[Learner.Options].npasses-1) {
    	if (ipass >= opts.miniBatchPasses) {
    		if (opts.doRayleighRitz && ipass % 2 == 1)
    			RayleighRitz;
    		else
    			subspaceIter;
    	}
    	P.clear;
    	batchCount = 0;
    	batchStep = opts.batchesPerUpdate;
    }
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
    var batchesPerUpdate = 10;
    var evalType = 0;
    var doRayleighRitz = true;
    var subMean = true;
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
    nopts.miniBatchPasses = 0;
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
  
  class FilePredOptions extends Learner.Options with SVD.Opts with FileSource.Opts with FileSink.Opts;
  
  // This function constructs a predictor from an existing model 
  def predictor(model:Model, infnames:String, outfnames:String):(Learner, FilePredOptions) = {
    val nopts = new FilePredOptions;
    nopts.dim = model.opts.dim;
    nopts.fnames = List(FileSource.simpleEnum(infnames, 1, 0));
    nopts.ofnames = List(FileSource.simpleEnum(outfnames, 1, 0));
    val newmod = new SVD(nopts);
    newmod.refresh = false
    model.copyTo(newmod);
    implicit val threads = threadPool(4);
    val nn = new Learner(
        new FileSource(nopts), 
        newmod, 
        null,
        null,
        new FileSink(nopts),
        nopts)
    (nn, nopts)
  }
 
} 



