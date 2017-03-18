package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GDMat,GIMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMach.datasources._
import BIDMach.datasinks._
import BIDMach.updaters._
import BIDMach.Learner

/**
 * Sparse Matrix Factorization with L2 loss (similar to ALS). 
 * 
 * '''Parameters'''
 - dim(256): Model dimension
 - uiter(5): Number of iterations on one block of data
 - miter(5): Number of CG iterations for model updates - not currently used in the SGD implementation.
 - lambdau(5f): Prior on the user (data) factor
 - lambdam(5f): Prior on model 
 - regumean(0f): prior on instance mean
 - regmmean(0f): Prior on feature mean
 - startup(1): Skip CG for this many iterations
 - traceConvergence(false): Print out trace info for convergence of the u iterations.
 - doUser(false): Apply the per-instance mean estimate. 
 - weightByUser(false): Weight loss equally by users, rather than their number of choices. 
 - ueps(1e-10f): A safety floor constant
 - uconvg(1e-3f): Stop u iteration if error smaller than this. 
 *
 * Other key parameters inherited from the learner, datasource and updater:
 - batchSize: the number of samples processed in a block
 - npasses(2): number of complete passes over the dataset
 - useGPU(true): Use GPU acceleration if available.
 *
 * '''Example:'''
 * 
 * a is a sparse word x document matrix
 * {{{
 * val (nn, opts) = SFA.learner(a)
 * opts.what             // prints the available options
 * opts.uiter=2          // customize options
 * nn.train              // train the model
 * nn.modelmat           // get the final model
 * nn.datamat            // get the other factor (requires opts.putBack=1)
 * }}}
 */

class SMF(override val opts:SMF.Opts = new SMF.Options) extends FactorModel(opts) {

  var mm:Mat = null;
  var traceMem = false;
  var mzero:Mat = null;
  var slm:Mat = null; 
  var mlm:Mat = null; 
  var iavg:Mat = null;
  var avg:Mat = null;
  var lamu:Mat = null;
  var itemsum:Mat = null;
  var itemcount:Mat = null;
  var nfeats:Int = 0;
  var nratings:Double = 0;
  // For integrated ADAGrad updater
  var vexp:Mat = null;
  var texp:Mat = null;
  var pexp:Mat = null;
  var cscale:Mat = null;
  var lrate:Mat = null;
  var uscale:Mat = null;
  var sumsq:Mat = null;
  var firststep = -1f;
  var waitsteps = 0;
  var epsilon = 0f;
  var aopts:ADAGrad.Opts = null;

  // Daniel: doing this to set MB sizes in MHTest code.
  var numNonzerosMB:Int = -1
  
  override def init() = {
    // Get dimensions; for Netflix, size(mats(0)) = (17770,batchSize).
    mats = datasource.next;
    datasource.reset;
    nfeats = mats(0).nrows;
    val batchSize = mats(0).ncols;
    numNonzerosMB = mats(0).nnz
    val d = opts.dim;

    if (refresh) {
      // Randomly drawing mm, iavg, and avg (the three respective model
      // matrices). Note that nfeats is the number of items (e.g. movies).
      println("Inside refresh")
      mm = normrnd(0,0.01f,d,nfeats);
      mm = convertMat(mm);
      avg = mm.zeros(1,1)
      iavg = mm.zeros(nfeats,1);
      itemsum = mm.zeros(nfeats, 1);
      itemcount = mm.zeros(nfeats, 1);
      setmodelmats(Array(mm, iavg, avg));
    } 

    // Handle brief logic with GPUs. Careful with aliasing as well!!
    useGPU = opts.useGPU && Mat.hasCUDA > 0;    
    if (useGPU || useDouble) {
      gmats = new Array[Mat](mats.length);
    } else {
      gmats = mats;
    }  
    modelmats(0) = convertMat(modelmats(0));
    modelmats(1) = convertMat(modelmats(1));
    modelmats(2) = convertMat(modelmats(2));
    mm = modelmats(0);
    iavg = modelmats(1);
    avg = modelmats(2);

    // Here's some confusing stuff. Seems to be "small" stuff about constants.
    // uscale, an internal ADAGrad parameter but we use it (!!!).
    // cscale, an internal ADAGrad parameter but we ignore it.
    lamu = mm.ones(d, 1) ∘ opts.lambdau 
    if (opts.doUsers) lamu(0) = opts.regumean;
    slm = mm.ones(1,1) ∘ (opts.lambdam * batchSize);
    mlm = mm.ones(1,1) ∘ (opts.regmmean * batchSize);
    mzero = mm.zeros(1,1);
    uscale = mm.zeros(1,1);
    cscale = mm.ones(d, 1);
    cscale(0,0) = 0.0001f;
    if (opts.doUsers) mm(0,?) = 1f

    // The updatemats is the same length as the model matrices.
    updatemats = new Array[Mat](3);
    updatemats(2) = mm.zeros(1,1);

    // Set this to null to avoid the internal ADAGrad updater making updates.
    if (opts.aopts != null) initADAGrad(d, nfeats);
    vexp = convertMat(row(0.5f)); // External ADAGrad parameter, OK here.
  }

 
  /** An internal ADAGrad updater. Ignore this for our current experiments. */
  def initADAGrad(d:Int, m:Int) = {
  	aopts = opts.asInstanceOf[ADAGrad.Opts]
    firststep = -1f;
    lrate = convertMat(aopts.lrate);
    texp = if (aopts.texp.asInstanceOf[AnyRef] != null) convertMat(aopts.texp) else null;
    pexp = if (aopts.pexp.asInstanceOf[AnyRef] != null) convertMat(aopts.pexp) else null;
    vexp = convertMat(aopts.vexp);
    sumsq = convertMat(zeros(d, m));
    sumsq.set(aopts.initsumsq);
    waitsteps = aopts.waitsteps;
    epsilon = aopts.epsilon;
  }


  /**
   * Performs some number of passes over the minibatch to update the user
   * matrix. Try to understand how the user matrix gets updated ... note that
   * putBack = -1 by default. I think this is the user matrix update, so we're
   * holding the item matrix fixed (it's actually the model matrix, but the same
   * point holds) while updating the user by stochastic gradient descent.
   * 
   * We subtract biases, so predictions can be done with DDS(mm,user,sdata)
   * *without* re-adding biases. Also, we *do* clear out user here. is this
   * because John said we can't really save the entire *full* user matrix (the
   * one with size (dim,480189))?  ucounts sums up the number of nonzeros in
   * each columns of sdata0, then uci is something else on it. b might make
   * sense in some way, because the derivative term later is mm*(sdata-preds)
   * and the sdata-preds is supposed to be close to each other.
   *
   * We then update the user matrix several times based on current predictions.
   * Actually, this update makes sense because the normal SGD update for x_u
   * (user vectors) is x_u minus the following term:
   *
   *     alpha*(data-prediction)*item_vector + lambda*user_vector
   *
   * and that's what we have here! QUESTION, though, uscale is an integrated
   * ADAGrad value. Do we want it here? I'm also not sure why we need du to
   * have uscale and uci there ...
   * 
   * NOTE: Upon further inspection, it seems that `user` starts out as a matrix
   * of all zeros. So the user.clear with putBack<0 is un-necessary as it is
   * already cleared. I suppose in theory we should have some putBack mechanism
   * (that way, the user matrix value is stored from prior iterations) but John
   * said there's little reason to do that. Also, even with putBack=1, I can't
   * get the user matrix's values carried over. Hmmm ...
   *
   * @param sdata0 Training data minibatch of size (nitems, batchSize).
   * @param user Second matrix for computing predictions, of size (dim, batchSize).
   */ 
  def uupdate(sdata0:Mat, user:Mat, ipass:Int, pos:Long):Unit =  { 
    if (firststep <= 0) firststep = pos.toFloat;
    val step = (pos + firststep)/firststep;
    val texp = if (opts.asInstanceOf[Grad.Opts].texp.asInstanceOf[AnyRef] != null) {
      opts.asInstanceOf[Grad.Opts].texp.dv
    } else {
      opts.asInstanceOf[Grad.Opts].pexp.dv
    }
    uscale.set(opts.urate * math.pow(ipass+1, - texp).toFloat) 

    val sdata = sdata0 - (iavg + avg);
    if (putBack < 0) {
      user.clear
    }
    val b = mm * sdata;
    val ucounts = sum(sdata0 != 0f);
    val uci = (ucounts + 1f) ^ (- vexp);

    for (i <- 0 until opts.uiter) {
      val preds = DDS(mm, user, sdata);
      val deriv = b - mm * preds - (user ∘ lamu);
      val du = (deriv ∘ uscale ∘ uci);
      user ~ user + du;
      if (opts.traceConverge) {
        println("step %d, loss %f" format (i, ((norm(sdata.contents - preds.contents) ^ 2f) + (sum(user dot (user ∘ lamu)))).dv/sdata.nnz));
      }
    }
  }


  /**
   * Computes updates to the updatemats. Note again that we subtract (iavg+avg)
   * from the sdata, so that predictions are done with DDS(mm,user,sdata), and
   * the differences (for gradient update later) are stored. This is for
   * updating the item matrix, so we hold the user matrix fixed (it's updated in
   * uupdate) and compute updates for the item matrix (and the bias terms,
   * actually). Also, this might be why we don't use a user bias term. Note that
   * we call predictions once here, since mm and user are fixed for this method;
   * ideally, some other updater will use the updatemats we compute (i.e. the
   * gradients) to update the item matrix. The item matrix, if it wasn't clear,
   * is modelmats(0).
   * 
   * Note that there's some extra work with ipass < 1, I think to get reasonable
   * initialization values for our bias terms. Here, avg is the average rating
   * across the entire nonzeros of sdata0 (hence, our global bias), and iavg is
   * some (scaled) estimate of how much we should boost each individal item. The
   * iavg should be smaller since the avg already scales stuff to roughly 3.6.
   * 
   * During predictions, this method is NOT called, hence the biases aren't
   * updated.
   *
   * @param sdata0 Training data minibatch of size (nitems, batchSize).
   * @param user Second matrix for computing predictions, of size (dim,
   * 		batchSize). The matrix has the same values as the user matrix updated
   * 		from the most recent uupdate method call.
   */ 
  def mupdate(sdata0:Mat, user:Mat, ipass:Int, pos:Long):Unit = {
    val sdata = sdata0 - (iavg + avg);
    // values to be accumulated
    val preds = DDS(mm, user, sdata);
    val diffs = sdata + 2f; // I THINK 2f is only for avoiding aliasing, but why not 0f?
    diffs.contents ~ sdata.contents - preds.contents;

    if (ipass < 1) {
      itemsum ~ itemsum + sum(sdata0, 2); // sum horizontally
      itemcount ~ itemcount + sum(sdata0 != 0f, 2); // count #nonzeros horizontally 
      avg ~ sum(itemsum) / sum(itemcount);
      iavg ~ ((itemsum + avg) / (itemcount + 1)) - avg;
    } 
    
    // Compute gradient updates for the biases, and set wuser=user unless we're weighing.
    val icomp = sdata0 != 0f
    val icount = sum(sdata0 != 0f, 2);
    updatemats(1) = (sum(diffs,2) - iavg*mlm) / (icount + 1f);   // per-item term estimator
    updatemats(2) ~ sum(diffs.contents) / (diffs.contents.length + 1f);
    val wuser = if (opts.weightByUser) {
      val iwt = 100f / max(sum(sdata != 0f), 100f); 
      user ∘ iwt;
    } else {
      user;
    }
    if (firststep <= 0) firststep = pos.toFloat;

    // I get it! This derivative is virtually the same as what we had with the
    // user update, except user and mm swap locations, which is expected.
    if (opts.lsgd >= 0 || opts.aopts == null) {
      updatemats(0) = (wuser *^ diffs - (mm ∘ slm)) / ((icount + 1).t ^ vexp);   // simple derivative
    } else {
      if (texp.asInstanceOf[AnyRef] != null) {
        val step = (pos + firststep)/firststep;
        ADAGrad.multUpdate(wuser, diffs, modelmats(0), sumsq, null, lrate, texp, vexp, epsilon, step, waitsteps);
      } else {
        ADAGrad.multUpdate(wuser, diffs, modelmats(0), sumsq, null, lrate, pexp, vexp, epsilon, ipass + 1, waitsteps);
      }
    }
    if (opts.doUsers) mm(0,?) = 1f;
  }
  

  /** 
   * The evalfun normally called during training. Returns -RMSE on training data
   * minibatch (sdata0). It has an extra option to return a matrix of scores,
   * which will be useful for minibatch MH test updaters. We need a 1/(2*sigma^2) 
   * if we're assuming a Gaussian error distribution. 
   *
   * @param sdata Training data minibatch of size (nitems, batchSize).
   * @param user Second matrix for computing predictions, of size (dim,
   * 		batchSize). The values here are based on the values computed in the most
   * 		recent uupdate call.
   */
  def evalfun(sdata:Mat, user:Mat, ipass:Int, pos:Long):FMat = {
    val preds = DDS(mm, user, sdata) + (iavg + avg);
    if (ogmats != null) {
      ogmats(0) = user;
      if (ogmats.length > 1) {
        ogmats(1) = preds;
      }
    }
    val dc = sdata.contents
    val pc = preds.contents
    val diff = DMat(dc - pc);
    if (opts.matrixOfScores) {
      // TODO Temporary but should be OK for now (b/c we almost never increment MB).
      // The FMat(diff *@ diff) will make a vector, hence the broadcasting.
      // Also, I was getting a handful of *really* negative scores where log
      // p(.) went to -10000 or so. To prevent that, set a threshold at -15.
      //println(sqrt((diff ddot diff)/diff.length)) // Use for debugging and sanity checks.
      val sigma_sq = variance(diff).dv
      val scores = -ln(sqrt(2*math.Pi*sigma_sq)).v - (1.0f/(2*sigma_sq)).v * FMat(diff *@ diff)
      max(scores, -15f, scores)
      scores
    } else {
      val vv = diff ddot diff;
      -sqrt(row(vv/sdata.nnz))
    }
  }


  /** 
   * The evalfun normally called during TESTING (i.e. PREDICTION). Returns -RMSE
   * on the TRAINING minibatch in `sdata`. We should also store predictions in
   * `ogmats(1)`, which is what we can access externally via `preds(1)`. Thus,
   * it's predicting based on both training and testing.
   * 
   * @param sdata Training data minibatch of size (nitems, batchSize).
   * @param user Second matrix for computing predictions, size (dim, batchSize).
   * @param preds Matrix indicating the non-zero TESTING data points.
   */
  override def evalfun(sdata:Mat, user:Mat, preds:Mat, ipass:Int, pos:Long):FMat = {
    val spreds = DDS(mm, user, sdata) + (iavg + avg);
    val xpreds = DDS(mm, user, preds) + (iavg + avg);
    val dc = sdata.contents;
    val pc = spreds.contents;
    val vv = (dc - pc) ddot (dc - pc);

    println("mean values (train/t.pred): "+mean(dc)+" "+mean(pc))
    println("std. values (train/t.pred): "+sqrt(variance(dc))+" "+sqrt(variance(pc)))
    println("max. values (train/t.pred): "+maxi(dc)+" "+maxi(pc))
    println("min. values (train/t.pred): "+mini(dc)+" "+mini(pc))

    if (ogmats != null) {
      ogmats(0) = user;
      if (ogmats.length > 1) {
        ogmats(1) = xpreds;
      }
    }
    -sqrt(row(vv/sdata.nnz))
  }
  
  
  /** So I can set the MHTest container size appropriately. */
  def getNonzeros():Int = {
    return numNonzerosMB
  }


}


object SMF {
  trait Opts extends FactorModel.Opts {
  	var ueps = 1e-10f
  	var uconvg = 1e-3f
  	var lambdau = 5f
  	var lambdam = 5f
  	var regumean = 0f
  	var regmmean = 0f
  	var urate = 0.1f
  	var traceConverge = false
  	var doUsers = true
  	var weightByUser = false
  	var aopts:ADAGrad.Opts = null;
  	var minv = 1f;
  	var maxv = 5f;
  	var matrixOfScores = false;
    var lsgd = 0f;
  }  

  class Options extends Opts {} 
  
  def learner(mat0:Mat, d:Int) = {
    class xopts extends Learner.Options with SMF.Opts with MatSource.Opts with Grad.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = -1
    opts.npasses = 4
    opts.lrate = 0.1
    opts.initUval = 0f;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatSource(Array(mat0:Mat), opts),
  	    new SMF(opts), 
  	    null,
  	    new Grad(opts), 
  	    null,
  	    opts)
    (nn, opts)
  }

  /** 
   * Learner with single (training data) matrix as datasource, and an
   * **EXTERNAL** ADAGrad Opts. We will benchmark this with the learner using
   * an external MHTest. No internal ADAGrad updater.
   */
  def learner1(mat0:Mat, d:Int) = { 
    class xopts extends Learner.Options with SMF.Opts with MatSource.Opts with ADAGrad.Opts
    val opts = new xopts
    opts.dim = d 
    opts.putBack = -1
    opts.npasses = 4 
    opts.lrate = 0.1 
    opts.initUval = 0f; 
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    opts.aopts = null
    val nn = new Learner(
      new MatSource(Array(mat0:Mat), opts),
      new SMF(opts), 
      null,
      new ADAGrad(opts), 
      null,
      opts)
    (nn, opts)
  }

  /**
   * Learner with single (training data) matrix as datasource, and using our
   * MHTest updater. Use this for running experiments to benchmark with default
   * ADAGrad. For our experiments, we should NOT be using opts.aopts, which is
   * the internal ADAGrad updater. So that should be null ...
   */
  def learner2(mat0:Mat, d:Int) = { 
    class xopts extends Learner.Options with SMF.Opts with MatSource.Opts with ADAGrad.Opts with MHTest.Opts
    val opts = new xopts
    opts.dim = d 
    opts.putBack = -1
    opts.npasses = 4 
    opts.lrate = 0.1 
    opts.initUval = 0f; 
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    opts.aopts = null
    val nn = new Learner(
      new MatSource(Array(mat0:Mat), opts),
      new SMF(opts), 
      null,
      new MHTest(opts), 
      null,
      opts)
    (nn, opts)
  } 

  def learner(mat0:Mat, user0:Mat, d:Int) = {
    class xopts extends Learner.Options with SMF.Opts with MatSource.Opts with Grad.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = 1
    opts.npasses = 4
    opts.lrate = 0.1;
    opts.initUval = 0f;
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    val nn = new Learner(
        new MatSource(Array(mat0, user0), opts),
        new SMF(opts), 
        null,
        new Grad(opts), 
        null,
        opts)
    (nn, opts)
  }
  
  def predictor(model0:Model, mat1:Mat, preds:Mat) = {
  	class xopts extends Learner.Options with SMF.Opts with MatSource.Opts with Grad.Opts
    val model = model0.asInstanceOf[SMF]
    val nopts = new xopts;
    nopts.batchSize = math.min(10000, mat1.ncols/30 + 1)
    nopts.putBack = 1
    val newmod = new SMF(nopts);
    newmod.refresh = false
    newmod.copyFrom(model);
    val mopts = model.opts.asInstanceOf[SMF.Opts];
    nopts.dim = mopts.dim;
    nopts.uconvg = mopts.uconvg;
    nopts.lambdau = mopts.lambdau;
    nopts.lambdam = mopts.lambdam;
    nopts.regumean = mopts.regumean;
    nopts.doUsers = mopts.doUsers;
    nopts.weightByUser = mopts.weightByUser;
    val nn = new Learner(
        new MatSource(Array(mat1, preds), nopts), 
        newmod, 
        null,
        null,
        null,
        nopts)
    (nn, nopts)
   }
  
  /** A class for one of the SMF predictors. */
  class PredOpts extends Learner.Options with SMF.Opts with MatSource.Opts with MatSink.Opts with Grad.Opts with ADAGrad.Opts

  /** 
   * A predictor which will store the predictions in the predictor model
   * matrices. It forms an empty matrix to be populated by the `user` matrices,
   * which turns into the second factor matrix. It mirrors an SFA predictor code
   * which also forms this empty matrix into the matrix datasource, with the
   * only difference being the lack of an Minv option for `newmod`.
   * 
   * @param mat1 The TRAINING DATA matrix. NOT THE TESTING DATA!!! NOT THE
   * 		TESTING DATA!!!
   * @param preds The non-zeros of the TESTING data (not training).
   */
  def predictor1(model0:Model, mat1:Mat, preds:Mat) = { 
    val model = model0.asInstanceOf[SMF]
    val nopts = new PredOpts;
    nopts.batchSize = math.min(10000, mat1.ncols/30 + 1)
    nopts.putBack = -1
    nopts.initUval = 0f // Daniel: for consistency with training update.
    val newmod = new SMF(nopts);
    newmod.refresh = false
    newmod.copyFrom(model);
    val mopts = model.opts.asInstanceOf[SMF.Opts];
    nopts.dim = mopts.dim;
    nopts.uconvg = mopts.uconvg;
    nopts.lambdau = mopts.lambdau;
    nopts.lambdam = mopts.lambdam;
    nopts.regumean = mopts.regumean;
    nopts.doUsers = mopts.doUsers;
    nopts.weightByUser = mopts.weightByUser;
    nopts.nmats = 2;
    val nn = new Learner(
        new MatSource(Array(mat1, zeros(mopts.dim, mat1.ncols), preds), nopts), 
        newmod, 
        null,
        null,
        new MatSink(nopts),
        nopts)
    (nn, nopts)
  } 
}
