package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.datasinks._
import BIDMach.updaters._
import BIDMach._

/**
 * KMeans
 * {{{
 * val (nn, opts) = KMeans.learner(a)
 * opts.what             // prints the available options
 * opts.dim=200          // customize options
 * nn.train              // rain the learner
 * nn.modelmat           // get the final model
 * 
 * val (nn, opts) = KMeans.learnPar(a) // Build a parallel learner
 * opts.nthreads=2       // number of threads (defaults to number of GPUs)
 * nn.train              // train the learner
 * nn.modelmat           // get the final model
 * }}}
 */

class KMeans(override val opts:KMeans.Opts = new KMeans.Options) extends ClusteringModel(opts) {

//  var mm:Mat = null
  def um = {updatemats(0)}
  def umcount = {updatemats(1)}
  // var umcount:Mat = null
  var modelsreduced:Int = 1
  
  def mm = {modelmats(0)}
  def mmnorm = {modelmats(1)}
    
  override def init() = {
    super.init()
    if (refresh) {
      setmodelmats(Array(mm, mm dotr mm))
    }
    for (i <- 0 until modelmats.length) modelmats(i) = convertMat(modelmats(i))
    updatemats = Array(um, mm.zeros(mm.nrows, 1))
    for (i <- 0 until updatemats.length) updatemats(i) = convertMat(updatemats(i))
    //um = updatemats(0)
    //umcount = mm.zeros(mm.nrows, 1)
    //updatemats = Array(um, umcount)
  }
  
  def mupdate(sdata:Mat, ipass:Int):Unit = {
//  println("trace data %f" format sum(sum(sdata)).dv)
    val vmatch = -2 * mm * sdata + mmnorm + snorm(sdata)           // vmatch(i,j) = squared distance from data sample j to centroid i
    val bestm = vmatch <= mini(vmatch)                             // mini(vmatch) are the minimum
    bestm ~ bestm / sum(bestm)
    um ~ um + bestm *^ sdata     
    umcount ~ umcount + sum(bestm, 2)
  }
    
  def evalfun(sdata:Mat):FMat = {  
    val vmatch = -2 * mm * sdata + mmnorm + snorm(sdata)
    val (vm, im) = mini2(vmatch)
    if (ogmats != null) {ogmats(0) = im;}
    max(vm, 0f, vm)
    val vv = mean(vm).dv
    row(-vv)
  }
  
  override def evalfun(sdata:Mat, targ:Mat):FMat = {  
    val vmatch = -2 * mm * sdata + mmnorm + snorm(sdata)
    val (vm, im) = mini2(vmatch)
    if (ogmats != null) {ogmats(0) = im;}
    max(vm, 0f, vm)
    val vv = mean(vm).dv
    row(-vv)
  }
  
  override def updatePass(ipass:Int) = {
    if (ipass > 0) {
      max(umcount, 1f, umcount)
      mm ~ um / umcount
    }
    um.clear
    umcount.clear
    mmnorm ~ mm dotr mm
  }
  
  override def mergeModelFn(models:Array[Model], mm:Array[Mat], um:Array[Mat], istep:Long) = {}
  
  override def mergeModelPassFn(models:Array[Model], mmx:Array[Mat], umx:Array[Mat], ipass:Int) = {
    val nmodels = models.length
    mmx(0).clear
    if (ipass == 0) {                     // on first pass, model is random samples, so take a mixed sample
      val m0 = models(0).modelmats(0)
      val isel = umx(0).zeros(m0.nrows, 1)
      val vsel = min((nmodels-1).toFloat, floor(nmodels*rand(m0.nrows, 1)))
      for (i <- 0 until nmodels) {
        isel <-- (vsel == i.toFloat)
        umx(0) <-- models(i).modelmats(0)
        umx(0) ~ isel *@ umx(0)
        mmx(0) ~ mmx(0) + umx(0)
      }
    } else {                              // on later passes, average the centers
      for (i <- 0 until nmodels) {
        umx(0) <-- models(i).modelmats(0)
        mmx(0) ~ mmx(0) + umx(0)
      }
      mmx(0) ~ mmx(0) * (1f/nmodels)
    }
    mmx(1) ~ mmx(0) dotr mmx(0)
    for (i <- 0 until nmodels) {
      models(i).modelmats(0) <-- mmx(0)
      models(i).modelmats(1) <-- mmx(1)
    }
  }

  override def combineModels(ipass:Int, model: Model):Model = {
    val other:KMeans = model.asInstanceOf[KMeans]
    if (ipass == 0) {
      val total_models_reduced = modelsreduced + other.modelsreduced
      val isel = mm.zeros(mm.nrows, 1)
      val vsel = min((total_models_reduced-1).toFloat, floor(total_models_reduced*rand(mm.nrows, 1)))
      isel <-- (vsel < modelsreduced.toFloat)
      mm ~ isel *@ mm
      mm ~ mm + (1-isel) *@ other.mm
      modelsreduced = total_models_reduced
    } else {
      um ~ um + other.um
      umcount ~ umcount + other.umcount
    }
    this
  }
}

object KMeans  {
  trait Opts extends ClusteringModel.Opts {
  }
  
  class Options extends Opts {}
  
  def mkKMeansModel(fopts:Model.Opts) = {
    new KMeans(fopts.asInstanceOf[KMeans.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
    new Batch(nopts.asInstanceOf[Batch.Opts])
  } 
  
  class MatOptions extends Learner.Options with KMeans.Opts with MatSource.Opts with Batch.Opts
   
  def learner(mat0:Mat, d:Int):(Learner, MatOptions) = {
    val opts = new MatOptions
    opts.dim = d
    opts.batchSize = math.min(100000, mat0.ncols/30 + 1)
    opts.npasses = 10
    val nn = new Learner(
        new MatSource(Array(mat0:Mat), opts), 
        new KMeans(opts), 
        null,
        new Batch(opts), 
        null,
        opts)
    (nn, opts)
  }
  
  def learner(mat0:Mat):(Learner, MatOptions) = learner(mat0, 256)
  
  class FileOptions extends Learner.Options with KMeans.Opts with FileSource.Opts with Batch.Opts
  /**
   * KMeans with a files dataSource
   */
  def learner(fnames:List[(Int)=>String], d:Int):(Learner, FileOptions) = {
    val opts = new FileOptions
    opts.dim = d
    opts.fnames = fnames
    opts.batchSize = 10000
    implicit val threads = threadPool(4)
    val nn = new Learner(
        new FileSource(opts), 
        new KMeans(opts), 
        null,
        new Batch(opts), 
        null,
        opts)
    (nn, opts)
  }
  
  def learner(fnames:List[(Int)=>String]):(Learner, FileOptions) = learner(fnames, 256)
  
  def learner(fnames:String, d:Int):(Learner, FileOptions) = learner(List(FileSource.simpleEnum(fnames,1,0)), d)
  
  def learner(fnames:String):(Learner, FileOptions) = learner(List(FileSource.simpleEnum(fnames,1,0)), 256)

  class IteratorOptions extends Learner.Options with KMeans.Opts with IteratorSource.Opts with Batch.Opts

  def learner():(Learner, IteratorOptions) = {
    val opts = new IteratorOptions
    val nn = new Learner(
      null,
      new KMeans(opts),
      null,
      new Batch(opts),
      null,
      opts)
    (nn, opts)
  }
  
  class PredOptions extends Learner.Options with KMeans.Opts with MatSource.Opts with MatSink.Opts
  
    // This function constructs a predictor from an existing model 
  def predictor(model:Model, mat1:Mat):(Learner, PredOptions) = {
    val nopts = new PredOptions
    nopts.batchSize = math.min(10000, mat1.ncols/30 + 1)
    nopts.dim = model.opts.dim
    val newmod = new KMeans(nopts)
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
  
  class FilePredOptions extends Learner.Options with KMeans.Opts with FileSource.Opts with FileSink.Opts
  
    // This function constructs a file-based predictor from an existing model 
  def predictor(model:Model, infnames:String, outfnames:String):(Learner, FilePredOptions) = {
    val nopts = new FilePredOptions
    nopts.batchSize = 10000
    nopts.dim = model.opts.dim
    nopts.fnames = List(FileSource.simpleEnum(infnames,1,0))
    nopts.ofnames = List(FileSource.simpleEnum(outfnames,1,0))
    val newmod = new KMeans(nopts)
    newmod.refresh = false
    model.copyTo(newmod)
    implicit val threads = threadPool(4)
    val nn = new Learner(
        new FileSource(nopts), 
        newmod, 
        null,
        null,
        new FileSink(nopts),
        nopts)
    (nn, nopts)
  }
   
  class ParOptions extends ParLearner.Options with KMeans.Opts with MatSource.Opts with Batch.Opts
  
  def learnPar(mat0:Mat, d:Int):(ParLearnerF, ParOptions) = {
    val opts = new ParOptions
    opts.dim = d
    opts.batchSize = math.min(100000, mat0.ncols/30/opts.nthreads + 1)
    opts.npasses = 10
    opts.coolit = 0 // Assume we dont need cooling on a matrix input
    val nn = new ParLearnerF(
        new MatSource(Array(mat0:Mat), opts), 
        opts, mkKMeansModel _, 
        null, null, 
        opts, mkUpdater _,
        null, null,
        opts)
    (nn, opts)
  }
  
  def learnPar(mat0:Mat):(ParLearnerF, ParOptions) = learnPar(mat0, 256)
  
  class KSFopts extends ParLearner.Options with KMeans.Opts with FileSource.Opts with Batch.Opts
  
  def learnPar(fnames:String, d:Int):(ParLearnerF, KSFopts) = learnPar(List(FileSource.simpleEnum(fnames,1,0)), d)
  
  def learnPar(fnames:List[(Int)=>String], d:Int):(ParLearnerF, KSFopts) = {
    val opts = new KSFopts
    opts.dim = d
    opts.npasses = 10
    opts.fnames = fnames
    opts.batchSize = 20000
    implicit val threads = threadPool(12)
    val nn = new ParLearnerF(
        new FileSource(opts), 
        opts, mkKMeansModel _, 
        null, null, 
        opts, mkUpdater _,
        null, null,
        opts)
    (nn, opts)
  }

}


