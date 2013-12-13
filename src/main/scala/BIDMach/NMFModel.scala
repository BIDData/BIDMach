package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._


class NMFModel(opts:NMFModel.Opts = new NMFModel.Options) extends FactorModel(opts) { 
  
  var mm:Mat = null
  var mdiag:Mat = null
  var udiag:Mat = null
  
  override def init(datasource:DataSource) = {
  	super.init(datasource)
  	mm = modelmats(0)
    modelmats = new Array[Mat](2)
    modelmats(0) = mm
    modelmats(1) = mm.zeros(mm.nrows, mm.ncols)
  	updatemats = new Array[Mat](2)
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols)
    updatemats(1) = mm.zeros(mm.nrows, mm.ncols)
    udiag = mkdiag(opts.uprior*ones(opts.dim,1))
  	mdiag = mkdiag(opts.mprior*ones(opts.dim,1))
    if (useGPU) {
      udiag = GMat(udiag)
      mdiag = GMat(mdiag)
    }
  }
  
  override def uupdate(sdata:Mat, user:Mat) = {
	  val modeldata = mm * sdata
  	val mmu = mm *^ mm + udiag
    for (i <- 0 until opts.uiter) {
    	val quot =  modeldata / (mmu * user)               
    	min(10.0f, max(0.1f, quot, quot), quot)
    	user ~ user *@ quot
    	max(opts.minuser, user, user)
    }
  }  
  
  override def mupdate(sdata:Mat, user:Mat):Unit = {
    val uu = user *^ user + mdiag *@ (1.0f*size(user,2)/opts.nusers) 
    updatemats(0) ~ (user *^ sdata) *@ mm
    updatemats(1) ~ uu * mm
    max(updatemats(1), opts.NMFeps, updatemats(1))
  }

  override def mupdate2(sdata:Mat, user:Mat):Unit = {
    val uu = user *^ user + mdiag *@ (1.0f*size(user,2)/opts.nusers)
    updatemats(0) ~ user *^ sdata
    updatemats(1) ~ uu * mm
  }
  
  override def evalfun(sdata:Mat, user:Mat):FMat = {
    if (opts.doubleScore) {
      evalfunx(sdata, user)
    } else {
    	val modeldata =  mm * sdata
    	val uu = user *^ user + mdiag *@ (1.0f*size(user,2)/opts.nusers)
    	val mmm = mm *^ mm

    	val ll0 =  sdata.contents ddot sdata.contents
    	val ll1 =  modeldata ddot user
    	val ll2 =  uu ddot mmm
    	val v1  =              (-ll0 + 2*ll1 - ll2)/sdata.nnz
    	val v2 =               -opts.uprior*(user ddot user)/sdata.nnz
    	row(v1,v2)
    }
  }
  
  def evalfunx(sdata0:Mat, user0:Mat):FMat = { 
    val sdata = SDMat(sdata0)
    val user = DMat(user0)
    val mmf = DMat(mm)
    val mdiagf = DMat(mdiag)

	  val modeldata =  mmf * sdata
    val uu = user *^ user + mdiagf *@ (1.0f*size(user,2)/opts.nusers)
    val mmm = mmf *^ mmf

    val ll0 =  sdata.contents ddot sdata.contents
    val ll1 =  modeldata ddot user
    val ll2 =  uu ddot mmm
    val v1  =              (-ll0 + 2*ll1 - ll2)/sdata.nnz
    val v2 =               -opts.uprior*(user ddot user)/sdata.nnz
    row(v1,v2)
  }
}

object NMFModel  {
  trait Opts extends FactorModel.Opts {
    var NMFeps = 1e-12
    var uprior = 0.01f
    var mprior = 1e-4f
    var nusers = 100000
  }
  
  class Options extends Opts {}
  
  def mkNMFmodel(fopts:Model.Opts) = {
  	new NMFModel(fopts.asInstanceOf[NMFModel.Opts])
  } 
   
  def mkUpdater(nopts:Updater.Opts) = {
  	new IncNormUpdater(nopts.asInstanceOf[IncNormUpdater.Opts])
  }
        
  def learn(mat0:Mat, d:Int = 256) = {
    class xopts extends Learner.Options with NMFModel.Opts with MatDataSource.Opts with IncNormUpdater.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = 1
    opts.blockSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatDataSource(Array(mat0:Mat), opts), 
  			new NMFModel(opts), 
  			null, 
  			new IncNormUpdater(opts), opts)
    (nn, opts)
  }
  
  def learnBatch(mat0:Mat, d:Int = 256) = {
    class xopts extends Learner.Options with NMFModel.Opts with MatDataSource.Opts with BatchNormUpdater.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = 1
    opts.blockSize = math.min(100000, mat0.ncols/30 + 1)
    val nn = new Learner(
        new MatDataSource(Array(mat0:Mat), opts), 
        new NMFModel(), 
        null, 
        new BatchNormUpdater(opts), 
        opts)
    (nn, opts)
  }
  
  def learnFPar(
    nstart:Int=FilesDataSource.encodeDate(2012,3,1,0),
		nend:Int=FilesDataSource.encodeDate(2012,12,1,0)
		) = { 	
  	new LearnFParModel(
  			new NMFModel.Options, mkNMFmodel _, 
  	    new IncNormUpdater.Options, mkUpdater _, 
  	    (n:Int, i:Int) => SFilesDataSource.twitterWords(nstart, nend, n, i)
  	    )
  }
  
  def learnFParx(
    nstart:Int=FilesDataSource.encodeDate(2012,3,1,0),
		nend:Int=FilesDataSource.encodeDate(2012,12,1,0)
		) = {  	
  	 	new LearnFParModelx(
  	 		SFilesDataSource.twitterWords(nstart, nend),
  	    new NMFModel.Options, mkNMFmodel _, 
  	    new IncNormUpdater.Options, mkUpdater _)
  }
} 



