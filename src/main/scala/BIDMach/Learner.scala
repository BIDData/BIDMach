package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Plotting._
import scala.collection.immutable.List
import scala.collection.mutable.ListBuffer
import scala.actors.Actor

case class Learner(
    val datasource:DataSource, 
    val model:Model, 
    val regularizer:Regularizer, 
    val updater:Updater, 
		val opts:Learner.Options = new Learner.Options) {
  var results:FMat = null
  
  def run() = {
    flip 
    var done = false
    var ipass = 0
    var here = 0L
    val reslist = new ListBuffer[Float]
    val samplist = new ListBuffer[Float]
    while (ipass < opts.npasses && ! done) {
      datasource.reset
      updater.clear
      var istep = 0
      println("i=%2d" format ipass)
      while (datasource.hasNext) {
        val mats = datasource.next
        here += datasource.opts.blockSize
        if ((istep + 1) % opts.evalStep == 0 || ! datasource.hasNext) {
        	val scores = model.evalblockg(mats)
        	print("ll="); scores.data.foreach(v => print(" %4.3f" format v)); println(" mem=%f" format GPUmem._1)
        	reslist.append(scores(0))
        	samplist.append(here)
        } else {
        	model.doblockg(mats, here)
        	if (regularizer != null) regularizer.compute(here)
        	updater.update(here)
        	print(".")
        }   
        if (model.opts.putBack >= 0) datasource.putBack(mats, model.opts.putBack)
        istep += 1
      }
      updater.updateM
      ipass += 1
    }
    val gf = gflop
    println("Time=%5.4f secs, gflops=%4.2f" format (gf._2, gf._1))
    results = row(reslist.toList) on row(samplist.toList)
  }
}

case class ParLearner(
    val datasources:Array[DataSource], 
    val models:Array[Model], 
    val regularizers:Array[Regularizer], 
    val updaters:Array[Updater], 
		val opts:Learner.Options = new Learner.Options) {
  
  var um:FMat = null
  var mm:FMat = null
  var results:FMat = null
  
  def run() = {
    flip 
    val mm0 = models(0).modelmats(0)
    mm = zeros(mm0.nrows, mm0.ncols)
    um = zeros(mm0.nrows, mm0.ncols)
    
    val done = izeros(opts.nthreads, 1)
    var ipass = 0
    var here = 0L
    val reslist = new ListBuffer[Float]
    val samplist = new ListBuffer[Float]
    while (ipass < opts.npasses) {     
      for (i <- 0 until opts.nthreads) {
        setGPU(i)
      	datasources(i).reset
        updaters(i).clear
      }
      var istep = 0
      println("i=%2d" format ipass)
      while (datasources(0).hasNext) {
      	here += datasources(0).opts.blockSize
      	done.clear
        for (ithread <- 0 until opts.nthreads) {
        	Actor.actor {
        		setGPU(ithread) 
        		if (datasources(ithread).hasNext) {
        			val mats = datasources(ithread).next
        	  	if ((istep + ithread + 1) % opts.evalStep == 0 || ithread == 0 && !datasources(0).hasNext ) {
        	  		val scores = models(ithread).evalblockg(mats)
        	  		print("ll="); scores.data.foreach(v => print(" %4.3f" format v)); println(" %d mem=%f" format (getGPU, GPUmem._1))
        	  		reslist.append(scores(0))
        	  		samplist.append(here)
        	  	} else {
        	  		models(ithread).doblockg(mats, here)
        	  		if (regularizers != null && regularizers(ithread) != null) regularizers(ithread).compute(here)
        	  		updaters(ithread).update(here)
        	  		print(".")
        	  	}
        			if (models(ithread).opts.putBack >= 0) datasources(ithread).putBack(mats, models(ithread).opts.putBack)
        		}
        		done(ithread) = 1   
        	}
        }
      	while (mini(done).v == 0) Thread.sleep(1)
      	istep += opts.nthreads
      	if (istep % opts.updateStep == 0) syncmodels(models)
      }
      println
      for (i <- 0 until opts.nthreads) {setGPU(i); updaters(i).updateM}
      ipass += 1
    }
    val gf = gflop
    println("Time=%5.4f secs, gflops=%4.2f" format (gf._2, gf._1))
    results = row(reslist.toList) on row(samplist.toList)
  }
     
  def syncmodels(models:Array[Model]) = {
	  mm.clear
	  for (i <- 0 until models.length) {
	  	setGPU(i)
	  	um <-- models(i).modelmats(0)
	  	mm ~ mm + um
	  }
	  mm ~ mm * (1f/models.length)
	  for (i <- 0 until models.length) {
	  	setGPU(i)
	  	models(i).modelmats(0) <-- mm
	  }
	  setGPU(0)
  }
}


case class ParLearnerx(
    val datasource:DataSource, 
    val models:Array[Model], 
    val regularizers:Array[Regularizer], 
    val updaters:Array[Updater], 
		val opts:Learner.Options = new Learner.Options) {
  
  var um:FMat = null
  var mm:FMat = null
  var results:FMat = null
  var cmats:Array[Array[Mat]] = null
  
  def run() = {
    flip 
    val mm0 = models(0).modelmats(0)
    mm = zeros(mm0.nrows, mm0.ncols)
    um = zeros(mm0.nrows, mm0.ncols)
    cmats = new Array[Array[Mat]](opts.nthreads)
    for (i <- 0 until opts.nthreads) cmats(i) = new Array[Mat](datasource.omats.length)
    
    val done = iones(opts.nthreads, 1)
    var ipass = 0
    var here = 0L
    var feats = 0L
    val reslist = new ListBuffer[Float]
    val samplist = new ListBuffer[Float]
    while (ipass < opts.npasses) {
    	datasource.reset
      for (i <- 0 until opts.nthreads) {
        setGPU(i)
        updaters(i).clear
      }
      var istep = 0
      println("i=%2d" format ipass)
      while (datasource.hasNext) {
        for (ithread <- 0 until opts.nthreads) {
        	if (datasource.hasNext) {
        	  done(ithread) = 0
        		val mats = datasource.next
        		here += datasource.opts.blockSize
        		feats += mats(0).nnz
        		for (j <- 0 until mats.length) cmats(ithread)(j) = safeCopy(mats(j), ithread)
        		Actor.actor {
        			setGPU(ithread) 
        			if ((istep + ithread + 1) % opts.evalStep == 0 || !datasource.hasNext ) {
        				val scores = models(ithread).evalblockg(cmats(ithread))
        				print("ll="); scores.data.foreach(v => print(" %4.3f" format v)); println(" %d mem=%f" format (getGPU, GPUmem._1))
        				reslist.append(scores(0))
        				samplist.append(here)
        			} else {
        				models(ithread).doblockg(cmats(ithread), here)
        				if (regularizers != null && regularizers(ithread) != null) regularizers(ithread).compute(here)
        				updaters(ithread).update(here)
        				print(".")
        			}
        			done(ithread) = 1 
        		}  
        	}
        }
      	while (mini(done).v == 0) Thread.sleep(1)
      	istep += opts.nthreads
      	if (istep % opts.updateStep == 0) syncmodels(models)
      }
      println
      for (i <- 0 until opts.nthreads) {setGPU(i); updaters(i).updateM}
      ipass += 1
      saveAs("/big/twitter/test/results.mat", row(reslist.toList) on row(samplist.toList), "results")
    }
    val gf = gflop
    println("Time=%5.4f secs, gflops=%4.2f" format (gf._2, gf._1))
    results = row(reslist.toList) on row(samplist.toList)
  }
  
  def safeCopy(m:Mat, ithread:Int):Mat = {
    m match {
      case ss:SMat => {
        val out = SMat.newOrCheckSMat(ss.nrows, ss.ncols, ss.nnz, null, m.GUID, ithread, "safeCopy".##)
        ss.copyTo(out)
      }
    }
  }
     
  def syncmodels(models:Array[Model]) = {
	  mm.clear
	  for (i <- 0 until models.length) {
	  	setGPU(i)
	  	um <-- models(i).modelmats(0)
	  	mm ~ mm + um
	  }
	  mm ~ mm * (1f/models.length)
	  for (i <- 0 until models.length) {
	  	setGPU(i)
	  	models(i).modelmats(0) <-- mm
	  }
	  setGPU(0)
  }
}


object Learner {
	class Options {
		var npasses:Int = 1
		var evalStep = 15
		var updateStep = 32
		var nthreads = 4
  }
}

class TestLDA(mat:Mat) {
  var dd:MatDataSource = null
  var model:LDAModel = null
  var updater:Updater = null
  var lda:Learner = null
  var lopts = new Learner.Options
  var mopts = new LDAModel.Options
  var dopts = new MatDataSource.Options
  def setup = { 
    val aa = if (mopts.putBack >= 0) {
    	val a = new Array[Mat](2); a(1) = ones(mopts.dim, mat.ncols); a
    } else {
      new Array[Mat](1)
    }
    aa(0) = mat
    dd = new MatDataSource(aa, dopts)
    dd.init
    model = new LDAModel(mopts)
    model.init(dd)
    updater = new IncNormUpdater()
    updater.init(model)
    lda = new Learner(dd, model, null, updater, lopts)   
  }
  
  def init = {
    if (dd.omats.length > 1) dd.omats(1) = ones(model.opts.dim, dd.omats(0).ncols)
    dd.init
    model.init(dd)
    updater.init(model)
  }
  
  def run = lda.run
}


class TestParLDA(mat:Mat) {
  var dds:Array[DataSource] = null
  var models:Array[Model] = null
  var updaters:Array[Updater] = null
  var lda:ParLearner = null
  var lopts = new Learner.Options
  var mopts = new LDAModel.Options
  var dopts = new MatDataSource.Options
  
  def setup = {
    dds = new Array[DataSource](lopts.nthreads)
    models = new Array[Model](lopts.nthreads)
    updaters = new Array[Updater](lopts.nthreads)
    for (i <- 0 until lopts.nthreads) {
      setGPU(i)
    	val istart = i * mat.ncols / lopts.nthreads
    	val iend = (i+1) * mat.ncols / lopts.nthreads
    	val mm = mat(?, istart->iend)
    	val aa = if (mopts.putBack >= 0) {
    		val a = new Array[Mat](2); 
    		a(1) = ones(mopts.dim, mm.ncols); 
    		a
    	} else {
    		new Array[Mat](1)
    	}
    	aa(0) = mm
    	dds(i) = new MatDataSource(aa, dopts)
    	dds(i).init
    	models(i) = new LDAModel(mopts)
    	models(i).init(dds(i))
    	updaters(i) = new IncNormUpdater()
    	updaters(i).init(models(i))
    }
    setGPU(0)
    lda = new ParLearner(dds, models, null, updaters, lopts)   
  }
  
  def init = {
  	for (i <- 0 until lopts.nthreads) {
  	  setGPU(i)
  		if (dds(i).omats.length > 1) dds(i).omats(1) = ones(mopts.dim, dds(i).omats(0).ncols)
  		dds(i).init
  		models(i).init(dds(i))
  		updaters(i).init(models(i))
  	}
  	setGPU(0)
  }
  
  def run = lda.run
}

class TestFParLDA(
    nstart:Int=FilesDataSource.encodeDate(2012,3,1,0),
		nend:Int=FilesDataSource.encodeDate(2012,9,1,0)
		) {
  var dds:Array[DataSource] = null
  var models:Array[Model] = null
  var updaters:Array[Updater] = null
  var lda:ParLearner = null
  var lopts = new Learner.Options
  var mopts = new LDAModel.Options
  mopts.uiter = 8
  
  def setup = {
    dds = new Array[DataSource](lopts.nthreads)
    models = new Array[Model](lopts.nthreads)
    updaters = new Array[Updater](lopts.nthreads)
    for (i <- 0 until lopts.nthreads) {
      setGPU(i)
    	val istart = nstart + i * (nend - nstart) / lopts.nthreads
    	val iend = nstart + (i+1) * (nend - nstart) / lopts.nthreads
    	val dopts = new SFilesDataSource.Options
    	dopts.fcounts = icol(50000)
    	dopts.nstart = istart
    	dopts.nend = iend
    	dopts.lookahead = 3
    	dopts.blockSize = 100000
    	dopts.sBlockSize = 4000000
    	dds(i) = new SFilesDataSource(dopts)
    	dds(i).init
    	models(i) = new LDAModel(mopts)
    	models(i).init(dds(i))
    	updaters(i) = new IncNormUpdater()
    	updaters(i).init(models(i))
    }
    setGPU(0)
    lda = new ParLearner(dds, models, null, updaters, lopts)   
  }
  
  def init = {
  	for (i <- 0 until lopts.nthreads) {
  	  setGPU(i)
  		if (dds(i).omats.length > 1) dds(i).omats(1) = ones(mopts.dim, dds(i).omats(0).ncols)
  		dds(i).init
  		models(i).init(dds(i))
  		updaters(i).init(models(i))
  	}
  	setGPU(0)
  }
  
  def run = lda.run
}


class TestFParLDAx(
    nstart:Int=FilesDataSource.encodeDate(2012,3,1,0),
		nend:Int=FilesDataSource.encodeDate(2012,9,1,0)
		) {
  var dd:DataSource = null
  var models:Array[Model] = null
  var updaters:Array[Updater] = null
  var lda:ParLearnerx = null
  var lopts = new Learner.Options
  var mopts = new LDAModel.Options
  var dopts = new SFilesDataSource.Options
  mopts.uiter = 8
  dopts.fcounts = icol(50000)
  dopts.lookahead = 8
  dopts.blockSize = 100000
  dopts.sBlockSize = 4000000
  dopts.nstart = nstart
  dopts.nend = nend
  
  def setup = {
    models = new Array[Model](lopts.nthreads)
    updaters = new Array[Updater](lopts.nthreads)
    dd = new SFilesDataSource(dopts)
    dd.init
    for (i <- 0 until lopts.nthreads) {
      setGPU(i)
    	models(i) = new LDAModel(mopts)
    	models(i).init(dd)
    	updaters(i) = new IncNormUpdater()
    	updaters(i).init(models(i))
    }
    setGPU(0)
    lda = new ParLearnerx(dd, models, null, updaters, lopts)   
  }
  
  def init = {
	  dd.omats(1) = ones(mopts.dim, dd.omats(0).ncols)
  	for (i <- 0 until lopts.nthreads) {
  	  setGPU(i)
  		if (dd.omats.length > 1) 
  		dd.init
  		models(i).init(dd)
  		updaters(i).init(models(i))
  	}
  	setGPU(0)
  }
  
  def run = lda.run
}