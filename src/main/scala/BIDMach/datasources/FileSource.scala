package BIDMach.datasources
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,ND,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.io._


class FileSource(override val opts:FileSource.Opts = new FileSource.Options) extends DataSource(opts) {
	var sizeMargin = 0f;
	var blockSize = 0;
	@volatile var fileno = 0;                // Index of the file to be read. 
	var fileptrs:IMat = null;                // Indices of files to be written. 
	var colno = 0;                           // Column of input matrix that has been read                 
	var nstart = 0;                          // First file index
	var nend = 0;                            // Last file index (exclusive)
	var fnames:List[(Int)=>String] = null;   // List of file name mappings

	omats = null;
	var matqueue:Array[Array[Mat]] = null;
	var stop:Boolean = false;
	var pause:Boolean = true;
	var permfn:(Int)=>Int = null;
	var totalSize = 0;
	var fprogress:Float = 0;
	var executor:ExecutorService = null;
	var prefetchTasks:Array[Future[_]] = null;
	var prefetchers:Array[Prefetcher] = null;


	def softperm(nstart:Int, nend:Int) = {
		val dd1 = nstart / 24;
		val hh1 = nstart % 24;
		val dd2 = nend / 24;
		val hh2 = nend % 24;
		val (dmy, ii) = sort2(rand(dd2-dd1+1+opts.lookahead,1));
		(n:Int) => {
			val dd = n / 24;
			val hh = n % 24;
			val ddx = ii(dd-dd1)+dd1;
			val ddx0 = ddx % 31;
			val ddx1 = ddx / 31;
			val hhdd = hh + 24 * (ddx0 - 1);
			(ddx1 * 31 + (hhdd % 31 + 1)) * 24 + hhdd / 31
		}
	}

	def genperm(nstart:Int, nend:Int) = {
		val (dmy, ii) = sort2(rand(nend - nstart - 1,1));
		(n:Int) => {
			if (n >= nend - 1) {
				n
			} else {
				nstart + ii(n - nstart, 0);
			}
		}
	}

	def init = {
		initbase
		omats = new Array[Mat](fnames.size)
		for (i <- 0 until fnames.size) {
			var mm = loadMat(fnames(i)(nstart));
			val dims = mm.dims.copy;
			dims(dims.length-1) = blockSize;
			val hashdims = ND.hashInts(dims.data);
			omats(i) = mm match {
			case mf:FMat => FMat.newOrCheckFMat(dims, null, GUID, i, hashdims, "FileSource_FMat".##);
			case mi:IMat => IMat.newOrCheckIMat(dims, null, GUID, i, hashdims, "FileSource_IMat".##);
			case md:DMat => DMat.newOrCheckDMat(dims, null, GUID, i, hashdims, "FileSource_DMat".##);
			case ms:SMat => SMat.newOrCheckSMat(dims(0), dims(1), dims(1) * opts.eltsPerSample, null, GUID, i, ((dims(0)*1L) << 32) + dims(1), "FileSource_SMat".##);
			}
		}
	}

	def initbase = {
		stop = false;
		pause = true;
		if (opts.lookahead > 0) {
			executor = Executors.newFixedThreadPool(opts.lookahead + 2);
			prefetchers = new Array[Prefetcher](opts.lookahead);
			prefetchTasks = new Array[Future[_]](opts.lookahead);
		} 
		fileptrs = izeros(1, math.max(1, opts.lookahead));
		reset;
		colno = 0;
		fileno = nstart;                                                           // Number of the current output file
		matqueue = new Array[Array[Mat]](if (opts.lookahead > 0) (opts.lookahead + 2) else 2);               // Queue of matrices for each output matrix
		for (i <- 0 until matqueue.length) {
			matqueue(i) = new Array[Mat](fnames.size);
		}
		for (i <- 0 until opts.lookahead) {
			prefetchers(i) = new Prefetcher(i);
			prefetchTasks(i) = executor.submit(prefetchers(i));
		}
		pause = false;
	}

	def reset = {
		nstart = opts.nstart;
		nend = opts.nend;
		fnames = opts.fnames;
		blockSize = opts.batchSize;
		if (nend == 0) {
			while (fileExists(fnames(0)(nend))) {nend += 1}
		}
		while (!fileExists(fnames(0)(nstart)) && nstart < nend) {nstart += 1}
		if (nstart == nend) {
			throw new RuntimeException("Couldnt find any files");
		}
		if (opts.order == 0) {
			permfn = (a:Int) => a;
		} else if (opts.order == 1) {
			permfn = genperm(nstart, nend);
		} else {
			permfn = (n:Int) => {                                                    // Stripe reads across disks (different days)
				val (yy, mm, dd, hh) = FileSource.decodeDate(n);
				val hhdd = hh + 24 * (dd - 1);
				FileSource.encodeDate(yy, mm, hhdd % 31 + 1, hhdd / 31);
			}
		}
		fileptrs.synchronized {
			colno = 0;
			fileno = nstart;
			for (i <- 0 until math.max(1,opts.lookahead)) {
				fileptrs(i) = nstart + i;
			}
		}
		totalSize = nend - nstart;
	}

	def progress = {
		((fileno-nstart)*1f + fprogress)/ totalSize
	}

	def nmats = omats.length;

	def getWritten:Int = {
		var minv = 0;
		fileptrs.synchronized {
			minv = mini(fileptrs).v;
		}
		minv
	}

	def next:Array[Mat] = {
		val featType = opts.featType;
		val threshold = opts.featThreshold;
		val filex = fileno % matqueue.length;
		val filexp1 = (fileno+1) % matqueue.length;
		//    	        println("todo %d, fileno %d, filex %d, colno %d" format (todo, fileno, filex, colno))
		if (opts.lookahead > 0) {
			if (opts.traceFileSource > 0) println("next wait %d %d %d %s" format (fileno, filex, colno, fileptrs.toString));
			while (fileno > getWritten - 2) {
				Thread.sleep(1); //`yield`
			}
			if (opts.traceFileSource > 0) println("next go %d %d %d %s" format (fileno, filex, colno, fileptrs.toString));
		} else {
			fetch
		};
		var todo0 = 0;
		var todo1 = 0;
		var matqnc = 0;
		for (i <- 0 until fnames.size) {
			val matq0 = matqueue(filex)(i);
			val matq1 = matqueue(filexp1)(i);
			if (matq0.asInstanceOf[AnyRef] == null) {
				throw new RuntimeException("Missing file %d,%d" format (fileno,i));
			} else {
				matqnc = matq0.ncols;
				todo0 = math.min(blockSize, matqnc - colno);
				todo1 = if (todo0 == blockSize || (fileno+1) >= nend) 0 else math.min(blockSize - todo0, matq1.ncols);
				val off = Mat.oneBased;
				val dims = omats(i).dims.copy;
				dims(dims.length-1) = todo0 + todo1;
				omats(i) = checkCaches(dims, omats(i), GUID, i);
				if (todo1 == 0) {
					omats(i) = matq0.colslice(colno, colno+todo0, omats(i), 0);
				} else {
					omats(i) <-- (matq0.colslice(colno, colno+todo0, null, 0) \ matq1.colslice(0, todo1, null, 0));
				}

				if (featType == 0) {
					min(1f, omats(i), omats(i));
				} else if (featType == 2) {
					omats(i) ~ omats(i) >= threshold;
				}
			}
		};
		if (todo1 == 0) {
			fileptrs.synchronized {
				if (colno+todo0 == matqnc) {
					colno = 0;
					fileno += 1;
				} else {
					colno += todo0;
				}
			}
		} else {
			fileptrs.synchronized {
				fileno += 1;
				colno = todo1;
			}
		}
		fprogress = colno*1f / matqnc;
		//				println("hash %f" format sum(sum(omats(0))).dv);
		omats;
		}

		def fileExists(fname:String) = {
			val testme = new File(fname);
			testme.exists;
		}

		def fetch = {
			while (fileptrs(0) < fileno+2) {
				val pnew = permfn(fileptrs(0));
				val fexists = fileExists(fnames(0)(pnew));
				val filex = fileptrs(0) % matqueue.length;
				if (fexists) {
					for (i <- 0 until fnames.size) {
						matqueue(filex)(i) = loadMat(fnames(i)(pnew), matqueue(filex)(i));
					}
				}
				fileptrs(0) += 1;
			}
		}


		class Prefetcher(val ithread:Int) extends Runnable {

			def run() = {

				while  (!stop) {
					if (opts.traceFileSource > 0) println("prefetch wait %d %d %s" format (ithread, fileno, fileptrs.toString));
					while (pause || (fileptrs(ithread) > fileno+2+ithread && !stop)) {
						Thread.sleep(1); // Thread.`yield`
					}
					if (opts.traceFileSource > 0) println("prefetch go %d %d %s" format (ithread, fileno, fileptrs.toString));
					if (!stop) {
					  val ifile = fileptrs(ithread);
						val ifilex = ifile % matqueue.length;
						val inew = ifile + opts.lookahead;
						val pget = permfn(ifile);
						val fexists = fileExists(fnames(0)(pget));
						if (opts.traceFileSource > 0) println("prefetch %d %d pget %d %b" format (ithread, fileno, pget, fexists));
						for (i <- 0 until fnames.size) {
							if (fexists) {
								val fname = fnames(i)(pget);
								if (opts.traceFileSource > 0) println("prefetch %d %d pget %d reading %d %s" format (ithread, fileno, pget, i, fname));
								val newmat:Mat = try {
									loadMat(fname, matqueue(ifilex)(i));
								} catch {
								case e:Exception => {println(stackTraceString(e)); null}
								case _:Throwable => null
								}
								if (opts.traceFileSource > 0) println("prefetch %d %d pnew %d read %d %s " format (ithread, fileno, pget, i, fname));
								matqueue.synchronized {
									matqueue(ifilex)(i) = newmat;
								}
							} else {
								if (ifile < nend) {
									throw new RuntimeException("Missing file "+fnames(i)(pget));
								}
								matqueue.synchronized {
									matqueue(ifilex)(i) = null;
								}
							}
							//  			println("%d" format inew)
						}
						fileptrs.synchronized {
							fileptrs(ithread) = inew;
						}
					}
				}
			}
		};

		def checkCaches(dims:IMat, out:Mat, GUID:Long, i:Int):Mat = {
				if (ND.compareDims(dims.data, out.dims.data)) {
					out
				} else {
					val hashdims = ND.hashInts(dims.data);
					out match {
					case a:FMat => FMat.newOrCheckFMat(dims, null, GUID, i, hashdims, "FileSource_FMat".##);
					case a:IMat => IMat.newOrCheckIMat(dims, null, GUID, i, hashdims, "FileSource_IMat".##);
					case a:DMat => DMat.newOrCheckDMat(dims, null, GUID, i, hashdims, "FileSource_DMat".##);
					case a:SMat => SMat.newOrCheckSMat(dims(0), dims(1), a.nnz, null, GUID, i, hashdims, "FileSource_SMat".##);
					}
				}
		}

		def stackTraceString(e:Exception):String = {
				val sw = new StringWriter;
				e.printStackTrace(new PrintWriter(sw));
				sw.toString;
		}


		def hasNext:Boolean = {
				(fileno < nend)
		}

		override def close = {
				stop = true;
				for (i <- 0 until opts.lookahead) {
					prefetchTasks(i).cancel(true);
				}
				if (executor != null) executor.shutdown();
		}
}

object FileSource {

  def apply(opts:FileSource.Opts, nthreads:Int):FileSource = {
    implicit val ec = threadPool(nthreads);
    new FileSource(opts);
  }

  def apply(opts:FileSource.Opts):FileSource = apply(opts, 4);

  def apply(fname:String, opts:FileSource.Opts, nthreads:Int):FileSource = {
    opts.fnames = List(simpleEnum(fname, 1, 0));
    implicit val ec = threadPool(nthreads);
    new FileSource(opts);
  }

  def apply(fname:String, opts:FileSource.Opts):FileSource = apply(fname, opts, 4);

  def apply(fname:String):FileSource = apply(fname, new FileSource.Options, 4);

  def apply(fn1:String, fn2:String, opts:FileSource.Opts, nthreads:Int) = {
    opts.fnames = List(simpleEnum(fn1, 1, 0), simpleEnum(fn2, 1, 0));
    implicit val ec = threadPool(nthreads);
    new FileSource(opts);
  }

  def apply(fn1:String, fn2:String, opts:FileSource.Opts):FileSource = apply(fn1, fn2, opts, 4);

  def encodeDate(yy:Int, mm:Int, dd:Int, hh:Int) = (((12*yy + mm) * 31) + dd)*24 + hh

  def decodeDate(n:Int):(Int, Int, Int, Int) = {
    val days = n / 24
    val dd = (days - 1) % 31 + 1
    val months = (days - dd) / 31
    val mm = (months - 1) % 12 + 1
    val yy = (months - mm) / 12
    (yy, mm, dd, n % 24)
  }

  def sampleFun(fname:String):(Int)=>String = {
    (n:Int) => {
      val (yy, mm, dd, hh) = decodeDate(n)
      (fname format ((n / 24) % 16, yy, mm, dd, hh))
    }
  }

  def sampleFun(fname:String, m:Int, i:Int):(Int)=>String = {
    (n0:Int) => {
      val n = n0 * m + i
      val (yy, mm, dd, hh) = decodeDate(n)
      (fname format ((n / 24) % 16, yy, mm, dd, hh))
    }
  }

  def simpleEnum(fname:String, m:Int, i:Int):(Int)=>String = {
    (n0:Int) => {
      val n = n0 * m + i
      (fname format n)
    }
  }

  def simpleEnum(fname:String):(Int)=>String = simpleEnum(fname,1,0);

  trait Opts extends DataSource.Opts {
    val localDir:String = ""
    var fnames:List[(Int)=>String] = null
    var lookahead = 2
    var sampleFiles = 1.0f
    var nstart:Int = 0
    var nend:Int = 0
    var dorows:Boolean = false
    var order:Int = 0                          // 0 = sequential order, 1 = random
    var eltsPerSample = 10;
    var throwMissing:Boolean = false
    var traceFileSource = 0;
  }

  class Options extends Opts {}
}
