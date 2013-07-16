package BIDMach
import BIDMat.{Mat,BMat,CMat,CSMat,Dict,DMat,FMat,GMat,GIMat,GSMat,HMat,IDict,IMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import HMat._
import scala.actors._
import java.io._

class Featurizer(val opts:Featurizer.Options = new Featurizer.Options) {
  	
  def mergeDailyDicts(rebuild:Boolean,dictname:String="dict.gz",wcountname:String="wcount.gz"):Dict = {
    val dd = new Array[Dict](5)                                               // Big enough to hold log2(days per month)
  	val nmonths = 2 + (opts.nend - opts.nstart)/31
  	val md = new Array[Dict]((math.log(nmonths)/math.log(2)).toInt)           // Big enough to hold log2(num months)
	  for (d <- opts.nstart to opts.nend) {
	    val (year, month, day) = Featurizer.decodeDate(d)
	  	val fm = new File(opts.fromMonthDir(d) + wcountname)
	    if (rebuild || ! fm.exists) {
	    	val fd = new File(opts.fromDayDir(d) + wcountname)
	    	if (fd.exists) {
	    		val bb = HMat.loadBMat(opts.fromDayDir(d) + dictname)
	    		val cc = HMat.loadIMat(opts.fromDayDir(d) + wcountname)
	    		Dict.treeAdd(Dict(bb, cc, opts.threshold), dd)
	    		print("+")
	    	}
	    	if (day == 31) {	    	  
	    		val dx = Dict.treeFlush(dd)
	    		val (sv, iv) = sortdown2(dx.counts)
	    		val dxx = Dict(dx.cstr(iv), sv)
	    		HMat.saveBMat(opts.fromMonthDir(d)+dictname, BMat(dxx.cstr))
	    		HMat.saveDMat(opts.fromMonthDir(d)+wcountname, dxx.counts)
	    	}
	    }
	    if (day == 31) {
	  		val fm = new File(opts.fromMonthDir(d) + wcountname)
	  		if (fm.exists) {
	  			val bb = HMat.loadBMat(opts.fromMonthDir(d) + dictname)
	  			val cc = HMat.loadDMat(opts.fromMonthDir(d) + wcountname)
	  			Dict.treeAdd(Dict(bb, cc, 4*opts.threshold), md)
	  		}
	  	}
	  }
  	println
  	val dy = Dict.treeFlush(md)
  	val (sv, iv) = sortdown2(dy.counts)
  	val dyy = Dict(dy.cstr(iv), sv)
  	HMat.saveBMat(opts.fromDir + dictname, BMat(dyy.cstr))
  	HMat.saveDMat(opts.fromDir + wcountname, dyy.counts)
  	dyy
	}
  
  def mergeDailyIDicts(rebuild:Boolean,dictname:String="bdict.lz4",wcountname:String="bcnts.lz4"):IDict = {
    val alldict = Dict(HMat.loadBMat(opts.mainDict))
  	val dd = new Array[IDict](5)                                               // Big enough to hold log2(days per month)
  	val nmonths = 2 + (opts.nend - opts.nstart)/31
  	val md = new Array[IDict]((math.log(nmonths)/math.log(2)).toInt)           // Big enough to hold log2(num months)
  	var dy:IDict = null
  	var mdict:Dict = null                                                     
  	var domonth:Boolean = false
  	var lastmonth = 0
	  for (d <- opts.nstart to opts.nend) {
	    val (year, month, day) = Featurizer.decodeDate(d)
	    if (month != lastmonth) {
	    	mdict = Dict(HMat.loadBMat(opts.fromMonthDir(d) + opts.localDict))     // Load token dictionary for this month
	    	val fm = new File(opts.fromMonthDir(d) + wcountname)                   // Did we process this month?
	    	domonth = ! fm.exists
	    	lastmonth = month
	    }
	    if (rebuild || domonth) {
	    	val fd = new File(opts.fromDayDir(d) + wcountname)
	    	if (fd.exists) {
	    	  val dict = Dict(HMat.loadBMat(opts.fromDayDir(d) + opts.localDict))  // Load token dictionary for this day
	    		val bb = HMat.loadIMat(opts.fromDayDir(d) + dictname)                // Load IDict info for this day
	    		val cc = HMat.loadDMat(opts.fromDayDir(d) + wcountname)
	    		val map = dict --> mdict                                             // Map from this days tokens to month dictionary
	    		val bm = map(bb)                                                     // Map the ngrams
	    		val igood = find(min(bm, 2) >= 0)                                    // Find the good ones
	    		val bg = bm(igood,?)
	    		val cg = cc(igood)
	    		val ip = icol(0->igood.length)
	    		IDict.sortlex2or3cols(bg, ip)                                        // lex sort them
	    		IDict.treeAdd(IDict(bg, cg(ip), opts.threshold), dd)                 // accumulate them
	    		print("-")
	    	}
	    	if (day == 31) {	    	                                               // On the last day, save the accumulated results
	    		val dx = IDict.treeFlush(dd)
	    		if (dx != null) {
	  				HMat.saveIMat(opts.fromMonthDir(d)+dictname, dx.grams)
	  				HMat.saveDMat(opts.fromMonthDir(d)+wcountname, dx.counts)
	  			}
	    	}
	    }
	    if (day == 31) {                                                         // Unconditionally accumulate monthly dicts
	  		val fm = new File(opts.fromMonthDir(d) + wcountname)
	  		if (fm.exists) {
	  			val bb = HMat.loadIMat(opts.fromMonthDir(d) + dictname)              // Load the IDict data for this month
	  			val cc = HMat.loadDMat(opts.fromMonthDir(d) + wcountname)
	  			val map = mdict --> alldict
	  			val bm = map(bb)                                                     // Map to global token dictionary
	    		val igood = find(min(bm, 2) >= 0)                                    // Save the good stuff
	    		val bg = bm(igood,?)
	    		val cg = cc(igood)
	    		val ip = icol(0->igood.length)
	  			IDict.sortlex2or3cols(bg, ip)
	    		IDict.treeAdd(IDict(bg, cg(ip), 4*opts.threshold), md)
	  		}
	  	}
	  }
  	dy = IDict.treeFlush(md)                                                   // Final dictionary for the time period
  	println
  	val (sv, iv) = sortdown2(dy.counts)                                        // Sort down by ngram frequency
  	val dyy = IDict(dy.grams(iv,?), sv)
  	HMat.saveIMat(opts.fromDir + dictname, dyy.grams)
  	HMat.saveDMat(opts.fromDir + wcountname, dyy.counts)
  	dyy
	}
  
  def twitterScanner(opts:Featurizer.Options, dict:Dict, idata:IMat, bigramsx:IMat, trigramsx:IMat):(Int, Int) = {
  	val isstart =  dict(opts.startItem)
  	val isend =    dict(opts.endItem)
  	val itstart =  dict(opts.startText)
  	val itend =    dict(opts.endText)
  	val ioverrun = dict(opts.overrun)
  	var active = false
  	var intext = false
  	var istatus = -1
  	var nbi = 0
  	var ntri = 0
  	var len = idata.length
  	var i = 0
  	while (i < len) {
  		val tok = idata.data(i)-1
  		if (tok >= 0) {
  			if (tok == isstart) {
  				active = true
  				istatus += 1
  			} else if (tok == itstart && active) {     							  
  				intext = true
  			} else if (tok == itend || tok == ioverrun) {
  				intext = false
  			} else if (tok == isend) {
  				intext = false
  				active = false
  			} else {
  				if (intext) {
  					val tok1 = idata.data(i-1)-1
  					if (tok1 >= 0) {   
  						if (tok1 != itstart) {
  							bigramsx(nbi, 0) = tok1
  							bigramsx(nbi, 1) = tok
  							nbi += 1
  							val tok2 = idata.data(i-2)-1
  							if (tok2 >= 0) {
  								if (tok2 != itstart) {
  									trigramsx(ntri, 0) = tok2
  									trigramsx(ntri, 1) = tok1
  									trigramsx(ntri, 2) = tok
  									ntri += 1
  								}
  							}
  						}
  					}
  				}
  			}
  		}
  		i += 1
  	}
  	(nbi, ntri)
  }
  
  def gramDicts(scanner:(Featurizer.Options, Dict, IMat, IMat, IMat)=>(Int, Int)) = {
    val nthreads = math.min(opts.nthreads, math.max(1, Mat.hasCUDA))
    for (ithread <- 0 until nthreads) {
      Actor.actor {
        if (Mat.hasCUDA > 0) setGPU(ithread)
      	val bigramsx = IMat(opts.guessSize, 2)
      	val trigramsx = IMat(opts.guessSize, 3)
      	val bdicts = new Array[IDict](5)
      	val tdicts = new Array[IDict](5)

      	for (idir <- (opts.nstart+ithread) until opts.nend by nthreads) {
      		val fname = opts.fromDayDir(idir)+opts.localDict
      		val fnew = opts.fromDayDir(idir)+"tcnts.lz4"
      		if (fileExists(fname) && !fileExists(fnew)) {
      			val dict = Dict(loadBMat(fname))
      			for (ifile <- 0 until 24) { 
      				val fn = opts.fromDayDir(idir)+opts.fromFile(ifile)
      				if (fileExists(fn)) {
      					val idata = loadIMat(fn)
      					val (nbi, ntri) = scanner(opts, dict, idata, bigramsx, trigramsx)
      					val bigrams = bigramsx(0->nbi, ?)
      					val bid = IDict.dictFromData(bigrams)
      					val trigrams = trigramsx(0->ntri, ?)
      					val trid = IDict.dictFromData(trigrams)
      					IDict.treeAdd(bid, bdicts)
      					IDict.treeAdd(trid, tdicts)      		
      				} 
      			}
      			val bf = IDict.treeFlush(bdicts)
      			val tf = IDict.treeFlush(tdicts)
      			saveIMat(opts.fromDayDir(idir) + "bdict.lz4", bf.grams)
      			saveDMat(opts.fromDayDir(idir) + "bcnts.lz4", bf.counts)
      			saveIMat(opts.fromDayDir(idir) + "tdict.lz4", tf.grams)
      			saveDMat(opts.fromDayDir(idir) + "tcnts.lz4", tf.counts)
      			print(".")
      		}
      	}
      }
    }
  }
  
  def fileExists(fname:String) = {
    val testme = new File(fname)
    testme.exists
  }
}

object Featurizer {
  def encodeDate(yy:Int, mm:Int, dd:Int) = (372*yy + 31*mm + dd)
  
  def decodeDate(n:Int):(Int, Int, Int) = {
    val yy = (n - 32) / 372
    val days = n - 32 - 372 * yy
    val mm = days / 31 + 1
    val dd = days - 31 * (mm - 1) + 1
    (yy, mm, dd)
  }
  
  def dirxMap(fname:String):(Int)=>String = {
    (n:Int) => {    
    	val (yy, mm, dd) = decodeDate(n)
    	(fname format (n % 16, yy, mm, dd))
    }    
  }
  
  def dirMap(fname:String):(Int)=>String = {
    (n:Int) => {    
    	val (yy, mm, dd) = decodeDate(n)
    	(fname format (yy, mm, dd))
    }    
  }
  
  class Options {
    var fromDayDir:(Int)=>String = dirxMap("/disk%02d/twitter/tokenized/%04d/%02d/%02d/")
    var fromDir = "/big/twitter/tokenized/"
    var fromMonthDir:(Int)=>String = dirMap(fromDir + "%04d/%02d/")
    var fromYearDir:(Int)=>String = dirMap(fromDir + "%04d/")
    var toDayDir:(Int)=>String = dirMap("/disk%02d/twitter/featurized/%04d/%02d/%02d/") 
    var fromFile:(Int)=>String = (n:Int) => ("tweet%02d.gz" format n)
    var toFile:(Int)=>String = (n:Int) => ("tweet%02d.txt" format n)
    var localDict:String = "dict.gz"
    var nstart:Int = encodeDate(2011,11,22)
    var nend:Int = encodeDate(2013,3,3)
    var startItem:String = "<status>"
    var endItem:String = "</status>"
    var startText:String = "<text>"
    var endText:String = "</text>"
    var overrun:String = "<user>"
    var mainDict:String = "/big/twitter/tokenized/alldict.gz"
    var mainCounts:String = "/big/twitter/tokenized/allwcount.gz"
    var threshold = 10
    var guessSize = 100000000
    var nthreads = 2
  }
  
 
}