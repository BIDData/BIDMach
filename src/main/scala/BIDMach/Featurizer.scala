package BIDMach
import BIDMat.{Mat,BMat,CMat,CSMat,Dict,DMat,FMat,GMat,GIMat,GSMat,HMat,IDict,IMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import HMat._
import scala.actors._
import java.io._

class Featurizer(val opts:Featurizer.Options = new Featurizer.Options) {
  
  var alldict:Dict = null
  var allbdict:IDict = null
  var alltdict:IDict = null
  	
  def mergeDicts(rebuild:Boolean,dictname:String="dict.gz",wcountname:String="wcount.gz"):Dict = {
    val dd = new Array[Dict](5)                                                 // Big enough to hold log2(days per month)
  	val nmonths = 2 + (opts.nend - opts.nstart)/31
  	val md = new Array[Dict](1+(math.log(nmonths)/math.log(2)).toInt)           // Big enough to hold log2(num months)
	  for (d <- opts.nstart to opts.nend) {
	    val (year, month, day) = Featurizer.decodeDate(d)
	  	val fm = new File(opts.fromMonthDir(d) + wcountname)
	    if (rebuild || ! fm.exists) {
	    	val fd = new File(opts.fromDayDir(d) + wcountname)
	    	if (fd.exists) {
	    		val bb = loadBMat(opts.fromDayDir(d) + dictname)
	    		val cc = loadIMat(opts.fromDayDir(d) + wcountname)
	    		Dict.treeAdd(Dict(bb, cc, opts.threshold), dd)
	    		print(".")
	    	}
	    	if (day == 31) {	    	  
	    		val dx = Dict.treeFlush(dd)
	    		val (sv, iv) = sortdown2(dx.counts)
	    		val dxx = Dict(dx.cstr(iv), sv)
	    		saveBMat(opts.fromMonthDir(d)+dictname, BMat(dxx.cstr))
	    		saveDMat(opts.fromMonthDir(d)+wcountname, dxx.counts)
	    	}
	    }
	    if (day == 31) {
	  		val fm = new File(opts.fromMonthDir(d) + wcountname)
	  		if (fm.exists) {
	  			val bb = loadBMat(opts.fromMonthDir(d) + dictname)
	  			val cc = loadDMat(opts.fromMonthDir(d) + wcountname)
	  			Dict.treeAdd(Dict(bb, cc, 4*opts.threshold), md)
	  			println("%04d-%02d" format (year,month))
	  		}
	  	}
	  }
  	println
  	val dy = Dict.treeFlush(md)
  	val (sv, iv) = sortdown2(dy.counts)
  	val dyy = Dict(dy.cstr(iv), sv)
  	saveBMat(opts.mainDir + dictname, BMat(dyy.cstr))
  	saveDMat(opts.mainDir + wcountname, dyy.counts)
  	dyy
	}
  
  def mergeIDicts(rebuild:Boolean=false, dictname:String="bdict.lz4", wcountname:String="bcnts.lz4"):IDict = {
    if (alldict == null) alldict = Dict(loadBMat(opts.mainDict))
  	val dd = new Array[IDict](5)                                               // Big enough to hold log2(days per month)
  	val nmonths = 2 + (opts.nend - opts.nstart)/31
  	val md = new Array[IDict](1+(math.log(nmonths)/math.log(2)).toInt)         // Big enough to hold log2(num months)
  	var dy:IDict = null
  	var mdict:Dict = null                                                     
  	var domonth:Boolean = false
  	var lastmonth = 0
	  for (d <- opts.nstart to opts.nend) {
	    val (year, month, day) = Featurizer.decodeDate(d)
	    if (month != lastmonth) {
	      val dfname = opts.fromMonthDir(d) + opts.localDict
	      if (fileExists(dfname)) {
	      	mdict = Dict(loadBMat(dfname))                                       // Load token dictionary for this month
	      	val fm = new File(opts.fromMonthDir(d) + wcountname)                 // Did we process this month?
	      	domonth = rebuild || !fm.exists
	      } else {
	        mdict = null
	        domonth = false
	      }
	    	lastmonth = month
	    }
	    if (domonth) {
	    	val fd = new File(opts.fromDayDir(d) + wcountname)
	    	if (fd.exists) {
	    	  val dict = Dict(loadBMat(opts.fromDayDir(d) + opts.localDict))  // Load token dictionary for this day
	    		val bb = loadIMat(opts.fromDayDir(d) + dictname)                     // Load IDict info for this day
	    		val cc = loadDMat(opts.fromDayDir(d) + wcountname)
	    		val map = dict --> mdict                                             // Map from this days tokens to month dictionary
// Kludge to deal with (old) scanner problem
	    		val ig = find(maxi(bb, 2) < 0x7fffffff)
	    		val bb2 = bb(ig, ?)
	    		val bm = map(bb2) // Map the ngrams
	    		val cc2 = cc(ig,0)
// Done kludge
	    		val igood = find(mini(bm, 2) >= 0)                                    // Find the good ones
	    		val bg = bm(igood,?)
	    		val cg = cc2(igood)
	    		val ip = icol(0->igood.length)
	    		sortlexInds(bg, ip)                                            // lex sort them
	    		IDict.treeAdd(IDict(bg, cg(ip), opts.threshold), dd)                 // accumulate them
	    		print(".")
	    	}
	    	if (day == 31) {	    	                                               // On the last day, save the accumulated results
	    		val dx = IDict.treeFlush(dd)
	    		if (dx != null) {
	  				saveIMat(opts.fromMonthDir(d)+dictname, dx.grams)
	  				saveDMat(opts.fromMonthDir(d)+wcountname, dx.counts)
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
	    		val igood = find(mini(bm, 2) >= 0)                                   // Save the good stuff
	    		val bg = bm(igood,?)
	    		val cg = cc(igood)
	    		val ip = icol(0->igood.length)
	  			sortlexInds(bg, ip)
	    		IDict.treeAdd(IDict(bg, cg(ip), 4*opts.threshold), md)
	    		println("%04d-%02d" format (year,month))
	  		}
	  	}
	  }
  	dy = IDict.treeFlush(md)                                                   // Final dictionary for the time period
  	println
  	val (sv, iv) = sortdown2(dy.counts)                                        // Sort down by ngram frequency
  	val dyy = IDict(dy.grams(iv,?), sv)
  	saveIMat(opts.mainDir + dictname, dyy.grams)
  	saveDMat(opts.mainDir + wcountname, dyy.counts)
  	dyy
	}
  
 
  def mkIDicts(rebuild:Boolean=false, scanner:Scanner=TwitterScanner) = {
    val nthreads = math.min(opts.nthreads, math.max(1, Mat.hasCUDA))
    val done = izeros(nthreads,1)
    for (ithread <- 0 until nthreads) {
      Actor.actor {
        if (Mat.hasCUDA > 0) setGPU(ithread+Mat.hasCUDA-nthreads)
      	val bigramsx = IMat(opts.guessSize, 3)
      	val trigramsx = IMat(opts.guessSize, 4)
      	val bdicts = new Array[IDict](5)
      	val tdicts = new Array[IDict](5)

      	for (d <- (opts.nstart+ithread) to opts.nend by nthreads) {
      		val (year, month, day) = Featurizer.decodeDate(d)
      		val fname = opts.fromDayDir(d)+opts.localDict
      		val fnew = opts.fromDayDir(d)+opts.triCnts
      		if (fileExists(fname) && (rebuild || !fileExists(fnew))) {
      			val dict = Dict(loadBMat(fname))
      			for (ifile <- 0 until 24) { 
      				val fn = opts.fromDayDir(d)+opts.fromFile(ifile)
      				if (fileExists(fn)) {
      					val idata = loadIMat(fn)
      					val (nuni, nbi, ntri, nusers) = scanner.scan(opts, dict, idata, null, bigramsx, trigramsx, null)
      					val bigrams = bigramsx(0->nbi, 0->2) 
      					val bid = if (nbi > 0) IDict.dictFromData(bigrams) else null
      					val trigrams = trigramsx(0->ntri, 0->3)
      					val trid = if (ntri > 0) IDict.dictFromData(trigrams) else null
      					IDict.treeAdd(bid, bdicts)
      					IDict.treeAdd(trid, tdicts)      		
      				} 
      			}
      			val bf = IDict.treeFlush(bdicts)
      			val tf = IDict.treeFlush(tdicts)
      			saveIMat(opts.fromDayDir(d) + opts.biDict, bf.grams)
      			saveDMat(opts.fromDayDir(d) + opts.biCnts, bf.counts)
      			saveIMat(opts.fromDayDir(d) + opts.triDict, tf.grams)
      			saveDMat(opts.fromDayDir(d) + opts.triCnts, tf.counts)
      			print(".")
      		}
      		if (ithread == 0 && day/nthreads == 31/nthreads) println("%04d-%02d" format (year,month))
      	}
        done(ithread,0) = 1
      }
    }
    while (mini(done).v == 0) Thread.`yield`
  }
  
  def mkUniFeats(map:IMat, gramsx:IMat, ng:Int):IMat = {
  	val unis = map(gramsx(0->ng, 0))
  	val igood = find(unis >= 0) 
  	val gg = unis(igood, 0)
  	val ggn = gramsx(igood, 1)
    val feats = ggn \ gg
    sortlex(feats)
    val (outr, ix, iy) = uniquerows(feats)
    val fcounts = (ix(1->ix.length, 0) on iy.length) - ix
    outr \ fcounts 
  }
  
  def mkGramFeats(map:IMat, gramsx:IMat, ng:Int, alldict:IDict):IMat = {
  	val grams = map(gramsx(0->ng, 0->(gramsx.ncols-1)))
  	val igood = find(mini(grams, 2) >= 0) 
  	val gg = grams(igood,?)
  	val ggn = gramsx(igood, gramsx.ncols-1)
  	val gmap = IDict(gg) --> alldict
  	val igood2 = find(gmap >= 0)
    val feats = ggn(igood2,0) \ gmap(igood2,0)
    sortlex(feats)
    val (outr, ix, iy) = uniquerows(feats)
    val fcounts = (ix(1->ix.length, 0) on iy.length) - ix
    outr \ fcounts 
  }
  
  def featurize(rebuild:Boolean=false, scanner:Scanner=TwitterScanner) = {
    if (alldict == null) alldict = Dict(HMat.loadBMat(opts.mainDict))
  	if (allbdict == null) allbdict = IDict(HMat.loadIMat(opts.mainBDict))
  	if (alltdict == null) alltdict = IDict(HMat.loadIMat(opts.mainTDict))
  	alldict.makeHash
  	allbdict.makeSorted
  	alltdict.makeSorted
    val nthreads = math.min(opts.nthreads, math.max(1, Mat.hasCUDA))
    val done = izeros(nthreads,1)
    for (ithread <- 0 until nthreads) {
      Actor.actor {
        if (Mat.hasCUDA > 0) setGPU(ithread+Mat.hasCUDA-nthreads)
        val unigramsx = IMat(opts.guessSize, 2)
      	val bigramsx = IMat(opts.guessSize, 3)
      	val trigramsx = IMat(opts.guessSize, 4)
      	val userids = IMat(opts.guessSize/10, 2)
      	for (d <- (opts.nstart+ithread) to opts.nend by nthreads) {
      		val (year, month, day) = Featurizer.decodeDate(d)
      		val fdict = opts.fromDayDir(d)+opts.localDict
      		if (fileExists(fdict)) {
      			var dict:Dict = null 
      			var map:IMat = null
      			val fd = new File(opts.toDayDir(d))
      		  if (!fd.exists) fd.mkdirs
      		  for (ifile <- 0 until 24) { 
      		  	val fn = opts.fromDayDir(d)+opts.fromFile(ifile)
      		  	val fx = opts.toDayDir(d)+opts.toTriFeats(ifile)
      		  	if (fileExists(fn) && (rebuild || !fileExists(fx))) {
      		  		if (dict == null) {
      		  			dict = Dict(loadBMat(fdict))
      		  			map = dict --> alldict
      		  		}
      		  		val idata = loadIMat(fn)
      		  		val (nuni, nbi, ntri, nusers) = scanner.scan(opts, dict, idata, unigramsx, bigramsx, trigramsx, userids)
      		  		val unifeats = mkUniFeats(map, unigramsx, nuni)
      		  		val bifeats = mkGramFeats(map, bigramsx, nbi, allbdict)
      		  		val trifeats = mkGramFeats(map, trigramsx, ntri, alltdict)   
      		  		saveIMat(opts.toDayDir(d) + opts.toUniFeats(ifile), unifeats)
      		  		saveIMat(opts.toDayDir(d) + opts.toBiFeats(ifile), bifeats)
      		  		saveIMat(opts.toDayDir(d) + opts.toTriFeats(ifile), trifeats)
      		  		saveIMat(opts.toDayDir(d) + opts.toUserids(ifile), userids(nusers, ?))
      		  		if (ifile == 23) print(".")
      		  	} 
      		  }   		
      		}
      		if (ithread == 0 && day/nthreads == 31/nthreads) println("%04d-%02d" format (year,month))
      	}
        done(ithread,0) = 1
      }
    }
    while (mini(done).v == 0) Thread.`yield`
  }
  
  def fileExists(fname:String) = {
    val testme = new File(fname)
    testme.exists
  }  
    
  def loadDicts() = {
	  if (alldict == null) alldict = Dict(HMat.loadBMat(opts.mainDict))
	  if (allbdict == null) allbdict = IDict(HMat.loadIMat(opts.mainBDict))
	  if (alltdict == null) alltdict = IDict(HMat.loadIMat(opts.mainTDict))
	  val alld = alldict.cstr
	  val bg = allbdict.grams
	  val tg = alltdict.grams
	  val bd = CSMat(bg.nrows,1)
	  val td = CSMat(tg.nrows,1)
	  var i = 0
	  while (i < bg.nrows) {
	    bd(i) = alld(bg(i,0)) + " " + alld(bg(i,1))
	    i += 1
	  }
	  i = 0
	  while (i < tg.nrows) {
	    td(i) = (alld(tg(i,0)) + " " + alld(tg(i,1))) + (" " + alld(tg(i,2)))
	    i += 1
	  }
	  (alld, bd, td)
  }
}

object Featurizer {
  
  def alloptions = {
    val ff = new Featurizer
    val newopts = new Featurizer.Options{
      override val tokDirName = "twitter/smiley/tokenized/"
      override val featDirName = "twitter/smiley/featurized/"
    }
    val fs = new Featurizer(newopts)
    (ff,fs)
  }
  
  def buildAll(rebuild:Boolean=false) = {
    buildMainDict(rebuild)
    buildMainGDicts(rebuild)
    buildFeatures(rebuild)
  }
  
  def buildMainDict(rebuild:Boolean=false) = {
  	val (ff,fs) = alloptions    
    val d1 = ff.mergeDicts(rebuild)
    val d2 = fs.mergeDicts(rebuild)
    val dd = Dict.union(d1, d2)
    val (sc, ic) = sortdown2(dd.counts)
    saveBMat(ff.opts.mainDict, BMat(dd.cstr(ic,0)))
  	saveDMat(ff.opts.mainCounts, sc)
  }
 
  def buildMainGDicts(rebuild:Boolean=false) = {
    val (ff, fs) = alloptions
    ff.mkIDicts(rebuild)
    fs.mkIDicts(rebuild)

  	val bd1 = ff.mergeIDicts(rebuild)
  	val bd2 = fs.mergeIDicts(rebuild)
  	val bdd = IDict.merge2(bd1,bd2)
  	val (sbc, ibc) = sortdown2(bdd.counts)
    saveIMat(ff.opts.mainBDict, IMat(bdd.grams(ibc,?)))
  	saveDMat(ff.opts.mainBCounts, sbc)
  	
  	val td1 = ff.mergeIDicts(rebuild, "tdict.lz4", "tcnts.lz4")
  	val td2 = fs.mergeIDicts(rebuild, "tdict.lz4", "tcnts.lz4")
  	val tdd = IDict.merge2(td1,td2)
  	val (stc, itc) = sortdown2(tdd.counts)
    saveIMat(ff.opts.mainTDict, IMat(tdd.grams(itc,?)))
    saveDMat(ff.opts.mainTCounts, stc)
  }
  
  def buildFeatures(rebuild:Boolean=false) = {
    val (ff, fs) = alloptions
    ff.featurize(rebuild)
    fs.featurize(rebuild) 
  }
  
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
    val tokDirName = "twitter/tokenized/"
    val featDirName = "twitter/featurized/"
  	def mainDir = "/big/" + tokDirName
  	def mainDict:String = "/big/twitter/tokenized/alldict.gz"
    def mainCounts:String = "/big/twitter/tokenized/allwcount.gz"
    def mainBDict:String = "/big/twitter/tokenized/allbdict.lz4"
    def mainBCounts:String = "/big/twitter/tokenized/allbcnts.lz4"
    def mainTDict:String = "/big/twitter/tokenized/alltdict.lz4"
    def mainTCounts:String = "/big/twitter/tokenized/alltcnts.lz4"
  	def fromYearDir:(Int)=>String = dirMap(mainDir + "%04d/")
    def fromMonthDir:(Int)=>String = dirMap(mainDir + "%04d/%02d/")
    def fromDayDir:(Int)=>String = dirxMap("/disk%02d/" + tokDirName + "%04d/%02d/%02d/")
    def toDayDir:(Int)=>String = dirxMap("/disk%02d/" + featDirName + "%04d/%02d/%02d/") 
    var fromFile:(Int)=>String = (n:Int) => ("tweet%02d.gz" format n)
    var toUniFeats:(Int)=>String = (n:Int) => ("unifeats%02d.lz4" format n)
    var toBiFeats:(Int)=>String = (n:Int) => ("bifeats%02d.lz4" format n)
    var toTriFeats:(Int)=>String = (n:Int) => ("trifeats%02d.lz4" format n)
    var toUserids:(Int)=>String = (n:Int) => ("userids%02d.lz4" format n)
    var localDict:String = "dict.gz"
    var biDict:String = "bdict.lz4"
    var triDict:String = "tdict.lz4"
    var biCnts:String = "bcnts.lz4"
    var triCnts:String = "tcnts.lz4"
    var nstart:Int = encodeDate(2011,11,22)
    var nend:Int = encodeDate(2013,7,16)
    var threshold = 10
    var guessSize = 100000000
    var nthreads = 2
  }
}

trait Scanner { 
	def scan(opts:Featurizer.Options, dict:Dict, idata:IMat, unigramsx:IMat, bigramsx:IMat, trigramsx:IMat, userids:IMat):(Int, Int, Int, Int)
}

object TwitterScanner extends Scanner {  
	def scan(opts:Featurizer.Options, dict:Dict, idata:IMat, unigramsx:IMat, bigramsx:IMat, trigramsx:IMat, userids:IMat):(Int, Int, Int, Int) = {
		val isstart =  dict("<status>")
		val isend =    dict("</status>")
		val irstart =  dict("<retweet>")
		val irend =    dict("</retweet>")
		val itstart =  dict("<text>")
		val itend =    dict("</text>")
		val iuser  =   dict("<user>")
		val iuend  =   dict("</user>")
		val iistart =  dict("<id>")
		val iiend  =   dict("</id>")
		var state = 0
		/*
		 * state = 0: outside a status
		 * state = 1: inside a status
		 * state = 2: inside the first-level user tag
		 * state = 3: inside the id of a first-level user tag
		 * state = 4: inside first-level text field
		 * state = 5: inside a retweet
		 * state = 6: inside the status tag in a retweet
		 * state = 7: inside user field of retweet
		 * state = 8: inside the id of a user in a retweet
		 * state = 9: inside text field of retweet
		 */

		var istatus = -1
		var nuni = 0
		var nbi = 0
		var ntri = 0
		var len = idata.length
		var i = 0
		while (i < len) {
			val tok = idata.data(i)-1
			if (tok == isend) {
				state = 0
			} else {
				state match {
				case 0 => if (tok == isstart) {
					state = 1
					istatus += 1
				}
				case 1 => if (tok == iuser) {
					state = 2
				} else if (tok == itstart) {
					state = 4
				}	else if (tok == irstart) {
					state = 5
				} 
				case 2 => if (tok == iistart) {
					state = 3
				} else if (tok == iuend) {
					state = 1
				} else if (tok == irstart) {
				  state = 5
				}
				case 3 => if (tok+1 < 0) {
					if (userids != null) {
					  userids(istatus,0) = -(tok+1)
					  userids(istatus,1) = 0
					}
				} else if (tok == iiend) {
					state = 2
				}
				case 4 => if (tok == itend) {
					state = 1
				} else if (tok == iuser) {
					state = 2
				} else if (tok+1 > 0) {
					if (unigramsx != null) {
						unigramsx(nuni, 0) = tok
						unigramsx(nuni, 1) = istatus
						nuni += 1
					}
					if (idata.data(i-1) > 0) {  
						val tok1 = idata.data(i-1)-1
						if (tok1 != itstart) {
							bigramsx(nbi, 0) = tok1
							bigramsx(nbi, 1) = tok
							bigramsx(nbi, 2) = istatus
							nbi += 1
							if (idata.data(i-2) > 0) {
								val tok2 = idata.data(i-2)-1
								if (tok2 != itstart) {
									trigramsx(ntri, 0) = tok2
									trigramsx(ntri, 1) = tok1
									trigramsx(ntri, 2) = tok
									trigramsx(ntri, 3) = istatus
									ntri += 1
								}
							}
						}
					}
				}
				case 5 => if (tok == isstart) {
					state = 6
				} else if (tok == irend) {
					state = 1
				}
				case 6 => if (tok == iuser) {
					state = 7
				} else if (tok == itstart) {
					state = 9
				} 
				case 7 => if (tok == iuend) {
					state = 6
				} else if (tok == iistart) {
					state = 8
				}
				case 8 => if (tok == iiend) {
					state = 7
				} else if (tok-1 < 0) {
					if (userids != null) userids(istatus, 1) = -(tok+1)
				}
				case 9 => if (tok == itend) {
					state = 6
				} else if (tok == iuser) {
				  state = 7
				}
				}
			}
			i += 1
		}
		(nuni, nbi, ntri, istatus)
	}
}