package BIDMach
import BIDMat.{Mat,BMat,CMat,CSMat,Dict,DMat,FMat,GMat,GIMat,GSMat,HMat,IDict,IMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import HMat._
import scala.actors._
import java.io._

class Featurizer(val opts:Featurizer.Options = new Featurizer.Options) {
	def countGrams = {
    val alldict = Dict(loadBMat(opts.mainDict))
    val isstart = alldict(opts.startItem)
    val isend = alldict(opts.endItem)
    val itstart = alldict(opts.startText)
    val itend = alldict(opts.endText)
      
    for (ithread <- 0 until math.max(1, Mat.hasCUDA)) {
      Actor.actor {
        setGPU(ithread)
      	val bigramsx = IMat(opts.guessSize, 2)
      	val trigramsx = IMat(opts.guessSize, 3)
      	val bdicts = new Array[IDict](24)
      	val tdicts = new Array[IDict](24)

      	for (idir <- (opts.nstart+ithread) until opts.nend by opts.nthreads) {
      		val fname = opts.fromDir(idir)+opts.localDict
      		if (fileExists(fname)) {
      			val dict = Dict(loadBMat(fname))
      			val dmap = dict --> alldict
      			for (ifile <- 0 until 24) { 
      				val fn = opts.fromDir(idir)+opts.fromFile(ifile)
      				if (fileExists(fn)) {
      					val idata = loadIMat(fn)
      					var active = false
      					var intext = false
      					var i = 0
      					var istatus = -1
      					var nbi = 0
      					var ntri = 0
      					var len = idata.length
      					while (i < len) {
      						if (idata.data(i) > 0) {
      							val tok = dmap(idata.data(i)-1)
      							if (tok == isstart) {
      								active = true
      								istatus += 1
      							} else if (tok == itstart && active) {     							  
      								intext = true
      							} else if (tok == itend) {
      								intext = false
      							} else if (tok == isend) {
      								intext = false
      								active = false
      							} else {
      								if (intext && idata.data(i-1) > 0) {      			
      									val tok1 = dmap(idata.data(i-1)-1)
      									if (tok1 != itstart) {
      										bigramsx(nbi, 0) = tok1
      										bigramsx(nbi, 1) = tok
      										nbi += 1
      										if (idata.data(i-2) > 0) {
      											val tok2 = dmap(idata.data(i-2)-1)
      											if (tok2 != itstart) {
      												trigramsx(nbi, 0) = tok2
      												trigramsx(nbi, 1) = tok1
      												trigramsx(nbi, 2) = tok
      												ntri += 1
      											}
      										}
      									}
      								}
      							}
      						}
      						i += 1
      					}
      					val bigrams = IMat(nbi, 2, bigramsx.data)
      					val bid = IDict.dictFromData(bigrams)
      					val trigrams = IMat(nbi, 3, trigramsx.data)
      					val trid = IDict.dictFromData(trigrams)
      					bdicts(ifile) = bid
      					tdicts(ifile) = trid      		
      				} else {
      					bdicts(ifile) = null
      					tdicts(ifile) = null
      				}
      			}
      			val bf = IDict.union(bdicts)
      			val tf = IDict.union(tdicts)
      			saveIMat(opts.fromDir(idir) + "bdict.lz4", bf.grams)
      			saveDMat(opts.fromDir(idir) + "bcnts.lz4", bf.counts)
      			saveIMat(opts.fromDir(idir) + "tdict.lz4", tf.grams)
      			saveDMat(opts.fromDir(idir) + "tcnts.lz4", tf.counts)
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
  
  def dirMap(fname:String):(Int)=>String = {
    (n:Int) => {    
    	val (yy, mm, dd) = decodeDate(n)
    	(fname format (n % 16, yy, mm, dd))
    }    
  }
  
  class Options {
    var fromDir:(Int)=>String = dirMap("/disk%02d/twitter/tokenized/%04d/%02d/%02d/")
    var toDir:(Int)=>String = dirMap("/disk%02d/twitter/featureized/%04d/%02d/%02d/") 
    var fromFile:(Int)=>String = (n:Int) => ("tweet%02d.gz" format n)
    var toFile:(Int)=>String = (n:Int) => ("tweet%02d.txt" format n)
    var localDict:String = "dict.gz"
    var nstart:Int = encodeDate(2011,11,21)
    var nend:Int = encodeDate(2013,3,3)
    var startItem:String = "<status>"
    var endItem:String = "</status>"
    var startText:String = "<text>"
    var endText:String = "</text>"
    var mainDict:String = "/big/twitter/tokenized/alldict.gz"
    var mainCounts:String = "/big/twitter/tokenized/allwcount.gz"
    var guessSize = 200000000
    var nthreads = 4
  }
  
 
}