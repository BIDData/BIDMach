package BIDMach
import BIDMat.{Mat,BMat,CMat,CSMat,Dict,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import scala.actors._
import java.io._


object Twitter { 
  
   def dodicts(threshold:Int=10, rebuild:Boolean=false):Unit = {
  		 val stokdir = "/twitter/smiley/tokenized/"
    	 val tokdir = "/twitter/tokenized/"
  		 val dy1 = mergedicts(2011, 2013, "/disk%02d" + stokdir, "/big" + stokdir, threshold, rebuild)
  		 val dy2 = mergedicts(2011, 2013, "/disk%02d" + tokdir, "/big" + tokdir, threshold, rebuild)
  		 val dy = Dict.union(dy1, dy2)
  		 val (sv, iv) = sortdown2(dy.counts)
  		 HMat.saveBMat("/big"+tokdir+"alldict.gz", BMat(dy.cstr(iv)))
  		 HMat.saveDMat("/big"+tokdir+"allwcount.gz", sv)
	}
  
	def mergedicts(year1:Int, year2:Int, infname:String, outfname:String, threshold:Int=10, rebuild:Boolean=false):Dict = {
  	val dd = new Array[Dict](6)
  	val md = new Array[Dict](6)
  	val yd = new Array[Dict](5)
  	var dy:Dict = null
  	var nmerged = 0
	  for (yy <- year1 to year2) {
	  	for (mm <- 1 to 12) {
	  		print("\n%d/%02d" format (yy, mm))
	  		val ff = new File(outfname + "%04d/%02d/wcount.gz" format (yy, mm))
	  		if (rebuild || ! ff.exists) {
	  			var ndone = 0
	  			for (id <- 1 to 31) {
	  				var ielem = 372*yy + 31*mm + id
	  				var idisk = ielem % 16
	  				val fname = (infname + "%04d/%02d/%02d/" format (idisk, yy, mm, id))
	  				val ff = new File(fname + "wcount.gz")
	  				if (ff.exists) {
	  					val bb = HMat.loadBMat(fname + "dict.gz")
	  					val cc = HMat.loadIMat(fname + "wcount.gz")
	  					dd(ndone % 6) = Dict(bb, cc, threshold)
	  					ndone = ndone + 1
	  					print("-")
	  					if (ndone % 6 == 0) {
	  						md(ndone / 6 - 1) = Dict.union(dd:_*)
	  						print("+")
	  					}
	  				}
	  			}
	  			if (ndone % 6 != 0) {
	  				md(ndone / 6) = Dict.union(dd.slice(0, ndone % 6):_*)
	  				print("+")
	  			}
	  			if (ndone > 0) {
	  				val dx = Dict.union(md.slice(0, (ndone-1)/6+1):_*)
	  				val (sv, iv) = sortdown2(dx.counts)
	  				val dxx = Dict(dx.cstr(iv), sv)
	  				HMat.saveBMat(outfname + "%04d/%02d/dict.gz" format (yy, mm), BMat(dxx.cstr))
	  				HMat.saveDMat(outfname + "%04d/%02d/wcount.gz" format (yy, mm), dxx.counts)
	  			}
//	  			println("")
	  		}
	  		val f2 = new File(outfname + "%04d/%02d/wcount.gz" format (yy, mm))
	  		if (f2.exists) {
	  			val bb = HMat.loadBMat(outfname + "%04d/%02d/dict.gz" format (yy, mm))
	  			val cc = HMat.loadDMat(outfname + "%04d/%02d/wcount.gz" format (yy, mm))
	  			yd(nmerged % 5) = Dict(bb, cc, 4*threshold)
	  			nmerged += 1
	  			print("*")
	  			if (nmerged % 5 == 0) {
	  			  val dm = Dict.union(yd:_*)
	  			  if (nmerged == 5) {
	  			    dy = dm
	  			  } else {
	  			  	dy = Dict.union(dy, dm)
	  			  }
	  			}
	  		}
	  	}
	  }
  	if (nmerged % 5 != 0) {
  		val dm = Dict.union(yd.slice(0, nmerged % 5):_*)
  		dy = Dict.union(dy, dm)
  	}
  	println
  	val (sv, iv) = sortdown2(dy.counts)
  	val dyy = Dict(dy.cstr(iv), sv)
  	HMat.saveBMat(outfname + "dict.gz", BMat(dyy.cstr))
  	HMat.saveDMat(outfname + "wcount.gz", dyy.counts)
  	dyy
	}
}