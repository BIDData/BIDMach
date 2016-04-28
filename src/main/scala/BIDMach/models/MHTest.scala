package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._

import java.text.NumberFormat
import edu.berkeley.bid.CUMACH._
import scala.collection.mutable._

class MHTest(var objective:Model, val proposer:Proposer, override val opts:MHTest.Opts = new MHTest.Options) extends Model(opts) {

	parent_model = objective
	// TODO: init the MHTest, estimate the variance of
	// delta here. And load the distribution of X_corr
	// And we should put the parameters in 
	// _modelmats:Arrya[Mat]
	override def init() = {}

	// call proposer to get the theta',
	// then generate a x_corr from distribution of X_corr
	// Then decide whether to replace (i.e. accpet) _modelmats 
	override def dobatch(mats:Array[Mat], ipass:Int, here:Long) = {

	}

	// Call the parent class to compute the loss of the model
	override def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat = {
		// copy back the parameters
		// Notice: this is not the deep copy, we just
		// change the reference of the parent_model
		parent_model._modelmats = modelmats
		parent_model.evalbatch(mats, ipass, here)
	}

	// help methods

	// load the distribution of X_corr with known sigma
	def loadDistribution = {

	}

	// estimate the sd of the delta
	def estimateSd:Double = {
		1.0f
	}


	// compute the value of the delta
	def delta:Double = {
		0.0f
	}
	
}


object MHTest {
	trait  Opts extends Model.Opts {
		// TODO: define the parameters here
	}

	class Options extends Opts {}

	def learner(data:Mat, model:Model, proposer:Proposer) = {
		class xopts extends Learner.Options with MHTest.Opts with MatSource.Opts with IncNorm.Opts 
	    val opts = new xopts
	    // TODO: define the parameters for the opts

	    val nn = new Learner(
	      new MatSource(Array(data:Mat), opts),
	      new MHTest(model, proposer, opts),
	      null,
	      new IncNorm(opts),
	      null,
	      opts)
	    (nn, opts)
	}

	def Ecdf(x: FMat, ecdfmat: FMat, hash:FMat) = {
		val ecdf = new Ecdf(x, ecdfmat, hash)
		ecdf
	}
}

abstract class Proposer() {

	// init the proposer class.
	def init() = {

	}

	// Function to compute the Pr(theta' | theta_t)
	def proposeProb(): Double = {
		1.0f
	}

	// Function to propose the next parameter, i.e. theta'
	def proposeNext():Array[Mat] = {
		null
	}
}

// Class of the emprical cdf of X_corr, there should be three
// matrix to hold the data computed from the matlab
// there are pre-computed txt file at /data/EcdfForMHtest

class Ecdf(val x:FMat, val ecdfmat:FMat, val hash:FMat) {
	var sd:Double = 1.0f
	var ratio:Double = 0.995f
	var f:FMat = null
	
	def init(sd:Double, ratio:Double=0.995) = {
		// looking for the closest index in the hash
		var index:Int = (ratio * hash.ncols.toDouble).toInt;
		if (index >= hash.ncols) {
			index = hash.ncols - 1
		}
		f = ecdfmat(?,index)
	}

	def generateXcorr : Double = {
		var u:Float = rand(1,1)(0,0)
		//println(u)
		binarySearch(u)
	}

	def binarySearch(u:Float) : Double = {
		var start : Int = 0
		var end : Int = f.nrows - 1
		var mid : Int = 0
		while (end > start + 1) {
			mid = (start + end) / 2
			//println("start: " + start + ", end: "+ end + ", mid: " + mid + " val:"+ f(mid))
			if (u < f(mid)) {
				end = mid;
			} else if (u > f(mid)) {
				start = mid;
			} else {
				return x(mid) * sd
			}
		}
		//println("start: " + start + ", end: "+ end)
		(x(start) + x(end))/2 * sd
	}
}

