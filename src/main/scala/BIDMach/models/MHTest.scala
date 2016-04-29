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

	// just for testing
	def Ecdf(x: FMat, ecdfmat: FMat, hash:FMat) = {
		val ecdf = new Ecdf(x, ecdfmat, hash)
		ecdf
	}

	// for testing
	def Langevin_Proposer(init_step:Float, model:Model) = {
		val lp = new Langevin_Proposer(init_step, model)
		lp
	}

	// create a fully connected nn model, just model,
	// not learner
	// TODO: We need to write this function so that it can generate a model, 
	// which we can use to compute the jump prob and loss.
	def constructNNModel(nslabs:Int, width:Int, taper:Float, ntargs:Int, nonlin:Int = 1):Model = {
		null
	}

}

abstract class Proposer() {

	// init the proposer class.
	def init():Unit = {

	}

	// Function to propose the next parameter, i.e. theta' and the delta
	def proposeNext(modelmats:Array[Mat], gmats:Array[Mat], ipass:Int, pos:Long):(Array[Mat], Double) = {
		null
	}
}

class Langevin_Proposer(val init_step:Float, val model:Model) extends Proposer() {
	var step:Float = 1.0f
	var candidate:Array[Mat] = null
	var learning_rate:Float = 0.75f

	override def init():Unit = {
		// init the candidate container
		// the candidate should have the same shape of the model.modelmats
		// here assume the model.modelmats are already initalized
		candidate = new Array[Mat](model.modelmats.length);
		for (i <- 0 until candidate.length) {
			candidate(i) =  model.modelmats(i).zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
		}
	}
	override def proposeNext(modelmats:Array[Mat], gmats:Array[Mat], ipass:Int, pos:Long):(Array[Mat], Double) = {
		// shadow copy the parameter value to the model's mat
		model.setmodelmats(modelmats);

		// compute the gradient
		model.dobatch(gmats, ipass, pos)
		// sample the new model parameters by the gradient and the stepsize
		// and store the sample results into the candidate array
		val stepi = init_step / step ^ learning_rate;
		for (i <- 0 until candidate.length) {
			candidate(i) ~ modelmats(i) - stepi / 2 * model.updatemats(i) + dnormrnd(0, (stepi ^ 0.5)(0,0), modelmats(i).nrows, modelmats(i).ncols)
		}
		step += 1.0f
		
		// compute the delta
		// loss_mat is a (ouput_layer.length, 1) matrix
		var loss_mat_prev = model.evalbatch(gmats, ipass, pos)
		val loss_prev:Float = ln(sum(loss_mat_prev))(0,0)
		// compute the log likelihood of the proposal prob
		// here I ignore the constants

		var loglik_prev_to_new = 0.0
		var loglik_new_to_prev = 0.0
		for (i <- 0 until candidate.length) {
			loglik_prev_to_new += (-1.0*sum(sum((candidate(i) - (modelmats(i) - stepi / 2.0 * model.updatemats(i)))^2)) / 2 / stepi).dv
		}

		// jump from the candidate for one more step
		model.setmodelmats(candidate)
		model.dobatch(gmats, ipass, pos)
		loss_mat_prev = model.evalbatch(gmats, ipass, pos)
		val loss_new:Float = ln(sum(loss_mat_prev))(0,0)

		for (i <- 0 until candidate.length) {
			loglik_new_to_prev += + (-1.0*sum(sum((modelmats(i) - (candidate(i) - stepi / 2.0 * model.updatemats(i)))^2)) / 2 / stepi).dv
		}
		val delta = (-loss_new) - (-loss_prev) + loglik_new_to_prev - loglik_prev_to_new
		(candidate, delta)
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

