package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._
import BIDMach.networks._

import java.text.NumberFormat
import edu.berkeley.bid.CUMACH._
import scala.collection.mutable._

class MHTest(var objective:Model, val proposer:Proposer, val x_ecdf: FMat, val ecdfmat: FMat, val hash_ecdf:FMat, 
	override val opts:MHTest.Opts = new MHTest.Options) extends Model(opts) {

	parent_model = objective
	var ecdf:Ecdf = new Ecdf(x_ecdf, ecdfmat, hash_ecdf)
	var delta:Double = 1.0
	var is_estimte_sd: Boolean = true
	var var_estimate_mat:FMat = zeros(1, opts.num_iter_estimate_var)
	var estimated_sd:Double = 1.0
	// TODO: init the MHTest, estimate the variance of
	// delta here. And load the distribution of X_corr
	// And we should put the parameters in 
	// _modelmats:Arrya[Mat]
	override def init() = {
		// init the ecdf
		// ecdf = new Ecdf(x_ecdf, ecdfmat, hash_ecdf)

		// init the var_estimate_mat container
		// var_estimate_mat = zeros(1, opts.num_iter_estimate_var)
		is_estimte_sd = true

		// init the proposer class
		proposer.init()

		// TODO: check whether we need to init the modelmats here
	}

	// call proposer to get the theta',
	// then generate a x_corr from distribution of X_corr
	// Then decide whether to replace (i.e. accpet) _modelmats 
	override def dobatch(mats:Array[Mat], ipass:Int, here:Long) = {
		if (ipass == opts.num_iter_estimate_var) {
			// compute the var for the delta
			estimated_sd = (variance(var_estimate_mat)^0.5).dv
			// init the ecdf
			ecdf.init(estimated_sd, opts.ratio_decomposite)
			is_estimte_sd = false
		}
		val (next_mat:Array[Mat], delta:Double) = proposer.proposeNext(modelmats, mats, ipass, here)
		if (is_estimte_sd) {
			var_estimate_mat(0,ipass) = delta
		} else{
			// do the test
			var x_corr = ecdf.generateXcorr
			if (x_corr + delta > 0) {
				// accpet the candiate
				for (i <- 0 until modelmats.length) {
					modelmats(i) ~ next_mat(i)
				}
			}
		}

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

	
}


object MHTest {
	trait  Opts extends Model.Opts {
		// TODO: define the parameters here
		var num_iter_estimate_var:Int = 1000
		// var batchSize:Int = 200 // the parents class already has it
		var ratio_decomposite:Double = 0.994
	}

	class Options extends Opts {}

	def learner(data:Mat, model:Model, proposer:Proposer, x_ecdf: FMat, ecdfmat: FMat, hash_ecdf:FMat) = {
		class xopts extends Learner.Options with MHTest.Opts with MatSource.Opts with IncNorm.Opts 
	    val opts = new xopts
	    // TODO: define the parameters for the opts

	    val nn = new Learner(
	      new MatSource(Array(data:Mat), opts),
	      new MHTest(model, proposer, x_ecdf, ecdfmat, hash_ecdf, opts),
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
		val opts = new Net.LearnOptions
		if (opts.links == null) {
     		opts.links = izeros(1,1);
      		opts.links.set(1);
    	}
		// opts.nend = 10
		opts.npasses = 50
		opts.batchSize = 200
		opts.reg1weight = 0.0001;
		opts.hasBias = true;
		opts.links = iones(1,1);
		opts.lrate = 0.01f;
		opts.texp = 0.4f;
		opts.evalStep = 311;
		opts.nweight = 1e-4f
		val net = Net.dnodes3(nslabs, width, taper, ntargs, opts, nonlin);
		opts.nodeset = net

		val model = new Net(opts)
		model
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

