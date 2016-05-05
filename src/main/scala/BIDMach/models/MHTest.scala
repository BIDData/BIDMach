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

	// parent_model = objective
	var ecdf:Ecdf = new Ecdf(x_ecdf, ecdfmat, hash_ecdf)
	var delta:Double = 1.0
	// var is_estimte_sd: Boolean = true
	var var_estimate_mat:FMat = zeros(1, 2)
	var sd_smooth_exp_param:Double = 0.7 // use the exp update to estimate var
	var estimated_sd:Double = 1.0
	var num_sd_compute:Int = 0
	var batch_est_sd0:Array[Mat] = null
	var batch_est_sd1:Array[Mat] = null
	// TODO: init the MHTest, estimate the variance of
	// delta here. And load the distribution of X_corr
	// And we should put the parameters in 
	// _modelmats:Arrya[Mat]
	override def init() = {
		// init the ecdf
		// ecdf = new Ecdf(x_ecdf, ecdfmat, hash_ecdf)

		// init the var_estimate_mat container
		// var_estimate_mat = zeros(1, opts.num_iter_estimate_var)
		// is_estimte_sd = true
		// objective.bind(datasource)
		objective.mats = mats
		objective.putBack = datasource.opts.putBack;
	  	objective.useGPU = opts.useGPU && Mat.hasCUDA > 0;
	  	objective.useDouble = opts.useDouble;
		objective.gmats = new Array[Mat](mats.length)




		objective.init()
		_modelmats = new Array[Mat](objective.modelmats.length)
		println("init")
		for (i <- 0 until objective.modelmats.length) {
			_modelmats(i) = objective.modelmats(i).zeros(objective.modelmats(i).nrows, objective.modelmats(i).ncols)
			_modelmats(i) <-- objective.modelmats(i)
			println(_modelmats(i))
		}
		// init the proposer class
		proposer.init()

		// init the batch_est_sd0/1
		var mat = datasource.next
		batch_est_sd0 = new Array[Mat](mat.length)
		for (i <- 0 until mat.length) {
			batch_est_sd0(i) = GMat(mat(i))
		}
		mat = datasource.next;
		batch_est_sd1 = new Array[Mat](mat.length)
		for (i <- 0 until mat.length) {
			batch_est_sd1(i) = GMat(mat(i))
		}

		// init ecdf
		ecdf.init(opts.ratio_decomposite)
	}

	// call proposer to get the theta',
	// then generate a x_corr from distribution of X_corr
	// Then decide whether to replace (i.e. accpet) _modelmats 
	override def dobatch(mats:Array[Mat], ipass:Int, here:Long) = {

		// estimate the variance
		estimated_sd = estimated_sd * sd_smooth_exp_param + (1-sd_smooth_exp_param) * computeVarDelta()
		// println ("estimated_sd : " + estimated_sd)
		proposer.changeToUpdateState()
		val (next_mat:Array[Mat], delta:Double) = proposer.proposeNext(_modelmats, mats, ipass, here)
		// do the test
		ecdf.updateSd(estimated_sd)
		var x_corr = ecdf.generateXcorr
		if (x_corr + delta > 0) {
			// accpet the candiate
			// println("accpet" + " " + delta + "; X_corr: " + x_corr)
			for (i <- 0 until _modelmats.length) {
				// println ("model mats " + _modelmats(i))
				// println("next: " + next_mat(i))
				_modelmats(i) <-- next_mat(i)
				//println ("updated modelmats " + _modelmats(i))
			}
			// println ("updated modelmats " + _modelmats(0))
		}

	}

	// Call the parent class to compute the loss of the model
	override def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat = {
		// copy back the parameters
		// Notice: this is not the deep copy, we just
		// change the reference of the parent_model
		// objective.setmodelmats(_modelmats)
		
		changeObjectiveModelMat(objective, _modelmats)
		// println ("print the eval model")
		for (i <- 0 until modelmats.length) {
			// println ("model mats: " + objective.modelmats(i))
		}
		objective.evalbatch(mats, ipass, here)
		//rand(1,1)
	}

	// help methods


	// change the reference of the modelmats in the model
	// as well as change the reference of modelmats at each layer
	def changeObjectiveModelMat(model:Model, mats:Array[Mat]):Unit = {

		for (i <- 0 until model.modelmats.length) {
			model.modelmats(i) <-- mats(i)
		}
	}

	def computeVarDelta():Double = {
		
		/**
		proposer.changeToEstimateSdState()
		val (next_mat0:Array[Mat], delta0:Double) = proposer.proposeNext(_modelmats, batch_est_sd0, 0, 0)
		val (next_mat1:Array[Mat], delta1:Double) = proposer.proposeNext(_modelmats, batch_est_sd1, 0, 1)
		var_estimate_mat(0) = delta0
		var_estimate_mat(1) = delta1
		proposer.changeToUpdateState()
		(variance(var_estimate_mat)^0.5).dv
		**/
		rand(1,1).dv
	}
}


object MHTest {
	trait  Opts extends Model.Opts {
		// TODO: define the parameters here
		// var num_iter_estimate_var:Int = 100
		// var batchSize:Int = 200 // the parents class already has it
		var ratio_decomposite:Double = 0.994
	}

	class Options extends Opts {}

	def learner(mat0:Mat, targ:Mat, model:Model, proposer:Proposer, x_ecdf: FMat, ecdfmat: FMat, hash_ecdf:FMat) = {
		class xopts extends Learner.Options with MHTest.Opts with MatSource.Opts with IncNorm.Opts 
	    val opts = new xopts

	    val nn = new Learner(
	      new MatSource(Array(mat0, targ), opts),
	      new MHTest(model, proposer, x_ecdf, ecdfmat, hash_ecdf, opts),
	      null,
	      new IncNorm(opts),
	      null,
	      opts)
	    (nn, opts)
	}

	class FDSopts extends Learner.Options with MHTest.Opts with FileSource.Opts
  
	def learner(fn1:String, fn2:String, model:Model, proposer:Proposer, x_ecdf: FMat, ecdfmat: FMat, hash_ecdf:FMat):(Learner, FDSopts) = learner(List(FileSource.simpleEnum(fn1,1,0),
			                                                                  FileSource.simpleEnum(fn2,1,0)), model, proposer, x_ecdf, ecdfmat, hash_ecdf);

	//def learner(fn1:String, x_ecdf: FMat, ecdfmat: FMat, hash_ecdf:FMat):(Learner, FDSopts) = learner(List(FileSource.simpleEnum(fn1,1,0)), x_ecdf, ecdfmat, hash_ecdf);

	def learner(fnames:List[(Int)=>String], model:Model, proposer:Proposer, x_ecdf: FMat, ecdfmat: FMat, hash_ecdf:FMat):(Learner, FDSopts) = {
		
		val opts = new FDSopts;
		opts.fnames = fnames
		opts.batchSize = 200;
		opts.eltsPerSample = 500;
		implicit val threads = threadPool(4);
		val ds = new FileSource(opts)
			val nn = new Learner(
					ds, 
			    new MHTest(model, proposer, x_ecdf, ecdfmat, hash_ecdf, opts), 
			    null,
			    null, 
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
	def Langevin_Proposer(init_step:Float, model:Model):Proposer = {
		val lp = new Langevin_Proposer(init_step, model)
		lp
	}

	def Gradient_descent_proposer(init_step:Float, model:Model):Proposer = {
		val lp = new Gradient_descent_proposer(init_step, model)
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
		opts.nweight = 1e-4f
		val net = Net.dnodes3(nslabs, width, taper, ntargs, opts, nonlin);
		opts.nodeset = net
		// opts.debug = 1
		val model = new Net(opts)
		model
	}

}

abstract class Proposer() {

	// init the proposer class.
	def init():Unit = {

	}

	def changeToUpdateState():Unit = {}

	def changeToEstimateSdState():Unit = {}

	// Function to propose the next parameter, i.e. theta' and the delta
	def proposeNext(modelmats:Array[Mat], gmats:Array[Mat], ipass:Int, pos:Long):(Array[Mat], Double) = {
		null
	}
}

class Langevin_Proposer(val init_step:Float, val model:Model) extends Proposer() {
	var step:Float = 1.0f
	var candidate:Array[Mat] = null
	var learning_rate:Float = 0.75f
	var random_matrix:Array[Mat] = null
	// var random_matrixFMat:Array[FMat] = null
	var stepi:Mat = null
	var is_estimte_sd = true
	override def init():Unit = {
		// init the candidate container
		// the candidate should have the same shape of the model.modelmats
		// here assume the model.modelmats are already initalized
		candidate = new Array[Mat](model.modelmats.length)
		random_matrix = new Array[Mat](model.modelmats.length)
		// random_matrixFMat = new Array[FMat](model.modelmats.length)
		stepi = model.modelmats(0).zeros(1,1)
		for (i <- 0 until candidate.length) {
			candidate(i) =  model.modelmats(i).zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
			random_matrix(i) = model.modelmats(i).zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
			// random_matrixFMat(i) = zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
		}
	}

	override def changeToUpdateState():Unit = {
		is_estimte_sd = false
	}

	override def changeToEstimateSdState():Unit = {
		is_estimte_sd = true
	}

	override def proposeNext(modelmats:Array[Mat], gmats:Array[Mat], ipass:Int, pos:Long):(Array[Mat], Double) = {
		// deep copy the parameter value to the model's mat
		for (i <- 0 until modelmats.length) {
			model.modelmats(i) <-- modelmats(i)
		}
		// model.setmodelmats(modelmats);

		// compute the gradient
		model.dobatch(gmats, ipass, pos)
		
		// sample the new model parameters by the gradient and the stepsize
		// and store the sample results into the candidate array
		stepi <-- init_step / step ^ learning_rate;
		for (i <- 0 until candidate.length) {
			// println(candidate(i))
			// println(modelmats(i))
			// random_matrixFMat(i) <-- dnormrnd(0, (stepi ^ 0.5).dv, modelmats(i).nrows, modelmats(i).ncols)
			// random_matrix(i) <-- random_matrixFMat(i)
			normrnd(0, (stepi ^ 0.5).dv, random_matrix(i))
			candidate(i) <-- modelmats(i) + stepi / 2.0 * model.updatemats(i) 
			candidate(i) <-- candidate(i) + random_matrix(i)
			// println(" rand matrix")
			// println(random_matrix(i))
			// println("candiate")
			// println(candidate(i))
		}
		if (!is_estimte_sd) {
			step += 1.0f
		}
		
		// compute the delta
		// loss_mat is a (ouput_layer.length, 1) matrix
		var loss_mat_prev = model.evalbatch(gmats, ipass, pos)
		// println (loss_mat_prev)
		val loss_prev:Float = (sum(loss_mat_prev))(0,0)
		// compute the log likelihood of the proposal prob
		// here I ignore the constants

		var loglik_prev_to_new = 0.0
		var loglik_new_to_prev = 0.0
		for (i <- 0 until candidate.length) {
			// println ((candidate(i) - (modelmats(i) - stepi / 2.0 * model.updatemats(i))))
			loglik_prev_to_new += (-1.0*sum(sum((abs(candidate(i) - (modelmats(i) + stepi / 2.0 * model.updatemats(i))))^2)) / 2 / stepi).dv
		}

		// jump from the candidate for one more step
		// model.setmodelmats(candidate)
		for (i <- 0 until candidate.length) {
			model.modelmats(i) <-- candidate(i)
		}

		model.dobatch(gmats, ipass, pos)
		loss_mat_prev = model.evalbatch(gmats, ipass, pos)
		val loss_new:Float = (sum(loss_mat_prev))(0,0)
		for (i <- 0 until candidate.length) {
			// println("the model update " + ((modelmats(i) - (candidate(i) - stepi / 2.0 * model.updatemats(i)))^2))
			loglik_new_to_prev +=  (-1.0*sum(sum((abs(modelmats(i) - (candidate(i) + stepi / 2.0 * model.updatemats(i))))^2)) / 2 / stepi).dv
		}
		val delta = (-loss_new) - (-loss_prev) + loglik_new_to_prev - loglik_prev_to_new
		// println("new loss: " + loss_new + "; prev loss " + loss_prev+ "; loglike new " + loglik_new_to_prev +"; log old" + loglik_prev_to_new)
		(candidate, delta)
	}


}


class Gradient_descent_proposer (val init_step:Float, val model:Model) extends Proposer() {
	var step:Float = 1.0f
	var candidate:Array[Mat] = null
	var learning_rate:Float = 0.4f
	var stepi:Mat = null
	var is_estimte_sd = true
	var mu:Float = 0.0f
	var moument:Array[Mat] = null

	override def init():Unit = {
		// init the container here
		candidate = new Array[Mat](model.modelmats.length)
		moument = new Array[Mat](model.modelmats.length)	
		stepi = model.modelmats(0).zeros(1,1)
		for (i <- 0 until candidate.length) {
			candidate(i) =  model.modelmats(i).zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
			moument(i) = model.modelmats(i).zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
		}
	}

	override def proposeNext(modelmats:Array[Mat], gmats:Array[Mat], ipass:Int, pos:Long):(Array[Mat], Double) = {
		// just do the one step gradient descent
		if (!is_estimte_sd) {
			// model.setmodelmats(modelmats);
			for (i <- 0 until modelmats.length) {
				model.modelmats(i) <-- modelmats(i)
			}
			// compute the gradient
			model.dobatch(gmats, ipass, pos)
			// println ("the input data is " + gmats(0))
			// println ("ipass " + ipass + ", pos: " + pos)
			
			// println ("the model modelmats in proposer " + model.modelmats(0))
			// sample the new model parameters by the gradient and the stepsize
			// and store the sample results into the candidate array
			stepi <-- init_step / (step ^ learning_rate);
			for (i <- 0 until candidate.length) {
				// println("proposer next " + stepi )
				// println("the updates " + model.updatemats(i))
				// println("Update mats:", mean(mean(snorm(model.updatemats(i)))))
				// println("Model mats:", mean(mean(snorm(model.modelmats(i)))))
				// println("model mats: ", model.modelmats(i))
				moument(i) <-- moument(i) * mu + stepi * model.updatemats(i)
				candidate(i) <-- modelmats(i) + moument(i) 
				// println("the candidada " + candidate(i))
			}
			step += 1.0f
			// println()
			// println("the step is " + step + " init step is " + init_step + "learning_rate is " + learning_rate + ", step size is " + stepi)
			// println()
		}
		// for delta, we just return a very large value
		(candidate, 1000000.0)
	}

	override def changeToUpdateState():Unit = {
		is_estimte_sd = false
	}

	override def changeToEstimateSdState():Unit = {
		is_estimte_sd = true
	}
}

// Class of the emprical cdf of X_corr, there should be three
// matrix to hold the data computed from the matlab
// there are pre-computed txt file at /data/EcdfForMHtest

class Ecdf(val x:FMat, val ecdfmat:FMat, val hash:FMat) {
	var sd:Double = 1.0
	var ratio:Double = 0.995f
	var f:FMat = null
	
	def init(ratio:Double=0.995) = {
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

	def updateSd (inputsd:Double):Unit = {
		sd = inputsd
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
		// println("inside ecdf, " + ((x(start) + x(end))/2) + " - " + sd)
		(x(start) + x(end))/2 * sd
	}
}

