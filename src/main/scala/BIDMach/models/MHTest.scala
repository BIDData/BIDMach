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

class MHTest(var objective:Model, val proposer:Proposer, val ecdfmat: FMat, val hash_ecdf:FMat, 
	override val opts:MHTest.Opts = new MHTest.Options) extends Model(opts) {

	var ecdf:Ecdf = new Ecdf(ecdfmat, hash_ecdf)
	var delta:Double = 1.0
	var var_estimate_mat:FMat = null
	var sd_smooth_exp_param:Double = 0.7 // use the exp update to estimate var
	var estimated_sd:Double = 1.0
	var accpet_count:Float = 0.0f
	var reject_count:Float = 0.0f
	var batch_est_data:Array[Array[Mat]] = null
	var help_mats:Array[Mat] = null
	var data_buffer:Array[Mat] = null	// the array to hold the previous data batch

	override def init() = {
		// init the ecdf
		
		objective.mats = mats
		objective.putBack = datasource.opts.putBack;
	  	objective.useGPU = opts.useGPU && Mat.hasCUDA > 0;
	  	objective.useDouble = opts.useDouble;
		objective.gmats = new Array[Mat](mats.length)

		objective.init()
		_modelmats = new Array[Mat](objective.modelmats.length)
		println("init")
		// init the proposer class
		proposer.init()

		if (proposer.has_help_mats) {
			help_mats = new Array[Mat](objective.modelmats.length)
		}
		
		for (i <- 0 until objective.modelmats.length) {
			_modelmats(i) = objective.modelmats(i).zeros(objective.modelmats(i).nrows, objective.modelmats(i).ncols)
			_modelmats(i) <-- objective.modelmats(i)
			if (proposer.has_help_mats) {
				help_mats(i) = objective.modelmats(i).zeros(objective.modelmats(i).nrows, objective.modelmats(i).ncols)
			}
			println(_modelmats(i))
		}
		

		// init the batch_est_sd0/1
		var mat = datasource.next
		// put the mat into the data buffer
		data_buffer = new Array[Mat](mat.length)
		for (i <- 0 until mat.length) {
			data_buffer(i) = GMat(mat(i).zeros(mat(i).nrows, mat(i).ncols))
			data_buffer(i) <-- mat(i)
		}

		// init the container
		var_estimate_mat = zeros(1, opts.num_data_est_sd)

		batch_est_data = Array.ofDim[Mat](opts.num_data_est_sd, mat.length)
		for (i_batch <- 0 until opts.num_data_est_sd) {
			mat = datasource.next
			for (i_mat <- 0 until mat.length) {
				batch_est_data(i_batch)(i_mat) = GMat(mat(i_mat))
			}
		}

		// init ecdf
		ecdf.init()
	}

	// call proposer to get the theta',
	// then generate a x_corr from distribution of X_corr
	// Then decide whether to replace (i.e. accpet) _modelmats 
	override def dobatch(mats:Array[Mat], ipass:Int, here:Long) = {

		// estimate the variance
		estimated_sd = estimated_sd * sd_smooth_exp_param + (1-sd_smooth_exp_param) * computeVarDelta()
		if (java.lang.Double.isNaN(estimated_sd)) {
			throw new RuntimeException("NaN for the sd 3 ")
		}
		if (here == 0) {
			accpet_count = 0.0f
			reject_count = 0.0f
		}
		proposer.changeToUpdateState()
		// propose the data
		val (next_mat:Array[Mat], update_v, delta:Double) = proposer.proposeNext(_modelmats, help_mats, mats, ipass, here)
		
		// compute the delta by another batch
		val delta_new = proposer.computeDelta(next_mat, _modelmats, update_v, help_mats, data_buffer, 0, 0)

		// update the data buffer

		for (i <- 0 until mats.length) {
			data_buffer(i) <-- mats(i)
		}

		// do the test
		// println ("the delta is " + delta)
		if (opts.is_always_accpet) {
			// always accept
			for (i <- 0 until _modelmats.length) {
				// println ("model mats " + _modelmats(i))
				// println("next: " + next_mat(i))
				if (proposer.has_help_mats) {
					help_mats(i) <-- (update_v.asInstanceOf[Array[Mat]])(i)
				}
				_modelmats(i) <-- next_mat(i)
			}
			changeObjectiveModelMat(objective, _modelmats)
			accpet_count += 1.0f
		} else {
			if (estimated_sd < 1.2f) {
				ecdf.updateSd(estimated_sd)
				var x_corr = ecdf.generateXcorr
				if (x_corr + delta_new > 0) {
					// accpet the candiate
					// println("accpet" + " " + delta + "; X_corr: " + x_corr)
					for (i <- 0 until _modelmats.length) {
						// println ("model mats " + _modelmats(i))
						// println("next: " + next_mat(i))
						if (proposer.has_help_mats) {
							help_mats(i) <-- (update_v.asInstanceOf[Array[Mat]])(i)
						}
						_modelmats(i) <-- next_mat(i)
					}
					changeObjectiveModelMat(objective, _modelmats)
					accpet_count += 1.0f
					//println ("updated modelmats " + objective.modelmats(0))
				} else {
					reject_count += 1.0f
				}
			} else {
				println ("skip the large var " + estimated_sd)
				reject_count += 1.0f
			}
		}
		
		

	}

	// Call the parent class to compute the loss of the model
	override def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat = {
		// copy back the parameters
		// Notice: this is not the deep copy, we just
		// change the reference of the parent_model
		// objective.setmodelmats(_modelmats)
		
		changeObjectiveModelMat(objective, _modelmats)
		var accpe_ratio = accpet_count / (accpet_count + reject_count)
		if (java.lang.Double.isNaN(estimated_sd)) {
			throw new RuntimeException("ADA0 2 ")

		}
		val loss = objective.evalbatch(mats, ipass, here)
		println ("REST the sd of delat sdDelta: " + estimated_sd + " accpet ratio is AccRate: " + accpe_ratio + " the loss: " + loss)
		loss
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
		
		
		proposer.changeToEstimateSdState()

		for (i <- 0 until opts.num_data_est_sd) {
			
			var (next_mat0, update_v, delta) = proposer.proposeNext(_modelmats, help_mats, batch_est_data(i), 0, 0)
			var_estimate_mat(0,i) = delta
		}
		proposer.changeToUpdateState()
		var varianceVal = variance(var_estimate_mat)
		// println("the var is "+ varianceVal + ",  the vect is " + var_estimate_mat)
		if (varianceVal.dv < 0) {
			varianceVal(0,0) = 1e-5f
		}
		(varianceVal^0.5).dv
	
	}
}


object MHTest {
	trait  Opts extends Model.Opts {
		// TODO: define the parameters here
		// var num_iter_estimate_var:Int = 100
		// var batchSize:Int = 200 // the parents class already has it
		var ratio_decomposite:Double = 0.994
		var num_data_est_sd:Int = 3
		var is_always_accpet:Boolean = false
	}

	class Options extends Opts {}

	def learner(mat0:Mat, targ:Mat, model:Model, proposer:Proposer, ecdfmat: FMat, hash_ecdf:FMat) = {
		class xopts extends Learner.Options with MHTest.Opts with MatSource.Opts with IncNorm.Opts 
	    val opts = new xopts

	    val nn = new Learner(
	      new MatSource(Array(mat0, targ), opts),
	      new MHTest(model, proposer, ecdfmat, hash_ecdf, opts),
	      null,
	      new IncNorm(opts),
	      null,
	      opts)
	    (nn, opts)
	}

	class FDSopts extends Learner.Options with MHTest.Opts with FileSource.Opts
  
	def learner(fn1:String, fn2:String, model:Model, proposer:Proposer, ecdfmat: FMat, hash_ecdf:FMat):(Learner, FDSopts) = learner(List(FileSource.simpleEnum(fn1,1,0),
			                                                                  FileSource.simpleEnum(fn2,1,0)), model, proposer, ecdfmat, hash_ecdf);


	def learner(fnames:List[(Int)=>String], model:Model, proposer:Proposer, ecdfmat: FMat, hash_ecdf:FMat):(Learner, FDSopts) = {
		
		val opts = new FDSopts;
		opts.fnames = fnames
		opts.batchSize = 200;
		opts.eltsPerSample = 500;
		implicit val threads = threadPool(4);
		val ds = new FileSource(opts)
			val nn = new Learner(
					ds, 
			    new MHTest(model, proposer, ecdfmat, hash_ecdf, opts), 
			    null,
			    null, 
			    null,
			    opts)
		(nn, opts)
	} 

	// just for testing
	def Ecdf(ecdfmat: FMat, hash:FMat) = {
		val ecdf = new Ecdf(ecdfmat, hash)
		ecdf
	}

	// for testing
	def Langevin_Proposer(lr:Float, t:Float, v:Float, cp:Float, model:Model):Proposer = {
		val lp = new Langevin_Proposer(lr, t, v, cp, model)
		lp
	}

	def Gradient_descent_proposer(lr:Float, u:Float, t:Float, v:Float, cp:Float, model:Model):Proposer = {

		val lp = new Gradient_descent_proposer(lr, u, t, v, cp, model)
		lp
	}

	def SGHMC_proposer (lr:Float, a:Float, t:Float, v:Float, cp:Float, k:Float, batchSize:Float, model:Model):Proposer = {
		val lp = new SGHMC_proposer(lr, a, t, v, cp, k, batchSize, model)
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
		// opts.lookahead = 0   /// turn off prefetch
		// opts.debug = 1
		val model = new Net(opts)
		model
	}

}

abstract class Proposer() {
	// init the proposer class.
	var has_help_mats:Boolean
	def init():Unit = {

	}

	def changeToUpdateState():Unit = {}

	def changeToEstimateSdState():Unit = {}

	// Function to propose the next parameter, i.e. theta' and the delta
	def proposeNext(modelmats:Array[Mat], prev_v:Array[Mat], gmats:Array[Mat], ipass:Int, pos:Long):(Array[Mat], Array[Mat], Double) = {
		null
	}

	def computeDelta(mats_new:Array[Mat], mats_old:Array[Mat], new_v:Array[Mat], prev_v:Array[Mat], gmats:Array[Mat], ipass:Int, pos:Long): Double ={
		-1.0
	}
}

class Langevin_Proposer(val lr:Float, val t:Float, val v:Float, val cp:Float, val model:Model) extends Proposer() {

	var step:Mat = null	// record the step by itself
	var candidate:Array[Mat] = null
	var stepi:Mat = null
	var is_estimte_sd = true
	var sumSq:Array[Mat] = null // container for g*g
	var lrate:Mat = null
	var te:Mat = null
	var ve:Mat = null
	var updatemats:Array[Mat] = null // just a reference
	var epsilon:Float = 1e-5f
    var initsumsq = 1e-5f
	var clipByValue:Mat = null
	var newsquares:Array[Mat] = null
	var random_matrix:Array[Mat] = null
	var sumSq_tmp_container:Array[Mat] = null
	override var has_help_mats:Boolean = false

	override def init():Unit = {

		candidate = new Array[Mat](model.modelmats.length)
		sumSq = new Array[Mat](model.modelmats.length)
		sumSq_tmp_container = new Array[Mat](model.modelmats.length)
		newsquares = new Array[Mat](model.modelmats.length)
		random_matrix = new Array[Mat](model.modelmats.length)

		stepi = model.modelmats(0).zeros(1,1)
		step = model.modelmats(0).ones(1,1)
		
		te = model.modelmats(0).zeros(1,1)
		te(0,0) = t
		ve = model.modelmats(0).zeros(1,1)
		ve(0,0) = v
		lrate = model.modelmats(0).zeros(1,1)
		lrate(0,0) = lr
		
		if (cp > 0) {
			clipByValue = model.modelmats(0).zeros(1,1)
			clipByValue(0,0) = cp
		}
		for (i <- 0 until candidate.length) {
			candidate(i) =  model.modelmats(i).zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
			sumSq(i) = model.modelmats(i).ones(model.modelmats(i).nrows, model.modelmats(i).ncols) *@ initsumsq
			sumSq_tmp_container(i) = model.modelmats(i).ones(model.modelmats(i).nrows, model.modelmats(i).ncols) *@ initsumsq
			newsquares(i) = model.modelmats(i).zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
			random_matrix(i) = model.modelmats(i).zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
		}
		println("finish init the proposer")
		println("step: " + step + ", stepi" + stepi + ", te: " + te + ", ve: " + ve +", lrate: " + lrate)

	}

	override def changeToUpdateState():Unit = {
		is_estimte_sd = false
	}

	override def changeToEstimateSdState():Unit = {
		is_estimte_sd = true
	}

	override def proposeNext(modelmats:Array[Mat], prev_v:Array[Mat], gmats:Array[Mat], ipass:Int, pos:Long):(Array[Mat], Array[Mat], Double) = {
		// deep copy the parameter value to the model's mat
		for (i <- 0 until modelmats.length) {
			model.modelmats(i) <-- modelmats(i)
		}

		// compute the gradient
		model.dobatch(gmats, ipass, pos)
		
		updatemats = model.updatemats

		// sample the new model parameters by the gradient and the stepsize
		// and store the sample results into the candidate array
		stepi <-- lrate / (step ^ te) / 2.0f

		// adagrad to revise the grad
		for (i <- 0 until candidate.length) {
			// clip
			if (cp > 0f) {
            	min(updatemats(i), clipByValue,updatemats(i));
          		max(updatemats(i),-clipByValue,updatemats(i));
    		}

			// compute the ss
			val ss = sumSq(i)
			val um = updatemats(i)
			newsquares(i) <-- um *@ um
			
			sumSq_tmp_container(i) <-- ss // copy to tmp container

			ss ~ ss *@ (step - 1)
			ss ~ ss + newsquares(i)
			ss ~ ss / step
			val grad = ss ^ ve

			grad ~ grad + epsilon
			grad ~ um / grad
			grad ~ grad *@ stepi

			// for add the gassian noisy
			normrnd(0, ((stepi*2) ^ 0.5).dv, random_matrix(i))
			grad ~ grad + random_matrix(i)
			
			candidate(i) <-- modelmats(i) + grad
			if (java.lang.Double.isNaN(sum(sum(candidate(i))).dv)) throw new RuntimeException("candidate"+i);
		}
		

		// compute the delta

		val delta = computeDelta(candidate, modelmats, null, null, gmats, ipass, pos)

		// update the iteration only if it's update
		if (!is_estimte_sd) {
			step ~ step + 1.0f
		}
		// println ("delta:" + delta + " loss_new:" + loss_new + " loss_prev:" + loss_prev + " loglik_new_to_prev:" + loglik_new_to_prev + " loglik_prev_to_new:" + loglik_prev_to_new)

		if (java.lang.Double.isNaN(delta)) {
			// println ("delta:" + delta + " loss_new:" + loss_new + " loss_prev:" + loss_prev + " loglik_new_to_prev:" + loglik_new_to_prev + " loglik_prev_to_new:" + loglik_prev_to_new)
			throw new RuntimeException("Delta")
		}

		(candidate, null, delta)
	}


	override def computeDelta(mats_new:Array[Mat], mats_old:Array[Mat], new_v:Array[Mat], prev_v:Array[Mat], gmats:Array[Mat], ipass:Int, pos:Long): Double ={
		// copy the mats_old to the model
		for (i <- 0 until mats_old.length) {
			model.modelmats(i) <-- mats_old(i)
		}

		// compute the loss
		var loss_mat_prev = model.evalbatch(gmats, ipass, pos)
		val loss_prev = (sum(loss_mat_prev)).dv

		// compute the gradient and rescale it
		model.dobatch(gmats, ipass, pos)
		
		updatemats = model.updatemats

		// sample the new model parameters by the gradient and the stepsize
		// and store the sample results into the candidate array

		var loglik_prev_to_new = 0.0
		var loglik_new_to_prev = 0.0

		// adagrad to revise the grad
		for (i <- 0 until updatemats.length) {
			// clip
			if (cp > 0f) {
            	min(updatemats(i), clipByValue,updatemats(i));
          		max(updatemats(i),-clipByValue,updatemats(i));
    		}

			// compute the ss
			val ss2 = sumSq_tmp_container(i)
			val um2 = updatemats(i)
			newsquares(i) <-- um2 *@ um2 	// it's OK to reuse the newsquares

			ss2 ~ ss2 *@ (step - 1)
			ss2 ~ ss2 + newsquares(i)
			ss2 ~ ss2 / step
			val grad2 = ss2 ^ ve

			// de-affect of the ss2
			ss2 <-- ss2 *@ step
			ss2 <-- ss2 - newsquares(i)
			if (step.dv > 1) {
				ss2 <-- ss2 / (step - 1)
			}
			
			// so sumSq_tmp_container is still the old ss val

			grad2 ~ grad2 + epsilon
			grad2 ~ um2 / grad2
			grad2 ~ grad2 *@ stepi

			// re-use the space newsquares here
			// the pnt jump from modelmats is modelmats + grad2
			// println("the grad in the new to prev " + grad2)
			// println(" the newsquares: " + newsquares(i))
			// println("the stepi " + stepi)
			newsquares(i) <-- mats_old(i) + grad2
			newsquares(i) ~ newsquares(i) - mats_new(i)
			loglik_prev_to_new +=  (-1.0*sum(sum(newsquares(i) *@ newsquares(i))) / 2.0 / (stepi*2)).dv
	
		}

		// then jump from the new mats to the old ones
		// copy the data to the models
		for (i <- 0 until mats_new.length) {
			model.modelmats(i) <-- mats_new(i)
		}

		// eval the new data
		model.dobatch(gmats, ipass, pos)
		updatemats = model.updatemats
		loss_mat_prev = model.evalbatch(gmats, ipass, pos)	// re-use the old reference here
		val loss_new = (sum(loss_mat_prev)).dv

		// compute the new scaled gradient
		for (i <- 0 until updatemats.length) {
			// clip
			if (cp > 0f) {
            	min(updatemats(i), clipByValue,updatemats(i));
          		max(updatemats(i),-clipByValue,updatemats(i));
    		}

			// compute the ss
			val ss2 = sumSq_tmp_container(i)
			val um2 = updatemats(i)
			newsquares(i) <-- um2 *@ um2 	// it's OK to reuse the newsquares

			ss2 ~ ss2 *@ (step - 1)
			ss2 ~ ss2 + newsquares(i)
			ss2 ~ ss2 / step
			val grad2 = ss2 ^ ve

			// de-affect the ss2
			ss2 ~ ss2 *@ step
			ss2 ~ ss2 - newsquares(i)
			if (step.dv > 1) {
				ss2 ~ ss2 / (step - 1)
			}
			

			grad2 ~ grad2 + epsilon
			grad2 ~ um2 / grad2
			grad2 ~ grad2 *@ stepi

			// re-use the space newsquares here
			// the pnt jump from candidate is candidate + grad2
			newsquares(i) <-- mats_new(i) + grad2
			newsquares(i) ~ newsquares(i) - mats_old(i)
			loglik_new_to_prev +=  (-1.0*sum(sum(newsquares(i) *@ newsquares(i))) / 2.0 / (stepi*2)).dv
		}

		val delta = (loss_new) - (loss_prev) + loglik_new_to_prev - loglik_prev_to_new

		if (java.lang.Double.isNaN(delta)) {
			println ("delta:" + delta + " loss_new:" + loss_new + " loss_prev:" + loss_prev + " loglik_new_to_prev:" + loglik_new_to_prev + " loglik_prev_to_new:" + loglik_prev_to_new)
			throw new RuntimeException("Delta")
		}
		delta

	}

}


// the stochastic gradient hamiltonian monte carlo updater
class SGHMC_proposer (val lr:Float, val a:Float, val t:Float, val v:Float, val cp:Float, val k:Float, val batchSize:Float, val model:Model) extends Proposer() {

	var step:Mat = null	// record the step by itself
	var candidate:Array[Mat] = null
	var stepi:Mat = null
	var is_estimte_sd:Boolean = true
	var alpha:Mat = null
	var v_old:Array[Mat] = null	// the v in the paper
	var sumSq:Array[Mat] = null // container for g*g
	var lrate:Mat = null
	var te:Mat = null
	var ve:Mat = null
	var noise_matrix:Array[Mat] = null // contain the v_new
	var epsilon:Float = 1e-5f
    var initsumsq = 1e-5f
	var clipByValue:Mat = null
	var newsquares:Array[Mat] = null
	var estimated_v:Mat = null
	var kir:Mat = null
	var m:Int = 1
	var adj_alpha:Mat = null
	var t_init:Mat = null

	override var has_help_mats:Boolean = true


	override def init():Unit = {
		// init the container here

		candidate = new Array[Mat](model.modelmats.length)
		sumSq = new Array[Mat](model.modelmats.length)	
		newsquares = new Array[Mat](model.modelmats.length)

		stepi = model.modelmats(0).zeros(1,1)
		step = model.modelmats(0).ones(1,1)
		
		te = model.modelmats(0).zeros(1,1)
		te(0,0) = t
		ve = model.modelmats(0).zeros(1,1)
		ve(0,0) = v
		lrate = model.modelmats(0).zeros(1,1)
		lrate(0,0) = lr
		v_old = new Array[Mat](model.modelmats.length)
		noise_matrix = new Array[Mat](model.modelmats.length)
		alpha = model.modelmats(0).zeros(1,1)
		alpha(0,0) = a

		estimated_v = model.modelmats(0).zeros(1,1)

		kir = model.modelmats(0).zeros(1,1)
		kir(0,0) = k

		t_init = model.modelmats(0).ones(1,1)
		t_init(0,0) = 1000.0f

		adj_alpha = model.modelmats(0).zeros(1,1)

		if (cp > 0) {
			clipByValue = model.modelmats(0).zeros(1,1)
			clipByValue(0,0) = cp
		}
		for (i <- 0 until candidate.length) {
			candidate(i) =  model.modelmats(i).zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
			sumSq(i) = model.modelmats(i).ones(model.modelmats(i).nrows, model.modelmats(i).ncols) *@ initsumsq
			newsquares(i) = model.modelmats(i).zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
			v_old(i) = model.modelmats(i).zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
			noise_matrix(i) = model.modelmats(i).zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
		}
		println("finish init the proposer")
		println("step: " + step + ", stepi" + stepi + ", te: " + te + ", ve: " + ve +", lrate: " + lrate)
	}


	override def changeToUpdateState():Unit = {
		is_estimte_sd = false
	}

	override def changeToEstimateSdState():Unit = {
		is_estimte_sd = true
	}

	// notice, the gradient computed by system is for max the objective...
	override def proposeNext(modelmats:Array[Mat], prev_v:Array[Mat], gmats:Array[Mat], ipass:Int, pos:Long):(Array[Mat], Array[Mat], Double) = {
		
		// compute the new v

		// copy the modelmats to the model
		for (i <- 0 until modelmats.length) {
			model.modelmats(i) <-- modelmats(i)
		}

		stepi <-- lrate / (step ^ te);

		// resample the v_old
		for (i <- 0 until v_old.length) {
			// normrnd(0, (stepi^0.5).dv, v_old(i))
			
			if (step.dv < -1.0) {
				normrnd(0, (stepi^0.5).dv, v_old(i))
			} else {
				v_old(i) <-- prev_v(i)
			}
			// normrnd(0, (stepi^0.5).dv, v_old(i))
		}	


		// copy the modelmats to candidates
		for (i <- 0 until modelmats.length) {
			candidate(i) <-- modelmats(i)
		}
		// do update for m steps
		for (j <- 0 until m) {
			for (i <- 0 until modelmats.length) {
				candidate(i) <-- candidate(i) + v_old(i)
				model.modelmats(i) <-- candidate(i)
			} 

			model.dobatch(gmats, ipass, pos)

			for (i <- 0 until candidate.length) {
				// clip
				if (cp > 0f) {
		        	min(model.updatemats(i), clipByValue, model.updatemats(i));
		      		max(model.updatemats(i),-clipByValue, model.updatemats(i));
				}

				// compute the ss
				val ss = sumSq(i)
				// since the gradient is the revise of the max for min problem
				val um = model.updatemats(i)
				newsquares(i) <-- um *@ um

				ss ~ ss *@ (step - 1)
				ss ~ ss + newsquares(i)
				ss ~ ss / step
				val grad = ss ^ ve

				grad ~ grad + epsilon
				grad ~ um / grad

				// estimate beta
				estimated_v ~ estimated_v *@ (1 - kir)
				estimated_v <-- estimated_v + sum(sum(grad *@ grad)) *@ kir / batchSize * 1000000 / grad.length
				// var tmp = 1 / batchSize * 1000000 / grad.length
				// println(tmp)
				// just add by my understanding not sure right
				// estimated_v <-- estimated_v / grad.length
				
				// just debug
				// println("estimated_v: " + estimated_v)
				
				adj_alpha <-- alpha
				
				
				if ((estimated_v*stepi/2.0).dv > alpha.dv) {
					adj_alpha = (estimated_v*stepi/2.0) + 1e-6f
					// println ("alpha change to be " + adj_alpha)
				}
				if (adj_alpha.dv > 0.2) {
					adj_alpha <-- alpha
				}	
				


				grad ~ grad *@ stepi

				// put the val into the container
				v_old(i) <-- (1.0-adj_alpha) *@ v_old(i) + grad
				// add the random noise
				val est_var = 2*(adj_alpha - estimated_v*stepi / 2.0) * stepi
				// println("the est var is " + estimated_v +" ,the var is " + est_var)
				if (est_var.dv < 0) {
					// println("the est var is " + estimated_v +" ,the var is " + est_var)
					est_var(0,0) = 1e-5f
				}
				
				normrnd(0, (est_var^0.5).dv, noise_matrix(i))
				v_old(i) <-- v_old(i) + noise_matrix(i)
				// println("the inserted noise is " + (est_var^0.5) + ", and "  + ((stepi * 0.001)^0.5) )
				/**
				// insert more noise?
				normrnd(0, ((stepi * 0.00001)^0.5).dv, noise_matrix(i))
				v_old(i) <-- v_old(i) + noise_matrix(i)
				**/
			}

		}	
		
		
		// compute the delta here
		// place the modelmats by the proposed one
		/**
		for (i <- 0 until candidate.length) {
			model.modelmats(i) <-- candidate(i)
		}
		val score_new = -1.0 * sum(model.evalbatch(gmats, ipass, pos))

		var enery_new = v_old(0).zeros(1,1)
		for (i <- 0 until candidate.length) {
			enery_new <-- enery_new + sum(sum(v_old(i) *@ v_old(i)))
		}
		enery_new ~ enery_new / 2 / stepi
		// println ("score_old: " + score_old + ", score_new: " + score_new + ", enery_new:" + enery_new + ", enery_old:"+enery_old)
		val delta = score_old + enery_old - score_new - enery_new
		**/
		// println ("the delta is " + delta)
		// incremental the count
		val delta = computeDelta(candidate, modelmats, v_old, prev_v, gmats, ipass, pos)
		if (!is_estimte_sd) {
			step ~ step + 1.0f
		}
		if (java.lang.Double.isNaN(delta.dv)) {
			throw new RuntimeException("Delta for proposer")
		}
		(candidate, v_old, delta)
	}


	override def computeDelta(mats_new:Array[Mat], mats_old:Array[Mat], new_v:Array[Mat], prev_v:Array[Mat], gmats:Array[Mat], ipass:Int, pos:Long): Double ={
		
		// compute the temperature
		val t_i = t_init / step ^(0.5)
		if (t_i.dv <= 1.0f) {
			t_i(0,0) = 1.0f
		}
		// val t_i = t_init

		for (i <- 0 until mats_old.length) {
			model.modelmats(i) <-- mats_old(i)
		}
		val score_old = -1.0 *sum(model.evalbatch(gmats, ipass, pos)) / t_i
		var enery_old = prev_v(0).zeros(1,1)
		for (i <- 0 until prev_v.length) {
			enery_old <-- enery_old + sum(sum(prev_v(i) *@ prev_v(i)))
		}
		enery_old ~ enery_old / 2 / stepi


		for (i <- 0 until mats_new.length) {
			model.modelmats(i) <-- mats_new(i)
		}
		val score_new = -1.0 *sum(model.evalbatch(gmats, ipass, pos)) / t_i

		var enery_new = v_old(0).zeros(1,1)
		for (i <- 0 until candidate.length) {
			enery_new <-- enery_new + sum(sum(v_old(i) *@ v_old(i)))
		}
		enery_new ~ enery_new / 2 / stepi
		// println ("score_old: " + score_old + ", score_new: " + score_new + ", enery_new:" + enery_new + ", enery_old:"+enery_old)
		val delta = score_old + enery_old - score_new - enery_new
		if (java.lang.Double.isNaN(delta.dv)) {
			throw new RuntimeException("Delta for proposer")
		}
		delta.dv
	} 
}


class Gradient_descent_proposer (val lr:Float, val u:Float, val t:Float, val v:Float, val cp:Float, val model:Model) extends Proposer() {
	var step:Mat = null	// record the step by itself
	var candidate:Array[Mat] = null
	var stepi:Mat = null
	var is_estimte_sd = true
	var mu:Mat = null
	var momentum:Array[Mat] = null
	var sumSq:Array[Mat] = null // container for g*g
	var lrate:Mat = null
	var te:Mat = null
	var ve:Mat = null
	var hasmomentum:Boolean = true
	var updatemats:Array[Mat] = null // just a reference
	var epsilon:Float = 1e-5f
    var initsumsq = 1e-5f
	var clipByValue:Mat = null
	var newsquares:Array[Mat] = null
	override var has_help_mats:Boolean = false


	override def init():Unit = {
		// init the container here
		hasmomentum = (u > 0)

		candidate = new Array[Mat](model.modelmats.length)
		sumSq = new Array[Mat](model.modelmats.length)	
		newsquares = new Array[Mat](model.modelmats.length)

		stepi = model.modelmats(0).zeros(1,1)
		step = model.modelmats(0).ones(1,1)
		
		te = model.modelmats(0).zeros(1,1)
		te(0,0) = t
		ve = model.modelmats(0).zeros(1,1)
		ve(0,0) = v
		lrate = model.modelmats(0).zeros(1,1)
		lrate(0,0) = lr
		if (hasmomentum) {
			momentum = new Array[Mat](model.modelmats.length)
			mu = model.modelmats(0).zeros(1,1)
			mu(0,0) = u
		}

		if (cp > 0) {
			clipByValue = model.modelmats(0).zeros(1,1)
			clipByValue(0,0) = cp
		}
		for (i <- 0 until candidate.length) {
			candidate(i) =  model.modelmats(i).zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
			sumSq(i) = model.modelmats(i).ones(model.modelmats(i).nrows, model.modelmats(i).ncols) *@ initsumsq
			newsquares(i) = model.modelmats(i).zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
			
			if (hasmomentum) {
				momentum(i) = model.modelmats(i).zeros(model.modelmats(i).nrows, model.modelmats(i).ncols)
			}
		}
		println("finish init the proposer")
		println("step: " + step + ", stepi" + stepi + ", te: " + te + ", ve: " + ve +", lrate: " + lrate)
	}

	override def proposeNext(modelmats:Array[Mat], prev_v:Array[Mat], gmats:Array[Mat], ipass:Int, pos:Long):(Array[Mat], Array[Mat], Double) = {
		// just do the one step gradient descent
		if (!is_estimte_sd) {

			for (i <- 0 until modelmats.length) {
				model.modelmats(i) <-- modelmats(i)
			}
			// compute the gradient
			model.dobatch(gmats, ipass, pos)
			updatemats = model.updatemats

			// sample the new model parameters by the gradient and the stepsize
			// and store the sample results into the candidate array
			stepi <-- lrate / (step ^ te);
			for (i <- 0 until candidate.length) {
				// clip
				if (cp > 0f) {
	            	min(updatemats(i), clipByValue,updatemats(i));
              		max(updatemats(i),-clipByValue,updatemats(i));
	    		}

				// compute the ss
				val ss = sumSq(i)
				val um = updatemats(i)
				newsquares(i) <-- um *@ um

				ss ~ ss *@ (step - 1)
				ss ~ ss + newsquares(i)
				ss ~ ss / step
				val grad = ss ^ ve

				grad ~ grad + epsilon
				grad ~ um / grad
				grad ~ grad *@ stepi
				if (hasmomentum) {
					grad ~ grad + momentum(i)
					momentum(i) ~ grad *@ mu
				}
				
				candidate(i) <-- modelmats(i) + grad
			}
			step ~ step + 1.0f
		}
		// for delta, we just return a very large value
		(candidate, null, 1000000.0)
	}

	override def changeToUpdateState():Unit = {
		is_estimte_sd = false
	}

	override def changeToEstimateSdState():Unit = {
		is_estimte_sd = true
	}

	override def computeDelta(mats_new:Array[Mat], mats_old:Array[Mat], new_v:Array[Mat], prev_v:Array[Mat], gmats:Array[Mat], ipass:Int, pos:Long): Double ={
		100.0
	} 
}

// Class of the emprical cdf of X_corr, there should be three
// matrix to hold the data computed from the matlab
// there are pre-computed txt file at /data/EcdfForMHtest

class Ecdf(val ecdfmat:FMat, val varvect:FMat) {
	var sd = 1.0f
	var f:FMat = null
	var x:FMat = null
	
	def init() = {
		// read the x
		x = ecdfmat(0, ?)
		updateSd(1.0)
	}

	def generateXcorr = {
		var u:Float = rand(1,1)(0,0)
		// println ("u is " + u) 
		val index = binarySearch(u, f)
		// println ("f is " + f)
		// println ("index is "+ index)
		x(0, index)
	}

	def updateSd (inputsd:Double):Unit = {
		sd = inputsd.toFloat
		if (sd > 1.2f) {
			throw new RuntimeException("Too large sd of Delta'")
		}
		// update the f
		// looking for the closest index in the hash
		val index = binarySearch(sd, varvect)
		f = ecdfmat(index+1, ?)
	}

	// return the closest index in xarray for u
	def binarySearch(u:Float, xarray:FMat) : Int = {
		var start : Int = 0
		var end : Int = xarray.ncols - 1
		var mid : Int = 0
		// println ("mid: "+ mid + " ,start: " + start + " ,end " + end)
		while (end > start + 1) {
			// println ("mid: "+ mid + " ,start: " + start + " ,end " + end)
			mid = (start + end) / 2
			if (u < xarray(0, mid)) {
				end = mid;
			} else if (u > xarray(0, mid)) {
				start = mid;
			} else {
				return mid
			}
		}
		// (x(start) + x(end))/2 * sd
		start
	}
}

