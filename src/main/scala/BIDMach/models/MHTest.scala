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

	var ecdf:Ecdf = new Ecdf(x_ecdf, ecdfmat, hash_ecdf)
	var delta:Double = 1.0
	var var_estimate_mat:FMat = null
	var sd_smooth_exp_param:Double = 0.7 // use the exp update to estimate var
	var estimated_sd:Double = 1.0
	var accpet_count:Float = 0.0f
	var reject_count:Float = 0.0f
	var batch_est_data:Array[Array[Mat]] = null
	var help_mats:Array[Mat] = null

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
		ecdf.init(opts.ratio_decomposite)
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
		val (next_mat:Array[Mat], update_v, delta:Double) = proposer.proposeNext(_modelmats, help_mats, mats, ipass, here)
		// do the test
		// println ("the delta is " + delta)
		ecdf.updateSd(estimated_sd)
		var x_corr = ecdf.generateXcorr
		if (x_corr + delta > 0) {
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
		println ("the sd of delat is " + estimated_sd + " accpet ratio is " + accpe_ratio)
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

		// compute the old loss
		var loss_mat_prev = model.evalbatch(gmats, ipass, pos)
		val loss_prev = (sum(loss_mat_prev)).dv

		// compute the jump probability
		var loglik_prev_to_new = 0.0
		var loglik_new_to_prev = 0.0
		for (i <- 0 until random_matrix.length) {
			loglik_prev_to_new += (-1.0*sum(sum(random_matrix(i) *@ random_matrix(i))) / 2.0 / (stepi*2)).dv
		}

		// jump from the candidate for one more step
		// model.setmodelmats(candidate)
		for (i <- 0 until candidate.length) {
			model.modelmats(i) <-- candidate(i)
		}
		model.dobatch(gmats, ipass, pos)
		updatemats = model.updatemats
		loss_mat_prev = model.evalbatch(gmats, ipass, pos)	// re-use the old reference here
		val loss_new = (sum(loss_mat_prev)).dv

		// compute the new scaled gradient
		for (i <- 0 until candidate.length) {
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

			grad2 ~ grad2 + epsilon
			grad2 ~ um2 / grad2
			grad2 ~ grad2 *@ stepi

			// re-use the space newsquares here
			// the pnt jump from candidate is candidate + grad2
			newsquares(i) <-- modelmats(i) - candidate(i)
			newsquares(i) ~ newsquares(i) - grad2
			loglik_new_to_prev +=  (-1.0*sum(sum(newsquares(i) *@ newsquares(i))) / 2.0 / (stepi*2)).dv
		}

		val delta = (loss_new) - (loss_prev) + loglik_new_to_prev - loglik_prev_to_new

		// update the iteration only if it's update
		if (!is_estimte_sd) {
			step ~ step + 1.0f
		}
		// println ("delta:" + delta + " loss_new:" + loss_new + " loss_prev:" + loss_prev + " loglik_new_to_prev:" + loglik_new_to_prev + " loglik_prev_to_new:" + loglik_prev_to_new)

		if (java.lang.Double.isNaN(delta)) {
			println ("delta:" + delta + " loss_new:" + loss_new + " loss_prev:" + loss_prev + " loglik_new_to_prev:" + loglik_new_to_prev + " loglik_prev_to_new:" + loglik_prev_to_new)
			throw new RuntimeException("Delta")
		}

		(candidate, null, delta)
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
	var m:Int = 3
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
		val score_old = -1.0 *sum(model.evalbatch(gmats, ipass, pos))
		stepi <-- lrate / (step ^ te);

		// resample the v_old
		for (i <- 0 until v_old.length) {
			// normrnd(0, (stepi^0.5).dv, v_old(i))
			if (step.dv < 100.0) {
				normrnd(0, (stepi^0.5).dv, v_old(i))
			} else {
				v_old(i) <-- prev_v(i)
			}
		}

		var enery_old = v_old(0).zeros(1,1)
		for (i <- 0 until candidate.length) {
			enery_old ~ enery_old + sum(sum(v_old(i) *@ v_old(i)))
		}
		enery_old ~ enery_old / 2 / stepi

		// copy the modelmats to candidates
		for (i <- 0 until modelmats.length) {
			candidate(i) <-- modelmats(i)
		}
		// do update for m steps
		for (j <- 0 until m) {
			for (i <- 0 until modelmats.length) {
				candidate(i) ~ candidate(i) + v_old(i)
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
				val um = -1.0 * model.updatemats(i)
				newsquares(i) <-- um *@ um

				ss ~ ss *@ (step - 1)
				ss ~ ss + newsquares(i)
				ss ~ ss / step
				val grad = ss ^ ve

				grad ~ grad + epsilon
				grad ~ um / grad

				// estimate beta
				estimated_v ~ estimated_v *@ (1 - kir)
				estimated_v ~ estimated_v + sum(grad *@ grad) *@ kir / batchSize

				grad ~ grad *@ stepi

				// put the val into the container
				v_old(i) ~ (1-alpha) *@ v_old(i) - grad

				// add the random noise
				val est_var = 2*(alpha - estimated_v*stepi / 2.0) * stepi
				normrnd(0, (est_var^0.5).dv, noise_matrix(i))
				v_old(i) ~ v_old(i) + noise_matrix(i)
			}

		}	
		
		
		// compute the delta here
		// place the modelmats by the proposed one
		for (i <- 0 until candidate.length) {
			model.modelmats(i) <-- candidate(i)
		}
		val score_new = -1.0 * sum(model.evalbatch(gmats, ipass, pos))

		var enery_new = v_old(0).zeros(1,1)
		for (i <- 0 until candidate.length) {
			enery_new ~ enery_new + sum(sum(v_old(i) *@ v_old(i)))
		}
		enery_new ~ enery_new / 2 / stepi
		// println ("score_old: " + score_old + ", score_new: " + score_new + ", enery_new:" + enery_new + ", enery_old:"+enery_old)
		val delta = score_old + enery_old - score_new - enery_new
		// println ("the delta is " + delta)
		// incremental the count
		if (!is_estimte_sd) {
			step ~ step + 1.0f
		}
		(candidate, v_old, delta.dv)
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
			if (u < f(mid)) {
				end = mid;
			} else if (u > f(mid)) {
				start = mid;
			} else {
				return x(mid) * sd
			}
		}
		(x(start) + x(end))/2 * sd
	}
}

