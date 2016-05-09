package BIDMach.models
// still needs to update new methods, just a brief framework here

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

class CUTMHTest(var objective:Model, val proposer:MatProposer,
	override val opts:CUTMHTest.Opts = new CUTMHTest.Options) extends Model(opts) {

    var l_mean: Double = 0.0
    var l_square_mean: Double = 0.0
    var batch_data_size: Double = 0.0
    var var_U0: Double = 0.0
    var Total_data_size: Double = 0.0
    var var_S: Double = 0.0


	def compute_U0():Double = {
		// compute u0 = 1/N * ...
		// log( u*  {rho(\theta_t)q(\theta'|theta_t)} /{rho(\theta')q(theta_t|theta')} )
		var u :Float = rand(1,1)(0,0);
		var_U0 = 0.0 
		return var_U0
	}

	def update_U(modelmats:Array[Mat], prev_mats:Array[Mat], ipass:Int, batchdatasize:Double, pos:Long, prev_U: Double): (Double , Double )={
		// compute new u 
		// var l_mean_new : Double = (l_mean * batchdatasize * (ipass-1)+ new_data)/(batchdatasize+batchsize)
		// var l_square_mean_new: Double = (l_square_mean * batchdatasize * (ipass-1) + new_data^2)(batchdatasize + batchsize)
		// l_mean = l_mean_new
		// l_square_mean = l_square_mean_new
		var sl : Double = sqrt( (l_square_mean - l_mean^2) *batchdatasize*ipass /(batchdatasize*ipass-1) )(0)
		var_S = sl/sqrt(ipass*batchdatasize)(0) * sqrt( 1 - ( ipass * batchdatasize - 1)/(Total_data_size - 1) )(0)
		batch_data_size = batch_data_size + batchdatasize
		(var_S, l_mean)
	}

	def calculate_Delta(mean_l: Double, s: Double , U_0: Double, batchdatasize: Double):Double ={
		// update_U()
		var delta: Double = 0.001 // 
		return delta
	}


	override def init() = {

		objective.mats = mats
		objective.putBack = datasource.opts.putBack;
		objective.useGPU = opts.useGPU && Mat.hasCUDA >0;
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

		// init the datamat
		var mat = datasource.next
	}

	override def dobatch(mats:Array[Mat], ipass:Int, here:Long) = {
		// call the proposer to get the new theta'

		var new_modelmats: Array[Mat] = null

		// generate the u0 variable
		// generate a new u variable then do the iterative test to decide to accept or not
		
		batch_data_size = 0

		var_U0 = compute_U0()
		var delta : Double = 0
		var accept_reject : Int = 0
		while(accept_reject == 0){
			// update_U
			delta = calculate_Delta(l_mean, var_S, var_U0, batch_data_size)
			if ((delta < opts.epsilon) && (l_mean > var_U0)) {
				accept_reject = 1 // accept
			}else if ( (delta < opts.epsilon) && (l_mean < var_U0) ){
				accept_reject = 2 // reject
			}
		}

		if (accept_reject == 1){
			//_modelmats = new_modelmats 
		}
	}

	override def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat = {
		// call the parent class to compute the loss of the model
		var res : FMat = null
		return res
	}

}




object CUTMHTest{
	trait Opts extends Model.Opts {
		var epsilon = 0.01 // epsilon for t-test

	}

	class Options extends Opts{}

}





abstract class MatProposer() {
	def init(): Unit = {

	}
}




class Langevin_MatProposer() extends MatProposer() {

	override def init(): Unit ={

	}

}

