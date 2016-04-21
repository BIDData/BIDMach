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

class MHTest(var objective:Model, val proposer:Proposer, override val opts:MHTest.opts = new MHTest.Options) extends Model(opts) {

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
		parent_model(mats, ipass, here)
	}

	// help methods

	// load the distribution of X_corr with known sigma
	def loadDistribution = {

	}

	// estimate the sd of the delta
	def estimateSd:Double = {
		1.0f
	}

	// generate the random X_corr
	def generateRandomXcorr:Double = {
		0.0f
	}

	// compute the value of the delta
	def delta:Double = {
		
	}
	
}


object MHTest {
	trait  Opts extends Model.Opts {
		// TODO: define the parameters here
	}

	class Options extends Opts {}

	def learner(data:Mat) = {
		class xopts extends Learner.Options with MHTest.Opts with MatSource.Opts with IncNorm.Opts 
	    val opts = new xopts
	    // TODO: define the parameters for the opts

	    val nn = new Learner(
	      new MatSource(Array(data:Mat), opts),
	      new MHTest(opts),
	      null,
	      new IncNorm(opts),
	      null,
	      opts)
	    (nn, opts)
	}
}

abstract class Proposer() {

	// init the proposer class.
	def init() = {

	}

	// Function to compute the Pr(theta' | theta_t)
	def proposeProb(): Double = {}

	// Function to propose the next parameter, i.e. theta'
	def proposeNext():Array[Mat] = {}
}