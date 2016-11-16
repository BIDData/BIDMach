package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,FND,GMat,GDMat,GIMat,GSMat,GSDMat,GND,HMat,IMat,JSON,LMat,ND,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.datasinks._
import BIDMach.updaters._
import BIDMach._

/**
 * A simple Gaussian Mixture Model to test out experiment 5.1 in
 *      "Bayesian Learning via Stochastic Gradient Langevin Dynamics" (ICML 2011)
 * The main purpose is to use this in the context of the MH test, to see that a
 * simple case works. Little heed is paid to code generalization. For instance, it
 * only works with n=2 components for the Multivariate Gaussian. Individually, this
 * method will simply optimize for "func" based on the data.
 * 
 * Written by Daniel Seita (May 2016).
 */
class GaussianMixture(override val opts:GaussianMixture.Opts = new GaussianMixture.Options) extends Model(opts) {

    var theta:Mat = null // References modelmats(0), i.e., the 2-D \theta vector we are interested in.
    
    /** Sets up the modelmats and updatemats, which each simply consist of one 2-D vector. */
    override def init() = {
        setmodelmats(new Array[Mat](1))
        modelmats(0) = convertMat(rand(2,1))
        theta = modelmats(0)
        updatemats = new Array[Mat](1)
        updatemats(0) = theta.zeros(theta.nrows, theta.ncols)
    }
  
    /** Based on the mini-batch, compute a gradient update term via finite differences. */
    override def dobatch(mats:Array[Mat], ipass:Int, here:Long) = {
        println("ipass = " + ipass + ", func(theta, mats(0)) = " + func(theta, mats(0)))
        val eps = 0.00001
        val base1 = FMat(1 on 0)
        val base2 = FMat(0 on 1)
        val term1 = func(theta+base1, mats(0)) - func(theta-base1, mats(0))
        val term2 = func(theta+base2, mats(0)) - func(theta-base2, mats(0))
        val gradient = FMat(term1 on term2) * (1.0 / (2*eps))
        updatemats(0) = convertMat(gradient)
    }
  
    /** Computes the posterior (log) probability of the current parameters given the data. */
    override def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat = {
        return FMat(func(theta, mats(0)))
    }
    
    /** Returns the function of interest, the log-likelihood, for arbitrary parameters. */
    def func(param:Mat, data:Mat):Float = {
        val c1 = ln((1.0 / (2 * scala.math.Pi * sqrt(10)))).dv.toFloat
        val c2 = ln((1.0 / (4 * sqrt(scala.math.Pi)))).dv.toFloat
        val inverse_covariance = FMat(0.1 \ 0 on 0 \ 1)
        val first = (c1 - 0.5*(param.t)*inverse_covariance*param).dv.toFloat
        var second = 0f
        for (i <- 0 until mats(0).length) {
            val x_i = mats(0)(i)
            val gauss1 = exp(-0.25 * (x_i-param(0)) * (x_i-param(0)))
            val gauss2 = exp(-0.25 * (x_i-(param(0)+param(1))) * (x_i-(param(0)+param(1))))
            second = second + c2 + ln(gauss1 + gauss2).dv.toFloat
        }
        return first + second
    }
}


object GaussianMixture {
    trait Opts extends Model.Opts {}
        
	class Options extends Opts {} 
	
	/** A learner with a single matrix data source. */
    def learner(data:Mat) = {
        class xopts extends Learner.Options with GaussianMixture.Opts with MatSource.Opts with ADAGrad.Opts 
        val opts = new xopts

        val nn = new Learner(
            new MatSource(Array(data:Mat), opts),
            new GaussianMixture(opts),
            null,
            new ADAGrad(opts),
            null,
            opts)
        (nn, opts)
    }
}
