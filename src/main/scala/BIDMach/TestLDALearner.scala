package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

import datasources._
import models._
import updaters._

object TestLDALearner {
  
  def runLDALearner(mat0: Mat, blockSize0: Int, ndims0: Int, useGPU0: Boolean, alpha0: Float, beta0: Float, nsamps0: Int, npasses0: Int, uiter0: Int): Unit = {
    val ds = new MatDataSource(Array(mat0))
    ds.opts.blockSize = blockSize0
    
    val model = new GibbsLDAModel()
    model.opts.dim = ndims0
    model.opts.useGPU = useGPU0
    model.opts.alpha = alpha0
    model.opts.beta = beta0
    model.opts.nsamps = nsamps0
    model.opts.putBack = 1
    model.opts.uiter = uiter0
    
    //val updater = new NaiveUpdater()
    val updater = new IncNormUpdater()
    
    val lopts = new Learner.Options()
    lopts.npasses = npasses0
    
    val learner = new Learner(ds, model, null, updater, lopts)
 
    learner.init
    learner.run
    
    saveFMat("/home/hzhao/code/gibbs/m1.lz4", FMat(model.asInstanceOf[Model].modelmats(0)))
  }
  
  def main(args: Array[String]): Unit = {
    val dirname = args(0)
    val blockSize0 = args(1).toInt
    val ndims0 = args(2).toInt
    //val nthreads = args(3).toInt
    val useGPU0 = args(3).toBoolean
    val alpha0 = args(4).toFloat
    val beta0 = args(5).toFloat
    val nsamps0 = args(6).toInt
    val npasses0 = args(7).toInt
    val uiter0 = args(8).toInt
    Mat.checkMKL
    Mat.checkCUDA
    Mat.useCache = true
    val mat0: SMat = loadSMat(dirname)
    runLDALearner(mat0, blockSize0, ndims0, useGPU0, alpha0, beta0, nsamps0, npasses0, uiter0)
  } 

}