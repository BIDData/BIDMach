package BIDMach.updaters

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._

class ADAGradUpdater(override val opts:ADAGradUpdater.Opts = new ADAGradUpdater.Options) extends Updater {

  var firstStep = 0f
  var modelmat:Mat = null
  var updatemat:Mat = null
  var sumSq:Mat = null
  var stepn:Mat = null
  var mask:Mat = null
  var ve:Mat = null
  var te:Mat = null
  var alpha:Mat = null
  var one:Mat = null

  override def init(model0:Model) = {
    model = model0
    modelmat = model.modelmats(0)
    mask = opts.mask
    if (sumSq.asInstanceOf[AnyRef] == null) {
      sumSq = modelmat.ones(size(modelmat,1), size(modelmat,2)) *@ opts.initsumsq
    } else {
      sumSq.set(opts.initsumsq)
    }
    stepn = modelmat.zeros(1,1)
    one = modelmat.ones(1,1)
    ve = modelmat.zeros(opts.vexp.nrows, opts.vexp.ncols)
    te = modelmat.zeros(opts.texp.nrows, opts.texp.ncols)
    alpha = modelmat.zeros(opts.alpha.nrows, opts.alpha.ncols)
    ve <-- opts.vexp
    te <-- opts.texp
    alpha <-- opts.alpha
  }

  def update2(ipass:Int, step:Long):Unit = {
    val nsteps = if (step == 0) 1f else {
      if (firstStep == 0f) {
        firstStep = step
        1f
      } else {
        step / firstStep
      }
    }
    stepn.set(nsteps)
    val nw = one / stepn
    val newsquares = updatemat *@ updatemat
    newsquares ~ newsquares *@ nw
    sumSq  ~ sumSq *@ (one - nw)
    sumSq ~ sumSq + newsquares
    if (opts.waitsteps < nsteps) {
      val tmp = sumSq ^ ve
      tmp ~ tmp *@ (stepn ^ te)
      tmp ~ tmp + opts.epsilon
      modelmat ~ modelmat + ((updatemat / tmp) *@ alpha)
      if (mask != null) modelmat ~ modelmat *@ mask
    }
  }

  def update(ipass:Int, step:Long):Unit = {
    updatemat = model.updatemats(0)
    val nsteps = if (step == 0) 1f else {
      if (firstStep == 0f) {
        firstStep = step
        1f
      } else {
        step / firstStep
      }
    }
    stepn.set(nsteps)
    val nw = 1f / stepn
    val newsquares = updatemat *@ updatemat
    newsquares ~ newsquares *@ nw
    sumSq  ~ sumSq *@ (1f - nw)
    sumSq ~ sumSq + newsquares
    if (opts.waitsteps < nsteps) {
      val tmp = sumSq ^ ve
      tmp ~ tmp *@ (stepn ^ te)
      tmp ~ tmp + opts.epsilon
      tmp ~ updatemat / tmp
      tmp ~ tmp *@ alpha
      modelmat ~ modelmat + tmp
      if (mask != null) modelmat ~ modelmat *@ mask
    }
  }
}


object ADAGradUpdater {
  trait Opts extends GradUpdater.Opts {
    var vexp:FMat = 0.5f
    var epsilon = 1e-15f
    var initsumsq = 1e-8f
  }

  class Options extends Opts {}
}

