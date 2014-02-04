package BIDMach.updaters

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._

class GradUpdater(override val opts:GradUpdater.Opts = new GradUpdater.Options) extends Updater {

  var firstStep = 0f
  var modelmat:Mat = null
  var updatemat:Mat = null
  var sumSq:Mat = null
  var stepn:Mat = null
  var mask:Mat = null
  var ve:Mat = null
  var te:Mat = null
  var alpha:Mat = null

  override def init(model0:Model) = {
    model = model0
    modelmat = model.modelmats(0)
    updatemat = model.updatemats(0)
    mask = opts.mask
    stepn = modelmat.zeros(1,1)
    te = modelmat.zeros(opts.texp.nrows, opts.texp.ncols)
    alpha = modelmat.zeros(opts.alpha.nrows, opts.alpha.ncols)
    te <-- opts.texp
    alpha <-- opts.alpha
  }

  def update(ipass:Int, step:Long):Unit = {
    val nsteps = if (step == 0) 1f else {
      if (firstStep == 0f) {
        firstStep = step
        1f
      } else {
        step / firstStep
      }
    }
    stepn.set(1f/nsteps)
    if (opts.waitsteps < nsteps) {
      val tmp = updatemat *@ (alpha *@ (stepn ^ te))
      modelmat ~ modelmat + tmp
      if (mask != null) modelmat ~ modelmat *@ mask
    }
  }
}


object GradUpdater {
  trait Opts extends Updater.Opts {
    var alpha:FMat = 1f
    var texp:FMat = 0.5f
    var waitsteps = 2
    var mask:FMat = null
  }

  class Options extends Opts {}
}

