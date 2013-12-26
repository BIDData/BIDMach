package BIDMach.updaters

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._


object BatchMultUpdater {
  trait Opts extends Updater.Opts {
    var eps = 1e-12
  }

  class Options extends Opts {}
}

