package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._



abstract class RegressionModel(opts:RegressionModel.Options) 
  extends Model {
  
  val options = opts   
  var lls:Mat = null

  override def initmodel:Unit = {
    modelmats = new Array[Mat](1)
    modelmats(0) = if (opts.useGPU) GMat(opts.nrows, opts.nmodels) else FMat(opts.nrows, opts.nmodels)
    lls = if (opts.useGPU) gzeros(opts.nmodels, 1) else zeros(opts.nmodels, 1)
  }
  
  def derivfn(targ:Mat, pred:Mat, lls:Mat):Mat 
  
  def llfun(targ:Mat, pred:Mat):Mat
  
  override def doblock(istart:Int, iend:Int):Unit = {
    val range = istart -> iend
    val sdata = datamats(0)(?, range)
    val target = datamats(1)(?, range)
    val mupdate = updatemats(0)
    
    val dd = derivfn(target, sdata, lls)    
    mupdate ~ mupdate + dd *^ sdata
  }
  
  def evalfun(istart:Int, iend:Int):Mat = {
    val range = istart -> iend
    val sdata = datamats(0)(?, range)
    val target = datamats(1)(?, range)
    llfun(target, sdata)
  }
 
}

object RegressionModel {
  class Options extends Model.Options {
    var nrows = 0
    var nmodels = 0
    var transpose:Boolean = false
  }
}
