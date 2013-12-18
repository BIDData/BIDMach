package BIDMach.updaters
 
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._

class CGUpdater(override val opts:CGUpdater.Opts = new CGUpdater.Options) extends Updater(opts) {
	var res:Mat = null
	var Ap:Mat = null
	var pm:Mat = null
	var rm:Mat = null
	var mm:Mat = null
	var lastStep = -1L
	
	override def init(model0:Model) = {
  	super.init(model0)
  	mm = model0.modelmats(0)
	  res = mm.zeros(mm.nrows, mm.ncols)
	  Ap = mm.zeros(mm.nrows, mm.ncols)
	  pm = mm.zeros(mm.nrows, mm.ncols)
	  rm = mm.zeros(mm.nrows, mm.ncols)
	  model.asInstanceOf[CGUpdateable].setpm(pm)
	  lastStep = -1
  }
	
	def update(ipass:Int, step:Long) = {
	  val updatemats = model.updatemats
	  if (ipass < opts.spasses) {
	    mm <-- updatemats(0)
	  } else {
	  	res ~ res + updatemats(0)
	  	Ap ~ Ap + updatemats(1)
	  }
	}
	
	override def updateM(ipass:Int) = {  
//	  if (ipass == 0) pm <-- res
	  if (ipass >= opts.spasses) {
	  	if (ipass == opts.spasses || opts.moving) rm <-- res	  
	  	CGUpdater.CGupdate(pm, rm, Ap, mm, opts.meps, opts.convgd)
	  }
	  Ap.clear
	  res.clear
	  lastStep = -1
  }
	
	override def clear = {  
  }
}

trait CGUpdateable {
  def setpm(pm:Mat)
}

object CGUpdater {
  trait Opts extends Updater.Opts {
  	var meps = 1e-12f
  	var convgd = 1e-1f
  	var moving = true
  	var spasses = 2
  }  
  class Options extends Opts {}
  
  def CGupdate(p:Mat, r:Mat, Ap:Mat, x:Mat, weps:Float, convgd:Float) = {
  	val pAp = (p dot Ap)
  	max(pAp, weps, pAp)
  	val rsold = (r dot r) + 0                // add 0 to make a new vector, Otherwise this will alias...
//   	println("    pAp min %f, rsoldm %f, rsoldx %f" format (mini(pAp).dv, maxi(rsold).dv, mean(rsold).dv))
  	val convec = rsold > convgd              // Check convergence
//  	println("    conv %f" format (1f - sum(convec).dv / convec.length))
  	val alpha = convec *@ (rsold / pAp)      // Only process unconverged elements
  	min(alpha, 1f, alpha)
//  	println("    alpha max %f" format (maxi(alpha).dv))
  	x ~ x + (p *@ alpha)
  	r ~ r - (Ap *@ alpha)
  	val rsnew = (r dot r)                    // ...down here
//  	println("    rsnewm %f, rsnewx %f" format (maxi(rsnew).dv, mean(rsnew).dv))
  	max(rsold, weps, rsold)
  	val beta = convec *@ (rsnew / rsold)
  	min(beta, 1f, beta)
//  	println("    beta max %f" format (maxi(beta).dv))
  	p ~ r + (p *@ beta)
  }  
}
