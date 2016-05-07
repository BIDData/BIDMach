package BIDMach
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GDMat,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Plotting._
import BIDMach.models._
import BIDMach.datasinks._


object Logging{
    def logGradientL2Norm(model:Model,data:Array[Mat]):Array[Mat] = {
      val m = model.modelmats
      val res = new Array[Float](m.length)
      for(i<-0 until m.length){
          res(i) = sum(snorm(m(i))).dv.toFloat
      }
      Array(FMat(m.length,1,res))
    }
  
    def logGradientL1Norm(model:Model,data:Array[Mat]):Array[Mat] = {
      val m = model.modelmats
      val res = new Array[Float](m.length)
      for(i<-0 until m.length){
          res(i) = sum(sum(abs(m(i)))).dv.toFloat
      }
      Array(FMat(m.length,1,res))
    }
    
    def getResults(model:Model): Array[Mat] = {
        model.opts.logDataSink match {
            case m:MatSink=>m.mats
            case f:FileSink=>{println("Found results at "+f.opts.ofnames.head(0));null}
            case null=>{println("No logDataSink found");null}
        }
    }
    
    def getResults(l:Learner): Array[Mat] = getResults(l.model)
}
