package BIDMach.viz

import BIDMach.Learner
import BIDMat.{FMat, GMat, Mat,IMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.networks._
import BIDMach.networks.layers._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.collection.mutable.HashMap


object utils {
    def getLayerInfo(layer:Layer): HashMap[String,Any] =
      if (layer == null) HashMap[String,Any]() else
      HashMap[String,Any]("name" -> layer.getClass.getSimpleName,
               "imodel" -> (layer match {case ml:ModelLayer=>ml.imodel;case _ => -1}) ,
               "modelName" -> (layer match {case ml:ModelLayer=>ml.opts.modelName;case _ => ""}),
               "inputDim" -> layer._inputs.map(i=>
                   if (i == null) "" else {                                               
                       val m = i.layer._outputs(i.term);
                       if (m == null)"@"+layer.getClass.getSimpleName else m.nrows+"*"+m.ncols
                   }),
               "outputDim" -> layer._outputs.map(m=>if (m == null)"@"+layer.getClass.getSimpleName else m.nrows+"*"+m.ncols),
               "modelDim" -> (layer match {
                   case ml:ModelLayer=>{
                       val m = ml.modelmats(ml.imodel);
                       m.nrows+"*"+m.ncols
                   }
                   case _ => ""
               }),
               "internalLayers" -> (layer match {
                   case ml:CompoundLayer=>ml.internal_layers.map(getLayerInfo(_));
                   case _ => Array[HashMap[String,Any]]()
               }))

  def getModelGraph(learner: Learner) = 
      learner.model match {
          case m:SeqToSeq =>              
              val layersInfo = m.layers.map(getLayerInfo(_))    
              HashMap[String,Any]("model" -> m.getClass.getSimpleName,
                             "inwidth" -> m.opts.inwidth,
                             "outwidth" -> m.opts.outwidth,                             
                             "height" -> m.opts.height,
                             "layers" -> layersInfo,
                             "id" -> "graph")
          case m:Net =>
              val layersInfo = m.layers.map(getLayerInfo(_))   
              for(i<-0 until m.layers.length) layersInfo(i)("layerId") = i

              HashMap[String,Any]("model" -> m.getClass.getSimpleName,
                             "layers" -> layersInfo,
                             "id" -> "graph")
          case _ =>
             HashMap[String,Any]()
  }
  val bidmachDir = "/raid/byeah/BIDMach/"
  val bidmachURL = "https://raw.githubusercontent.com/BIDData/BIDMach/master/src/main/"
  def process(learner: Learner)(msg:String) = {
      val data = msg.split("/")          
      if (data.head == "getCode")
//          scala.io.Source.fromFile(bidmachDir + "src/main/scala/BIDMach/networks/layers/"+data(1)).getLines.mkString("\n")
          scala.io.Source.fromURL(bidmachURL + "scala/BIDMach/networks/layers/" + data(1)).getLines.mkString("\n")
  }
  
    
  def getV(f:Double) = {
      val d = (f*256).toInt+128;
      if (d>255) 255 else d
  }  
    
  def filter2img(data: Mat, tensorFormat: Int, bw: Int = 1) = {
        val in_channel = data.dims(0)
        val w = data.dims(1)
        val h = data.dims(2)
        val num = data.dims(3)
        val new_data = (cpu(data).reshape(Array(w,h,in_channel,num))).asInstanceOf[FMat]
        val col = Math.sqrt(num).toInt
        val row = Math.ceil(num*1f/col).toInt
        val out = IMat(row*h*bw,col*w*bw)
        var i = 0
        var j = 0
        for(k<-0 until num){
            for(r<-0 until h)
                for(c<-0 until w){
                    val ii = i*h+r
                    val jj = j*w+c
                    //println(i,j,ii,jj,out.nrows,out.ncols)
                        
                    val ind = c+w*r+k*w*h*3
                    var res = getV(new_data.data(ind))
                    res += getV(new_data.data(ind+w*h))*256
                    res += getV(new_data.data(ind+w*h*2))*256*256
                    res += 255*256*256*256
                        
                    /*var res  = getV(new_data(c,r,0,k).dv)
                    res += getV(new_data(c,r,1,k).dv)*256
                    res += getV(new_data(c,r,2,k).dv)*256*256
                    res += 255*256*256*256*/
                    /*var res  = ((data(0,r,c,k)+0.5f).dv*256).toInt
                    res += ((data(1,r,c,k)+0.5f).dv*256).toInt*256
                    res += ((data(2,r,c,k)+0.5f).dv*256).toInt*256*256
                    res += 255*256*256*256*/
                    out(ii*bw->(ii*bw+bw),jj*bw->(jj*bw+bw)) = res
                }
            j+=1
            if (j == col) {j = 0;i+=1}
        }
        out
    }
    
}
