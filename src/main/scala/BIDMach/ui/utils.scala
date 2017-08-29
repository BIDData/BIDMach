package BIDMach.ui

import BIDMach.Learner
import BIDMat.{FMat, GMat, Mat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.networks._
import BIDMach.networks.layers._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.collection.immutable.HashMap


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
}