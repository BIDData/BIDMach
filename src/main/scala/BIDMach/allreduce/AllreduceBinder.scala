package BIDMach.allreduce

import BIDMach.Learner
import BIDMach.datasources.DataSource
import BIDMach.models.Model
import BIDMat.{FMat, IMat, Mat}
import BIDMach.allreduce.AllreduceNode.{DataSink, DataSource}
import akka.actor.Actor

class AllreduceBinder(model: Model, alpha: Double)  {

  private def getTotalLength : Int = {
    var ret = 0
    for (mat <- model.modelmats) {
      ret += mat.length
    }
    ret
  }

  def generateDumpModel(): AllreduceNode.DataSource = {
    val dumpModel: AllreduceNode.DataSource = r => {
      val totalLength : Int = getTotalLength
      val ret: Array[Float] = new Array[Float](totalLength)
      model.modelmats.synchronized {
        var current = 0
        for (mat <- model.modelmats) {
          val contents = FMat(mat).contents
          val offset = current
          for (i <- 0 until mat.length) {
            ret(current) = contents(offset + i)
            current += 1
          }
        }
      }
      AllReduceInput(ret)
    }
    dumpModel
  }

  private def averageValueOrElse(sum: Float, count: Int): Option[Float] ={
    count match {
      case 0 => Option.empty
      case _ => Some(sum/count)
    }
  }

  def generateAverageModel(): AllreduceNode.DataSink = {
    val averageModel: AllreduceNode.DataSink = sink => {
      val data = sink.data
      val count = sink.count
      var current = 0
      model.modelmats.synchronized {
        for (mat <- model.modelmats) {
          val contents = FMat(mat).contents
          val offset : Int = current
          for(i <- 0 until mat.length){
            val averaged = averageValueOrElse(data(current), count(current))
            if(averaged.isDefined){
              contents.update(offset+i, averaged.get * alpha + contents(offset+i) * (1-alpha))
            }
            current+=1
          }
        }
      }
    }
    averageModel
  }

}

