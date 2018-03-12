package BIDMach.allreduce

import BIDMach.Learner
import BIDMat.{FMat}

/**
  * This class binds
  * @param learner the learner we need to bind with
  * @param alpha the parameter used
  */
class AllreduceBinder(learner: Learner, alpha: Double){
  val model = learner.model

  /**
    * @return total length for the model
    */
  private def getTotalLength : Int = {
    var ret = 0
    model.modelmats.synchronized {
      for (mat <- model.modelmats) {
        val fmat = FMat(mat)
        ret += fmat.length
      }
    }
    ret
  }

  /**
    * Wait until the first round start in order to make sure the model parameter is properly initialized
    * TODO1: using a signal instead of thread sleep to decrease latency
    * TODO2: Possibly extend to certain round
    */
  private def waitUntilFirstRound(): Unit = {
    var waitInterval = 1
    while(!learner.synchronized{(learner.ipass > 0  || learner.istep > 0)
    }){
      Thread.sleep(waitInterval)
      waitInterval *=2
    }
  }

  def generateDumpModel(): AllreduceNode.DataSource = {
    val dumpModel: AllreduceNode.DataSource = r => {
      //check if the learner has trained for one round
      println("-- Dumping model --")
      waitUntilFirstRound()
      val totalLength : Int = getTotalLength
      val ret: Array[Float] = new Array[Float](totalLength)
      model.modelmats.synchronized {
        var current = 0
        for (mat <- model.modelmats) {
          val contents = FMat(mat).contents
          for (i <- 0 until mat.length) {
            ret(current) = contents(i)
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
      println(s"-- Averaging model round ${sink.iteration}--")
      val data = sink.data
      val count = sink.count
      var current = 0

      model.modelmats.synchronized {
        for (mat <- model.modelmats) {
          val contents = FMat(mat).contents
          for(i <- 0 until mat.length){
            val potential_averaged = averageValueOrElse(data(current), count(current))
            for(averaged <- potential_averaged){
              contents.update(i, averaged * alpha + contents(i) * (1-alpha))
            }
            current+=1
          }
        }
      }
    }
    averageModel
  }

}

