package BIDMach.allreduce

import BIDMat.{FMat, Mat}

/**
  * This class binds model data
  *
  * @param modelMats model mats
  */
class AllreduceBinder(modelMats: Array[Mat]) {

  val totalLength: Int = {
    var ret = 0
    modelMats.synchronized {
      for (mat <- modelMats) {
        val fmat = FMat(mat)
        ret += fmat.length
      }
    }
    ret
  }

  def generateDumpModel(): AllreduceNode.DataSource = {
    val dumpModel: AllreduceNode.DataSource = r => {
      //check if the learner has trained for one round
      println("-- Dumping model --")
      val ret: Array[Float] = new Array[Float](totalLength)
      modelMats.synchronized {
        var current = 0
        for (mat <- modelMats) {
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

  private def averageValueOrElse(sum: Float, count: Int): Option[Float] = {
    count match {
      case 0 => Option.empty
      case _ => Some(sum / count)
    }
  }

  def generateAverageModel(alpha: Double): AllreduceNode.DataSink = {
    val averageModel: AllreduceNode.DataSink = sink => {
      println(s"-- Averaging model of iteration ${sink.iteration}--")
      val data = sink.data
      val count = sink.count
      var current = 0

      modelMats.synchronized {
        for (mat <- modelMats) {
          val contents = FMat(mat).contents
          for (i <- 0 until mat.length) {
            val potential_averaged = averageValueOrElse(data(current), count(current))
            for (averaged <- potential_averaged) {
              contents.update(i, averaged * alpha + contents(i) * (1 - alpha))
            }
            current += 1
          }
        }
      }
    }
    averageModel
  }

}

