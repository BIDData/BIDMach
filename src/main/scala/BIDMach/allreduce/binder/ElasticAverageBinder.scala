package BIDMach.allreduce.binder

import BIDMach.allreduce.binder.AllreduceBinder.{DataSink, DataSource}
import BIDMach.models.Model
import BIDMat.FMat


/**
  * Linearize input model mats, and elastic-average update to the same model.
  *
  * @param model
  * @param alpha
  */
class ElasticAverageBinder(model: Model, alpha: Double) extends AllreduceBinder {

  override lazy val totalDataSize: Int = {
    var ret = 0
    model.modelmats.synchronized {
      for (mat <- model.modelmats) {
        val fmat = FMat(mat)
        ret += fmat.length
      }
    }
    ret
  }

  override def dataSource: DataSource = inputRequest => {

    println(s"--Dumping model data at ${inputRequest.iteration}--")
    val ret: Array[Float] = new Array[Float](totalDataSize)

    model.modelmats.synchronized {
      var current = 0
      for (mat <- model.modelmats) {
        val contents = FMat(mat).contents
        var i = 0
        while (i < mat.length) {
          ret(current) = contents(i)
          current += 1
          i += 1
        }
      }
    }
    AllReduceInput(ret)

  }

  private def averageValueOrElse(sum: Float, count: Int): Option[Float] = {
    count match {
      case 0 => Option.empty
      case _ => Some(sum / count)
    }
  }

  override def dataSink: DataSink = reducedOutput => {
    println(s"-- Averaging model of iteration ${reducedOutput.iteration}--")

    val data = reducedOutput.data
    val count = reducedOutput.count
    var current = 0

    assert(data.length == totalDataSize, "Reduced output should be the same as as model")

    model.modelmats.synchronized {
      for (mat <- model.modelmats) {
        val contents = FMat(mat).contents
        var i = 0
        while (i < mat.length) {
          val potential_averaged = averageValueOrElse(data(current), count(current))
          for (averaged <- potential_averaged) {
            contents.update(i, averaged * alpha + contents(i) * (1 - alpha))
          }
          current += 1
          i += 1
        }
      }
    }
  }

}

