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

    // backward traversing model mats, assuming forward traversal by the training model
    // using while instead of for loop due to performance
    var current = totalDataSize
    var i = model.modelmats.length - 1

    while (i >= 0) {
      val mat = model.modelmats(i)
      mat.synchronized {
        val contentData = FMat(mat).contents.data
        current -= contentData.length
        System.arraycopy(contentData, 0, ret, current, contentData.length)
      }
      i -= 1
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

    assert(data.length == totalDataSize, "Reduced output should be the same as as model")

    // backward traversing model mats, assuming forward traversal by the training model
    // using while instead of for loop due to performance
    var current = totalDataSize - 1
    var i = model.modelmats.length - 1

    while (i >= 0) {
      val mat = model.modelmats(i)
      mat.synchronized {
        val contents = FMat(mat).contents
        var j = mat.length - 1
        while (j >= 0) {
          averageValueOrElse(data(current), count(current)) match {
            case Some(averaged) => contents.update(j, averaged * alpha + contents(j) * (1 - alpha))
            case _ => // No update when reduced data has no content
          }
          j -= 1
          current -= 1
        }
      }
      i -= 1
    }

  }

}

