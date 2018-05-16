package BIDMach.allreduce.binder

import java.util.concurrent.atomic.AtomicInteger
import java.util.logging.Logger

import BIDMach.allreduce.binder.AllreduceBinder.{DataSink, DataSource}
import BIDMach.models.Model
import BIDMat.{FMat, GMat}


/**
  * Linearize input model mats, and elastic-average update to the same model.
  *
  * @param model
  * @param alphaFromIter
  */
class ElasticAverageBinder(model: Model, alphaFromIter: Int => Float, logger: Logger) extends AllreduceBinder {

  // Keeping track of elastic updates
  var tic = System.currentTimeMillis()
  val reduceCount = new AtomicInteger()

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



  override def dataSink: DataSink = reducedOutput => {

    reduceCount.synchronized {
      val currentCount: Int = reduceCount.getAndIncrement()
      val updateCounts = 10
      if (currentCount % updateCounts == 0) {
        val toc = System.currentTimeMillis()
        if (currentCount > 0) {
          logger.info(f"elastic_updates/s=${updateCounts/((toc - tic) / 1.0e3)}%2.2f, total_updates=$currentCount")
        }
        tic = toc
      }
    }
    val reducedData = reducedOutput.data

    assert(reducedData.length == totalDataSize, "Reduced output should be the same as as model")

    // backward traversing model mats, assuming forward traversal by the training model
    // using while instead of for loop due to performance
    var current = totalDataSize - 1
    var i = model.modelmats.length - 1
    val alpha = alphaFromIter(reducedOutput.iteration)

    while (i >= 0) {
      val mat = model.modelmats(i)
      mat.synchronized {
        val modelData = FMat(mat).data
        var j = mat.length - 1
        while (j >= 0) {
          modelData(j) = reducedData(current) * alpha + modelData(j) * (1 - alpha)
          j -= 1
          current -= 1
        }
        mat match {
          case gmat: GMat =>
            GMat.CPUtoGPUarraycopy(modelData, 0, gmat.pdata, 0, mat.length, "")
          case fmat: FMat =>
            // Already updated in-place fmat array during the elastic averaging
        }
      }
      i -= 1
    }

    assert(current == -1, "current should be zero after iteration")

  }

}

