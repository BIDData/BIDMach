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
    for (mat <- model.modelmats) {
      ret += mat.length
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
        current -= mat.length
        mat match {
          case gmat: GMat => GMat.GPUtoCPUarraycopy(gmat.pdata, 0, ret, current, gmat.length, "ElasticAverageBinder dataSource");
          case fmat: FMat => System.arraycopy(fmat.contents().data, 0, ret, current, fmat.length);
        }
        i -= 1
      }
    }
    AllReduceInput(ret)

  }



  override def dataSink: DataSink = reducedOutput => {

    reduceCount.synchronized {
      val currentCount: Int = reduceCount.getAndIncrement()
      val updateCounts = AllreduceBinder.updateCounts
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
    var current = totalDataSize
    var i = model.modelmats.length - 1
    val alpha = alphaFromIter(reducedOutput.iteration)

    while (i >= 0) {
      val mat = model.modelmats(i)
      mat.synchronized {
        mat match {
          case gmat: GMat =>
            val gReduced = GMat.make(gmat.dims)
            GMat.CPUtoGPUarraycopy(reducedData, current - gmat.length, gReduced.pdata, 0, gmat.length, "ElasticAverageBinder dataSink")
            gmat ~ gmat * (1 - alpha)
            gReduced ~ gReduced * alpha
            gmat ~ gReduced + gmat
            gReduced.free()
          case fmat: FMat =>
            val fReduced = FMat.make(fmat.dims)
            System.arraycopy(reducedData, current - fmat.length, fReduced.contents().data, 0, fmat.length)
            fmat ~ fmat * (1 - alpha)
            fReduced ~ fReduced * alpha
            fmat ~ fReduced + fmat
        }
        current -= mat.length
      }
      i -= 1
    }

    assert(current == 0, "current should be zero after iteration")

  }

}

