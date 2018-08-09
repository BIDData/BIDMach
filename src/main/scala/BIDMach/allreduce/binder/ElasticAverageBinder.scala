package BIDMach.allreduce.binder

import java.util.concurrent.atomic.AtomicInteger
import java.util.logging.Logger

import BIDMach.allreduce.binder.AllreduceBinder.{DataSink, DataSource}
import BIDMach.models.Model
import BIDMat.{FMat, GMat, Mat}


/**
  * Linearize input model mats, and elastic-average update to the same model.
  *
  * @param model
  * @param alphaFromIter
  */
class ElasticAverageBinder(model: Model, alphaFromIter: Int => Float, logger: Logger) extends AllreduceBinder {

  // Keeping track of elastic updates
  var tick = System.currentTimeMillis()
  val reduceCount = new AtomicInteger()
  
  var aelem: Mat = null

  override lazy val totalDataSize: Int = {
    var ret = 0
    model.modelmats.synchronized {
      for (mat <- model.modelmats) ret += mat.length
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
      current -= mat.length
      mat match {
        case gmat: GMat => GMat.GPUtoCPUarraycopy(gmat.pdata, 0, ret, current, gmat.length, "ElasticAverageBinder dataSource")
        case fmat: FMat => System.arraycopy(fmat.contents().data, 0, ret, current, fmat.length)
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
        val tock = System.currentTimeMillis()
        if (currentCount > 0) {
          logger.info(f"elastic_updates/s=${updateCounts/((tock - tick) / 1.0e3)}%2.2f, total_updates=$currentCount")
        }
        tick = tock
      }
    }
    val reducedData = reducedOutput.data

    assert(reducedData.length == totalDataSize, "Reduced output should be the same as model")

    // backward traversing model mats, assuming forward traversal by the training model
    // using while instead of for loop due to performance
    var current = totalDataSize
    var i = model.modelmats.length - 1
    val alpha = alphaFromIter(reducedOutput.iteration)
    if (aelem eq null) aelem = model.modelmats(0).zeros(1, 1)

    while (i >= 0) {
      val mat = model.modelmats(i)
      current -= mat.length
      mat.synchronized {
        mat match {
          case gmat: GMat =>
            val gReduced = GMat.make(gmat.dims)
            GMat.CPUtoGPUarraycopy(reducedData, current, gReduced.pdata, 0, gmat.length, "ElasticAverageBinder dataSink")
            gmat ~ gmat * aelem.set(1 - alpha)
            gReduced ~ gReduced * aelem.set(alpha)
            gmat ~ gReduced + gmat
            gReduced.free()
          case fmat: FMat =>
            val fReduced = FMat.make(fmat.dims)
            System.arraycopy(reducedData, current, fReduced.contents().data, 0, fmat.length)
            fmat ~ fmat * aelem.set(1 - alpha)
            fReduced ~ fReduced * aelem.set(alpha)
            fmat ~ fReduced + fmat
        }
      }
      i -= 1
    }

    assert(current == 0, "current should be zero after iteration")

  }

}

