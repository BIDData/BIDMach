package BIDMach.allreduce.binder

import BIDMach.BIDMachSpec
import BIDMach.models.Model
import BIDMat.MatFunctions._
import BIDMat.{FMat, Mat}

class ElasticAverageBinderSpec extends BIDMachSpec {

  val elasticRate = 0.5
  val model = new ElasticAverageTestModel()
  val binder = new ElasticAverageBinder(model, elasticRate)


  "Elastic binder" should "calculate total data size from model mats" in {

    binder.totalDataSize shouldEqual model.expectedDataSize()

  }

  "Elastic binder" should "linearize model mats" in {

    val source = binder.dataSource

    val allReduceInput = source(AllReduceInputRequest(iteration = 0))

    allReduceInput.data.length shouldEqual model.expectedDataSize()

    allReduceInput.data.toList shouldEqual List[Float](1, 3, 5, 7, 9, 2, 4, 6, 8, 10)

  }


  "Elastic binder" should "make weighted average between original and reduced values" in {

    val sink = binder.dataSink

    val averagedValue = (0 until 10).map(_.toFloat)

    sink(AllReduceOutput(averagedValue.toArray, iteration = 0))

    model.modelmats.length shouldEqual 2

    val mat1: FMat = model.modelmats(0).asInstanceOf[FMat]
    val mat2: FMat = model.modelmats(1).asInstanceOf[FMat]

    checkSimilar(mat1, row((0 until 5).map(_ * elasticRate).toArray) + row(1, 3, 5, 7, 9) * (1 - elasticRate))
    checkSimilar(mat2, col((5 until 10).map(_ * elasticRate).toArray) + col(2, 4, 6, 8, 10) * (1 - elasticRate))

  }

  "Elastic binder" should "fail when all reduce returns output of different size" in {

    val sink = binder.dataSink

    val list = (0 until 20).map(_ * 2.0f)

    intercept[AssertionError] {
      sink(AllReduceOutput(list.toArray, iteration = 0))
    }

  }


}


class ElasticAverageTestModel(val _modelmat: Array[Mat]) extends Model {

  def this() {
    this(Array[Mat](row(1, 3, 5, 7, 9), col(2, 4, 6, 8, 10)))
  }

  def expectedDataSize(): Int = {
    10
  }

  override def modelmats: Array[Mat] = {
    _modelmat
  }

  override def init(): Unit = ???

  override def dobatch(mats: Array[Mat], ipass: Int, here: Long): Unit = ???

  override def evalbatch(mats: Array[Mat], ipass: Int, here: Long): FMat = ???
}
