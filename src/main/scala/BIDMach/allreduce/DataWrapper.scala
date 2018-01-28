package BIDMach.allreduce

case class AllReduceInputRequest(iteration: Int)

case class AllReduceInput(data: Array[Float])

case class AllReduceOutput(data: Array[Float], count: Array[Int], iteration: Int)

