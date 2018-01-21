package BIDMach.allreduce

final case class AllReduceInputRequest(iteration: Int)

final case class AllReduceInput(data: Array[Float])

final case class AllReduceOutput(data: Array[Float], count: Array[Int], iteration: Int)

