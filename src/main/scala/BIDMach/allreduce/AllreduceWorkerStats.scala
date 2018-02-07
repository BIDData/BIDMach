package BIDMach.allreduce

import BIDMach.allreduce.ReceivePipeline.{HandledCompletely, Inner}
import akka.actor.ActorRef

import scala.collection.mutable


trait AllreduceWorkerStatsReporting extends ReceivePipeline {

  val scatterInCount: mutable.HashMap[Int, Int] = mutable.HashMap[Int, Int]()
  val reducedInCount: mutable.HashMap[Int, Int] = mutable.HashMap[Int, Int]()

  val scatterOutCount: mutable.HashMap[Int, Int] = mutable.HashMap[Int, Int]()
  val reducedOutCount: mutable.HashMap[Int, Int] = mutable.HashMap[Int, Int]()

  val checkPoint = 10

  def workerId: Int

  def sendTo(recipient: ActorRef, msg: Any) = {
    recipient ! msg
    msg match {
      case s: ScatterBlock => incr(scatterOutCount, s.value.size)
      case r: ReduceBlock => incr(reducedOutCount, r.value.size)
      case _ => Unit
    }
  }

  pipelineInner {

    case s: ScatterBlock => {
      incr(scatterInCount, s.value.size)
      Inner(s)
    }
    case r: ReduceBlock => {
      incr(reducedInCount, r.value.size)
      Inner(r)
    }
    case start: StartAllreduce => {

      if (start.round % checkPoint == 0 && start.round > checkPoint) {
        val totalFloatsOut: Long = aggrCount(scatterOutCount) + aggrCount(reducedOutCount)
        val totalFloatsIn: Long = aggrCount(scatterInCount) + aggrCount(reducedInCount)
        context.parent ! AllreduceStats(totalFloatsOut, totalFloatsIn)
        reset(scatterInCount)
        reset(reducedInCount)
        reset(reducedOutCount)
        reset(scatterOutCount)
      }
      Inner(start)
    }
  }

  private def incr(counter: mutable.HashMap[Int, Int], key: Int): Unit = {
    counter.update(key, counter.getOrElse(key, 0) + 1)
  }

  private def reset(counter: mutable.HashMap[Int, Int]): Unit = {
    counter.clear()
  }

  private def aggrCount(counter: mutable.HashMap[Int, Int]): Long = {
    counter.map(kv => kv._1 * kv._2).sum
  }

}


trait AllreduceWorkerStatsAggregator extends ReceivePipeline {

  var count = 0
  var tic = System.currentTimeMillis()
  var outgoingFloats: Long = 0
  var incomingFloats: Long = 0

  def throughputStats(floatSize: Long, secondElapsed: Double) = {
    val bytes = floatSize * 4.0
    val mBytes = bytes / 1.0e6
    (mBytes, mBytes / secondElapsed)
  }

  def dim: Int

  pipelineInner {
    case stats: AllreduceStats =>
      outgoingFloats += stats.outgoingFloats
      incomingFloats += stats.incomingFloats

      if (count % 100 == 0 && count > 100) {

        val timeElapsed = (System.currentTimeMillis() - tic) / 1.0e3

        val (outgoingMbytes, outgoingThroughput) = throughputStats(outgoingFloats, timeElapsed)
        val (incomingMbytes, incomingThroughput) = throughputStats(incomingFloats, timeElapsed)

        val reportOut = f"Dim$dim: Outgoing $outgoingMbytes%2.1f Mbytes in $timeElapsed%2.1f seconds at $outgoingThroughput%4.3f MBytes/sec"
        val reportIn = f"Dim$dim: Incoming $incomingMbytes%2.1f Mbytes in $timeElapsed%2.1f seconds at $incomingThroughput%4.3f MBytes/sec"
        println(s"----$reportOut\n----$reportIn")
        count = 0
        outgoingFloats = 0
        incomingFloats = 0
        tic = System.currentTimeMillis()
      }

      count += 1

      HandledCompletely
  }


}