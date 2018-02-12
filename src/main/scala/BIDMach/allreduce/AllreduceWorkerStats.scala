package BIDMach.allreduce

import BIDMach.allreduce.AllreduceNode.{DataSink, DataSource}
import BIDMach.allreduce.ReceivePipeline.{HandledCompletely, Inner}
import akka.actor.ActorRef

import scala.collection.mutable


trait StatsReporting extends ReceivePipeline {

  val scatterInCount: mutable.HashMap[Int, Int] = mutable.HashMap[Int, Int]()
  val reducedInCount: mutable.HashMap[Int, Int] = mutable.HashMap[Int, Int]()

  val scatterOutCount: mutable.HashMap[Int, Int] = mutable.HashMap[Int, Int]()
  val reducedOutCount: mutable.HashMap[Int, Int] = mutable.HashMap[Int, Int]()

  def reportingFrequency: Int

  def sendAndCount(recipient: ActorRef, msg: Any) = {
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

      if (start.round % reportingFrequency == 0 && start.round > reportingFrequency) {
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

class AllreduceWorkerWithStats(config: WorkerConfig, dataSource: DataSource, dataSink: DataSink) extends
  AllreduceWorker(config: WorkerConfig, dataSource: DataSource, dataSink: DataSink) with StatsReporting {

  override def reportingFrequency: Int = config.statsReportingRoundFrequency

  override def sendTo(recipient: ActorRef, msg: Any): Unit = {
    sendAndCount(recipient, msg)
  }
}


trait StatsAggregating extends ReceivePipeline {

  var tic = System.currentTimeMillis()
  var outgoingFloats: Long = 0
  var incomingFloats: Long = 0

  def dim: Int

  private def throughputStats(floatSize: Long, secondElapsed: Double) = {
    val bytes = floatSize * 4.0
    val mBytes = bytes / 1.0e6
    (mBytes, mBytes / secondElapsed)
  }

  pipelineInner {
    case stats: AllreduceStats =>
      outgoingFloats += stats.outgoingFloats
      incomingFloats += stats.incomingFloats

      val secondElapsed = (System.currentTimeMillis() - tic) / 1.0e3

      if (secondElapsed >= 10) {

        val (outgoingMbytes, outgoingThroughput) = throughputStats(outgoingFloats, secondElapsed)
        val (incomingMbytes, incomingThroughput) = throughputStats(incomingFloats, secondElapsed)

        val reportOut = f"Dim$dim: Outgoing $outgoingMbytes%2.1f Mbytes in $secondElapsed%2.1f seconds at $outgoingThroughput%4.3f MBytes/sec"
        val reportIn = f"Dim$dim: Incoming $incomingMbytes%2.1f Mbytes in $secondElapsed%2.1f seconds at $incomingThroughput%4.3f MBytes/sec"
        println(s"----$reportOut\n----$reportIn")
        outgoingFloats = 0
        incomingFloats = 0
        tic = System.currentTimeMillis()
      }

      HandledCompletely
  }
}

class AllreduceDimensionNodeWithStats(
                                       dimensionNodeConfig: DimensionNodeConfig,
                                       lineMasterConfig: LineMasterConfig,
                                       workerConfig: WorkerConfig,
                                       sources: List[DataSource],
                                       sinks: List[DataSink]
                                     ) extends
  AllreduceDimensionNode(
    dimensionNodeConfig: DimensionNodeConfig,
    lineMasterConfig: LineMasterConfig,
    workerConfig: WorkerConfig,
    sources: List[DataSource],
    sinks: List[DataSink]) with StatsAggregating {

  override def workerClassProvider() = {
    classOf[AllreduceWorkerWithStats]
  }
}