package BIDMach.allreduce

import BIDMach.allreduce.AllreduceNode.{DataSink, DataSource}
import akka.actor.{Actor, ActorRef, Props}

/**
  * Generate a set of independent round workers and a line master which may or may not be active depending on the selection.
  *
  * @param source data source
  * @param sink   data sink
  */
class AllreduceDimensionNode(
                              dimensionNodeConfig: DimensionNodeConfig,
                              lineMasterConfig: LineMasterConfig,
                              workerConfig: WorkerConfig,
                              source: DataSource,
                              sink: DataSink
                            ) extends Actor with akka.actor.ActorLogging {

  val assignedDimension = dimensionNodeConfig.dim

  var roundWorkers: Array[ActorRef] = Array.empty
  var lineMaster: Option[ActorRef] = None

  generateRoundWorkers()
  generateLineMaster()

  override def receive: Receive = {
    case _ =>
      log.error(s"\n----DimensionNode!dim=${assignedDimension}: I AM NOT SUPPOSED TO RECEIVE MSGs")
  }

  protected def workerClassProvider(): Class[_] = {
    classOf[AllreduceWorker]
  }

  def generateRoundWorkers(): Unit = {

    roundWorkers = {
      val arr = new Array[ActorRef](lineMasterConfig.roundWorkerPerDimNum)
      for (roundNth <- 0 until lineMasterConfig.roundWorkerPerDimNum) {
        val worker = context.actorOf(Props(
          workerClassProvider(),
          workerConfig,
          source,
          sink),
          s"Worker-round=${roundNth}"
        )
        log.info(s"\n----DimensionNode!dim=${assignedDimension}: Worker for round:$roundNth created with ${worker}")
        arr(roundNth) = worker
      }
      arr
    }
  }

  def generateLineMaster(): Unit = {

    val configWithAssignedDimension = lineMasterConfig.copy(dim = assignedDimension)

    lineMaster = Some(context.actorOf(Props(
      classOf[AllreduceLineMaster],
      configWithAssignedDimension),
      name = "LineMaster"))

    log.info(s"\n----DimensionNode!dim=${assignedDimension}: LineMaster is created with ${lineMaster}")
  }
}