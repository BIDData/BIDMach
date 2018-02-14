package BIDMach.allreduce

import BIDMach.allreduce.AllreduceNode.{DataSink, DataSource}
import akka.actor.{Actor, ActorRef, Props}


class AllreduceDimensionNode(
                              dimensionNodeConfig: DimensionNodeConfig,
                              lineMasterConfig: LineMasterConfig,
                              workerConfig: WorkerConfig,
                              roundSources: List[DataSource],
                              roundSinks: List[DataSink]
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
      for (i <- 0 until lineMasterConfig.roundWorkerPerDimNum) {
        val worker = context.actorOf(Props(
          workerClassProvider(),
          workerConfig,
          roundSources(i),
          roundSinks(i)),
          name = s"Worker-id=${i}"
        )
        log.info(s"\n----DimensionNode!dim=${assignedDimension}: Worker for round:$i created with ${worker}")
        arr(i) = worker
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