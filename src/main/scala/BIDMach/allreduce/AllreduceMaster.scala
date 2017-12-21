package BIDMach.allreduce

import akka.Done
import akka.actor.{Actor, ActorRef, ActorSystem, Props, RootActorPath, Terminated}
import akka.cluster.ClusterEvent.MemberUp
import akka.cluster.{Cluster, Member}
import akka.util.Timeout
import com.typesafe.config.ConfigFactory

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.concurrent.duration._
import scala.language.postfixOps

class AllreduceMaster(
                       totalWorkers : Int,
                       thAllreduce : Float,
                       thReduce : Float,
                       thComplete : Float,
                       maxLag : Int,
                       dataSize: Int,
                       maxRound: Int,
                       maxChunkSize: Int
                     ) extends Actor with akka.actor.ActorLogging{

  var workers = Map[Int, ActorRef]()
  val cluster = Cluster(context.system)

  var round = -1
  var numComplete = 0

  override def preStart(): Unit = cluster.subscribe(self, classOf[MemberUp])

  override def postStop(): Unit = cluster.unsubscribe(self)

  def receive = {

    case MemberUp(m) =>
      log.info(s"\n----Detect member ${m.address} up")
      register(m).onSuccess {
        case Done =>
          if (workers.size >= totalWorkers && round == -1) {
            println(s"----${workers.size} (out of ${totalWorkers}) workers are up")
            initWorkers()
            round = 0
            startAllreduce()
          }
      }

    case Terminated(a) =>
      log.info(s"\n----$a is terminated, removing it from the set")
      for ((idx, worker) <- workers){
        if(worker == a) {
          workers -= idx
        }
      }

    case c : CompleteAllreduce =>
      log.debug(s"\n----Node ${c.srcId} completes allreduce round ${c.round}")
      if (c.round == round) {
        numComplete += 1
        if (numComplete >= totalWorkers * thAllreduce && round < maxRound) {
          log.info(s"----${numComplete} (out of ${totalWorkers}) workers complete round ${round}\n")
          round += 1
          startAllreduce()
        }
      }
  }

  private def register(member: Member): Future[Done] =
    if (member.hasRole("worker")) {
      implicit val timeout = Timeout(5.seconds)
      context.actorSelection(RootActorPath(member.address) / "user" / "worker").resolveOne().map { workerRef =>
        context watch workerRef
        val newId: Integer = workers.size
        workers = workers.updated(newId, workerRef)
        log.info(s"\n----current size = ${workers.size}")
        Done
      }
    } else {
      Future.successful(Done)
    }

  private def initWorkers() = {
    for ((idx, worker) <- workers) {
      log.info(s"\n----Init worker $idx $worker")
      worker ! InitWorkers(workers, totalWorkers, self, idx, thReduce, thComplete, maxLag, dataSize, maxChunkSize)
    }
  }

  private def startAllreduce() = {
    println(s"\n----Start allreduce round ${round}")
    numComplete = 0
    for ((_, worker) <- workers) {
      worker ! StartAllreduce(round)
    }
  }
}



object AllreduceMaster {
  def main(args: Array[String]): Unit = {
    // Override the configuration of the port when specified as program argument
    
    val maxLag = 1
    val maxRound = 100
    
    val port = if (args.isEmpty) "2551" else args(0)
    val totalWorkers = if (args.length <= 1) 2 else args(1).toInt
    val dataSize = if (args.length <= 2) totalWorkers * 5 else args(2).toInt
    val maxChunkSize = if (args.length <= 3) 2 else args(3).toInt


    val thresholds = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 0.8f)
    val dataConfig = DataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize, maxRound=maxRound)
    val workerConfig = WorkerConfig(totalSize = totalWorkers, maxLag = maxLag)

    initMaster(port, thresholds, dataConfig, workerConfig)
  }

  private def initMaster(port: String, thresholds: ThresholdConfig, dataConfig: DataConfig, workerConfig: WorkerConfig) = {
    val config = ConfigFactory.parseString(s"\nakka.remote.netty.tcp.port=$port").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [master]")).
      withFallback(ConfigFactory.load())

    val system = ActorSystem("ClusterSystem", config)

    system.log.info(s"-------\n Port = ${port} \n Number of Workers = ${workerConfig.totalSize} \n Message Size = ${dataConfig.dataSize} \n Max Chunk Size = ${dataConfig.maxChunkSize}");
    system.actorOf(
      Props(
        classOf[AllreduceMaster],
        workerConfig.totalSize,
        thresholds.thAllreduce,
        thresholds.thReduce,
        thresholds.thComplete,
        workerConfig.maxLag,
        dataConfig.dataSize,
        dataConfig.maxRound,
        dataConfig.maxChunkSize
      ),
      name = "master"
    )
  }

  def startUp() = {
    main(Array())
  }

  def startUp(port: String, thresholds: ThresholdConfig, dataConfig: DataConfig, workerConfig: WorkerConfig): Unit = {
    initMaster(port: String, thresholds: ThresholdConfig, dataConfig: DataConfig, workerConfig: WorkerConfig)
  }

}

case class ThresholdConfig(thAllreduce: Float, thReduce: Float, thComplete: Float)
case class DataConfig(dataSize: Int, maxChunkSize: Int, maxRound: Int)
case class WorkerConfig(totalSize: Int, maxLag: Int)