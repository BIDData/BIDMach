import BIDMach.allreduce.{AllreduceLineMaster, MasterConfig, MetaDataConfig, ThresholdConfig}
import scala.concurrent.duration._

val nodeNum = 1
val workerNum = 3
val dataSize = 100

val maxChunkSize = 4

val maxRound = 100


val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 0.8f)
val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)
val masterConfig = MasterConfig(nodeNum = nodeNum, workerNum = workerNum, maxRound,
  discoveryTimeout = 5.seconds,
  threshold = threshold,
  metaData= metaData)

AllreduceLineMaster.startUp("2551", threshold, metaData, masterConfig = masterConfig)