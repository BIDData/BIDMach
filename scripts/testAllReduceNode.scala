import BIDMach.allreduce._

import scala.concurrent.duration._

val dimNum = 2
val dataSize = 100
val maxChunkSize = 4
val workerPerNodeNum = 3
val maxRound = 200

val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 0.8f)
val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

val nodeConfig = NodeConfig(dimNum = dimNum)

val workerConfig = WorkerConfig(
  discoveryTimeout = 5.seconds,
  threshold = threshold,
  metaData = metaData)

val lineMasterConfig = LineMasterConfig(
  workerPerNodeNum = workerPerNodeNum,
  dim = -1,
  maxRound = maxRound,
  discoveryTimeout = 5.seconds,
  threshold = threshold,
  metaData = metaData)


AllreduceNode.startUp("0", nodeConfig,lineMasterConfig, workerConfig)