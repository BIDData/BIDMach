import BIDMach.allreduce._

import scala.concurrent.duration._

val dimNum = 2
val dataSize = 800000
val maxChunkSize = 20000
val workerPerNodeNum = 3
val maxRound = 10000

val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 1f)
val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

val nodeConfig = NodeConfig(dimNum = dimNum, reportStats = true)

val workerConfig = WorkerConfig(
  discoveryTimeout = 5.seconds,
  statsReportingRoundFrequency = 10,
  threshold = threshold,
  metaData = metaData)

val lineMasterConfig = LineMasterConfig(
  workerPerNodeNum = workerPerNodeNum,
  dim = -1,
  maxRound = maxRound,
  discoveryTimeout = 5.seconds,
  threshold = threshold,
  metaData = metaData)


AllreduceNode.startUp("0", nodeConfig,lineMasterConfig, workerConfig, assertCorrectness=false, checkpoint = 10)