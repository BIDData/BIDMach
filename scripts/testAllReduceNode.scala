import BIDMach.allreduce._

import scala.concurrent.duration._

val workerPerNodeNum = 3
val dataSize = 500000

val maxChunkSize = 20000

val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 1f)
val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

val workerConfig = WorkerConfig(workerPerNodeNum = workerPerNodeNum,
  discoveryTimeout = 5.seconds,
  threshold = threshold,
  metaData= metaData)

AllreduceNode.startUp("0", workerConfig)