import BIDMach.allreduce.AllreduceMaster
import BIDMach.allreduce.ThresholdConfig
import BIDMach.allreduce.DataConfig
import BIDMach.allreduce.WorkerConfig


val threshold = ThresholdConfig(
  thAllreduce = 1,
  thReduce = 1,
  thComplete = 1
)

val dataConfig = DataConfig(
  dataSize = 150,
  maxChunkSize = 2,
  maxRound = 10000
)

val workerConfig = WorkerConfig(
  totalSize = 4,
  maxLag = 2
)

AllreduceMaster.startUp("2551", threshold, dataConfig, workerConfig)