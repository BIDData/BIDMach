import BIDMach.allreduce.AllreduceNode.getBasicConfigs
import BIDMach.allreduce.binder.{AssertCorrectnessBinder, NoOptBinder}
import BIDMach.allreduce.{AllreduceDummyLearner, AllreduceNode}

val learner = new AllreduceDummyLearner()
learner.ipass = 20

val dataSize = 50000
val maxChunkSize = 1779

val basicConfig = getBasicConfigs()
val modifiedConfig = basicConfig.copy(workerConfig =
  basicConfig.workerConfig.copy(
    metaData = basicConfig.workerConfig.metaData.copy(dataSize = dataSize, maxChunkSize = maxChunkSize),
    threshold =  basicConfig.workerConfig.threshold.copy(thComplete = 0.8f)
  )
)


val binder = new NoOptBinder(dataSize, 10)
AllreduceNode.startNodeAfterIter(learner = learner, iter = 0, nodeConfig = modifiedConfig, binder = binder)