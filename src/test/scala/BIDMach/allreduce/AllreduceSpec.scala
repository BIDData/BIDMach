package BIDMach.allreduce

import akka.actor.{ActorRef, ActorSystem, Props}
import akka.testkit.{ImplicitSender, TestKit}
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpecLike}

import scala.util.Random
import scala.concurrent.duration._

class AllReduceSpec extends TestKit(ActorSystem("MySpec")) with ImplicitSender
  with WordSpecLike with Matchers with BeforeAndAfterAll {
  override def afterAll {
    TestKit.shutdownActorSystem(system)
  }

  type DataSink = AllReduceOutput => Unit
  type DataSource = AllReduceInputRequest => AllReduceInput

  //basic setup
  val source: DataSource = createBasicDataSource(8)
  val sink: DataSink = r => {
    println(s"Data output at #${r.iteration}: ${r.data.toList}")
  }

  /**
    * Basic test source - array of integer with element-wise addition of the iteration count
    */
  def createBasicDataSource(size: Int): DataSource = {
    createCustomDataSource(size) {
      (idx: Int, iter: Int) => idx + iter.toFloat
    }
  }

  def createCustomDataSource(size: Int)(arrIdxAndIterToData: (Int, Int) => Float): DataSource = {
    req => {
      val floats = Array.range(0, size).map(arrIdxAndIterToData(_, req.iteration))
      println(s"Data source at #${req.iteration}: ${floats.toList}")
      AllReduceInput(floats)
    }
  }

  def assertiveDataSink(expectedOutput: List[List[Float]], expectedCount: List[List[Int]], iterations: List[Int]): DataSink = {
    r => {
      val pos = iterations.indexOf(r.iteration)
      pos should be >= 0
      println(s"output $r with data ${r.data.toList}")
      r.data.toList shouldBe expectedOutput(pos)
      r.count.toList shouldBe expectedCount(pos)
    }
  }

  "Flushed output of all reduce" must {
    val nodeId = 1
    val dataSize = 3
    val maxChunkSize = 2

    val workerNum = 2
    val workerPerNodeNum = 3

    val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 1f)
    val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

    val workerConfig = WorkerConfig(
      discoveryTimeout = 5.seconds,
      threshold = threshold,
      metaData = metaData)


    "reduce data over two completions" in {

      val generator = (idx: Int, iter: Int) => idx + iter.toFloat
      val source = createCustomDataSource(dataSize)(generator)

      val output1 = Array.range(0, dataSize).map(generator(_, 0)).map(_ * workerNum).toList // 0, 2, 4
      val output2 = Array.range(0, dataSize).map(generator(_, 1)).map(_ * workerNum).toList // 2, 4, 6

      val counts = List(2, 2, 2)

      // sink assertion
      val sink = assertiveDataSink(
        expectedOutput = List(output1, output2),
        expectedCount = List(counts, counts),
        List(0, 1))


      val worker = createNewWorker(workerConfig, source, sink)

      // Replace test actor with the worker itself, it can actually send message to self - not intercepted by testactor
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum).updated(nodeId, worker)
      var roundConfig = RoundConfig(0, 0, self, workers, nodeId)

      worker ! StartAllreduce(roundConfig)
      worker ! ScatterBlock(Array(2f), srcId = 0, destId = 1, chunkId = 0, roundConfig)
      worker ! ReduceBlock(Array(0f, 2f), srcId = 0, destId = 1, chunkId = 0, roundConfig, count = 2)

      fishForMessage(60.seconds) {
        case CompleteAllreduce(1, roundConfig) => true
        case _ => false
      }
      roundConfig = roundConfig.copy(round = 1)
      worker ! StartAllreduce(roundConfig)
      worker ! ScatterBlock(Array(3f), srcId = 0, destId = 1, chunkId = 0, roundConfig)
      worker ! ReduceBlock(Array(2f, 4f), srcId = 0, destId = 1, chunkId = 0, roundConfig, count = 2)

      fishForMessage(60.seconds) {
        case CompleteAllreduce(1, roundConfig) => true
        case _ => false
      }
    }

  }

  "Allreduce worker" must {

    "single-round allreduce" in {

      val idx = 0
      val thReduce = 1f
      val thComplete = 0.75f
      val workerPerNodeNum = 5
      val dataSize = 8
      val maxChunkSize = 2

      val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 1f)
      val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

      val workerConfig = WorkerConfig(
        discoveryTimeout = 5.seconds,
        threshold = threshold,
        metaData = metaData)

      val worker = createNewWorker(workerConfig, source, sink)
      val workerNum = 4

      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      var roundConfig = RoundConfig(0, 0, self, workers, idx)
      println("============start normal test!===========")
      worker ! StartAllreduce(roundConfig)

      // expect scattering to other nodes
      for (i <- 0 until 4) {
        roundConfig = roundConfig.copy(workerId = i)
        expectScatter(ScatterBlock(Array(2f * i, 2f * i + 1), srcId = 0, destId = i, 0, roundConfig))
      }

      // simulate sending scatter from other nodes
      roundConfig = roundConfig.copy(workerId = idx)
      for (i <- 0 until 4) {
        worker ! ScatterBlock(Array(2f * i, 2f * i), srcId = i, destId = 0, 0, roundConfig)
      }

      // expect sending reduced result to other nodes
      roundConfig = roundConfig.copy(workerId = 0)
      expectReduce(ReduceBlock(Array(12f, 12f), 0, 0, 0, roundConfig, 4))
      roundConfig = roundConfig.copy(workerId = 1)
      expectReduce(ReduceBlock(Array(12f, 12f), 0, 1, 0, roundConfig, 4))
      roundConfig = roundConfig.copy(workerId = 2)
      expectReduce(ReduceBlock(Array(12f, 12f), 0, 2, 0, roundConfig, 4))
      roundConfig = roundConfig.copy(workerId = 3)
      expectReduce(ReduceBlock(Array(12f, 12f), 0, 3, 0, roundConfig, 4))

      // simulate sending reduced block from other nodes
      roundConfig = roundConfig.copy(workerId = idx)
      worker ! ReduceBlock(Array(12f, 15f), 0, 0, 0, roundConfig, 4)
      worker ! ReduceBlock(Array(11f, 10f), 1, 0, 0, roundConfig, 4)
      worker ! ReduceBlock(Array(10f, 20f), 2, 0, 0, roundConfig, 4)
      worker ! ReduceBlock(Array(9f, 10f), 3, 0, 0, roundConfig, 4)
      
      roundConfig = roundConfig.copy(workerId = idx)
      expectMsg(CompleteAllreduce(0, roundConfig))
    }

    "uneven size sending to self first" in {

      val dataSize = 3
      val idx = 1
      val workerPerNodeNum = 1
      val maxChunkSize = 1

      val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 1f)
      val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

      val workerConfig = WorkerConfig(
        discoveryTimeout = 5.seconds,
        threshold = threshold,
        metaData = metaData)

      val worker = createNewWorker(workerConfig, createBasicDataSource(dataSize), sink)
      val workerNum = 2

      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      var roundConfig = RoundConfig(0, 0, self, workers, idx)

      println("============start normal test!===========")
      worker ! StartAllreduce(roundConfig)

      // expect scattering to other nodes
      roundConfig = roundConfig.copy(workerId = 1)
      expectScatter(ScatterBlock(Array(2f), srcId = 1, destId = 1, 0, roundConfig))
      roundConfig = roundConfig.copy(workerId = 0)
      expectScatter(ScatterBlock(Array(0f), srcId = 1, destId = 0, 0, roundConfig))
      expectScatter(ScatterBlock(Array(1f), srcId = 1, destId = 0, 1, roundConfig))

    }

    "single-round allreduce with nasty chunk size" in {

      val dataSize = 6

      val workerNum = 2
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val workerPerNodeNum = 5
      val maxChunkSize = 2

      val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 0.9f, thComplete = 0.8f)
      val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

      val workerConfig = WorkerConfig(
        discoveryTimeout = 5.seconds,
        threshold = threshold,
        metaData = metaData)

      val worker = createNewWorker(workerConfig, createBasicDataSource(dataSize), sink)
      var roundConfig = RoundConfig(0, 0, self, workers, idx)

      println("============start normal test!===========")
      worker ! StartAllreduce(roundConfig)

      // expect scattering to other nodes
      roundConfig = roundConfig.copy(workerId = 0)
      expectScatter(ScatterBlock(Array(0f, 1f), srcId = 0, destId = 0, 0, roundConfig))
      expectScatter(ScatterBlock(Array(2f), srcId = 0, destId = 0, 1, roundConfig))
      roundConfig = roundConfig.copy(workerId = 1)
      expectScatter(ScatterBlock(Array(3f, 4f), srcId = 0, destId = 1, 0, roundConfig))
      expectScatter(ScatterBlock(Array(5f), srcId = 0, destId = 1, 1, roundConfig))

      // // simulate sending scatter block from other nodes
      roundConfig = roundConfig.copy(workerId = 0)
      worker ! ScatterBlock(Array(0f, 1f), srcId = 0, destId = 0, 0, roundConfig)
      worker ! ScatterBlock(Array(2f), srcId = 0, destId = 0, 1, roundConfig)
      worker ! ScatterBlock(Array(0f, 1f), srcId = 1, destId = 0, 0, roundConfig)
      worker ! ScatterBlock(Array(2f), srcId = 1, destId = 0, 1, roundConfig)


      // expect sending reduced result to other nodes
      roundConfig = roundConfig.copy(workerId = 0)
      expectReduce(ReduceBlock(Array(0f, 1f), srcId = 0, destId = 0, 0, roundConfig, 1))
      roundConfig = roundConfig.copy(workerId = 1)
      expectReduce(ReduceBlock(Array(0f, 1f), srcId = 0, destId = 1, 0, roundConfig, 1))
      roundConfig = roundConfig.copy(workerId = 0)
      expectReduce(ReduceBlock(Array(2f), srcId = 0, destId = 0, 1, roundConfig, 1))
      roundConfig = roundConfig.copy(workerId = 1)
      expectReduce(ReduceBlock(Array(2f), srcId = 0, destId = 1, 1, roundConfig, 1))

      // simulate sending reduced block from other nodes
      roundConfig = roundConfig.copy(workerId = 0)
      worker ! ReduceBlock(Array(0f, 2f), 0, 0, 0, roundConfig, 1)
      worker ! ReduceBlock(Array(4f), 0, 0, 1, roundConfig, 1)

      worker ! ReduceBlock(Array(6f, 8f), 1, 0, 0, roundConfig, 1)
      roundConfig = roundConfig.copy(workerId = idx)
      expectMsg(CompleteAllreduce(0, roundConfig))
      worker ! ReduceBlock(Array(10f), 1, 0, 1, roundConfig, 1)
    }

    "single-round allreduce with nasty chunk size contd" in {
      val dataSize = 9
      val workerNum = 3
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 0.7f
      val thComplete = 0.7f
      val workerPerNodeNum = 5
      val maxChunkSize = 1

      val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = thReduce, thComplete = thComplete)
      val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

      val workerConfig = WorkerConfig(
        discoveryTimeout = 5.seconds,
        threshold = threshold,
        metaData = metaData)

      val worker = createNewWorker(workerConfig, createBasicDataSource(dataSize), sink)
      var roundConfig = RoundConfig(0, 0, self, workers, idx)

      println("============start normal test!===========")
      worker ! StartAllreduce(roundConfig)

      // expect scattering to other nodes
      roundConfig = roundConfig.copy(workerId = 0)
      expectScatter(ScatterBlock(Array(0f), srcId = 0, destId = 0, 0, roundConfig))
      expectScatter(ScatterBlock(Array(1f), srcId = 0, destId = 0, 1, roundConfig))
      expectScatter(ScatterBlock(Array(2f), srcId = 0, destId = 0, 2, roundConfig))
      roundConfig = roundConfig.copy(workerId = 1)
      expectScatter(ScatterBlock(Array(3f), srcId = 0, destId = 1, 0, roundConfig))
      expectScatter(ScatterBlock(Array(4f), srcId = 0, destId = 1, 1, roundConfig))
      expectScatter(ScatterBlock(Array(5f), srcId = 0, destId = 1, 2, roundConfig))
      roundConfig = roundConfig.copy(workerId = 2)
      expectScatter(ScatterBlock(Array(6f), srcId = 0, destId = 2, 0, roundConfig))
      expectScatter(ScatterBlock(Array(7f), srcId = 0, destId = 2, 1, roundConfig))
      expectScatter(ScatterBlock(Array(8f), srcId = 0, destId = 2, 2, roundConfig))

      // // simulate sending scatter block from other nodes
      roundConfig = roundConfig.copy(workerId = 0)
      worker ! ScatterBlock(Array(0f), srcId = 0, destId = 0, 0, roundConfig)
      worker ! ScatterBlock(Array(1f), srcId = 0, destId = 0, 1, roundConfig)
      worker ! ScatterBlock(Array(2f), srcId = 0, destId = 0, 2, roundConfig)
      worker ! ScatterBlock(Array(0f), srcId = 1, destId = 0, 0, roundConfig)
      worker ! ScatterBlock(Array(1f), srcId = 1, destId = 0, 1, roundConfig)
      worker ! ScatterBlock(Array(2f), srcId = 1, destId = 0, 2, roundConfig)
      worker ! ScatterBlock(Array(0f), srcId = 2, destId = 0, 0, roundConfig)
      worker ! ScatterBlock(Array(1f), srcId = 2, destId = 0, 1, roundConfig)
      worker ! ScatterBlock(Array(2f), srcId = 2, destId = 0, 2, roundConfig)


      // expect sending reduced result to other nodes
      //expectNoMsg()
      roundConfig = roundConfig.copy(workerId = 0)
      expectReduce(ReduceBlock(Array(0f), srcId = 0, destId = 0, 0, roundConfig, 2))
      roundConfig = roundConfig.copy(workerId = 1)
      expectReduce(ReduceBlock(Array(0f), srcId = 0, destId = 1, 0, roundConfig, 2))
      roundConfig = roundConfig.copy(workerId = 2)
      expectReduce(ReduceBlock(Array(0f), srcId = 0, destId = 2, 0, roundConfig, 2))
      roundConfig = roundConfig.copy(workerId = 0)
      expectReduce(ReduceBlock(Array(2f), srcId = 0, destId = 0, 1, roundConfig, 2))
      roundConfig = roundConfig.copy(workerId = 1)
      expectReduce(ReduceBlock(Array(2f), srcId = 0, destId = 1, 1, roundConfig, 2))
      roundConfig = roundConfig.copy(workerId = 2)
      expectReduce(ReduceBlock(Array(2f), srcId = 0, destId = 2, 1, roundConfig, 2))
      roundConfig = roundConfig.copy(workerId = 0)
      expectReduce(ReduceBlock(Array(4f), srcId = 0, destId = 0, 2, roundConfig, 2))
      roundConfig = roundConfig.copy(workerId = 1)
      expectReduce(ReduceBlock(Array(4f), srcId = 0, destId = 1, 2, roundConfig, 2))
      roundConfig = roundConfig.copy(workerId = 2)
      expectReduce(ReduceBlock(Array(4f), srcId = 0, destId = 2, 2, roundConfig, 2))


      // simulate sending reduced block from other nodes
      roundConfig = roundConfig.copy(workerId = 0)
      worker ! ReduceBlock(Array(0f), srcId = 0, destId = 0, 0, roundConfig, 2)
      worker ! ReduceBlock(Array(3f), srcId = 0, destId = 0, 1, roundConfig, 2)
      worker ! ReduceBlock(Array(6f), srcId = 0, destId = 0, 2, roundConfig, 2)
      worker ! ReduceBlock(Array(9f), srcId = 1, destId = 0, 0, roundConfig, 2)
      worker ! ReduceBlock(Array(12f), srcId = 1, destId = 0, 1, roundConfig, 2)
      worker ! ReduceBlock(Array(15f), srcId = 1, destId = 0, 2, roundConfig, 2)
      worker ! ReduceBlock(Array(18f), srcId = 2, destId = 0, 0, roundConfig, 2)
      roundConfig = roundConfig.copy(workerId = idx)
      expectMsg(CompleteAllreduce(0, roundConfig))
      worker ! ReduceBlock(Array(21f), srcId = 2, destId = 0, 1, roundConfig, 2)
      worker ! ReduceBlock(Array(24f), srcId = 2, destId = 0, 2, roundConfig, 2)
      expectNoMsg()
    }

    "missed scatter" in {
      val workerNum = 4
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 0.75f
      val thComplete = 0.75f
      val workerPerNodeNum = 5
      val dataSize = 4
      val maxChunkSize = 2

      val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = thReduce, thComplete = thComplete)
      val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

      val workerConfig = WorkerConfig(
        discoveryTimeout = 5.seconds,
        threshold = threshold,
        metaData = metaData)

      val worker = createNewWorker(workerConfig, createBasicDataSource(dataSize), sink)
      var roundConfig = RoundConfig(0, 0, self, workers, idx)

      println("===============start outdated scatter test!==============")
      worker ! StartAllreduce(roundConfig)

      roundConfig = roundConfig.copy(workerId = 0)
      expectScatter(ScatterBlock(Array(0f), 0, 0, 0, roundConfig))
      roundConfig = roundConfig.copy(workerId = 1)
      expectScatter(ScatterBlock(Array(1f), 0, 1, 0, roundConfig))
      roundConfig = roundConfig.copy(workerId = 2)
      expectScatter(ScatterBlock(Array(2f), 0, 2, 0, roundConfig))
      roundConfig = roundConfig.copy(workerId = 3)
      expectScatter(ScatterBlock(Array(3f), 0, 3, 0, roundConfig))

      roundConfig = roundConfig.copy(workerId = 0)
      worker ! ScatterBlock(Array(0f), 0, 0, 0, roundConfig)
      expectNoMsg()
      worker ! ScatterBlock(Array(2f), 1, 0, 0, roundConfig)
      expectNoMsg()
      worker ! ScatterBlock(Array(4f), 2, 0, 0, roundConfig)
      worker ! ScatterBlock(Array(6f), 3, 0, 0, roundConfig)

      roundConfig = roundConfig.copy(workerId = 0)
      expectReduce(ReduceBlock(Array(6f), 0, 0, 0, roundConfig, 3))
      roundConfig = roundConfig.copy(workerId = 1)
      expectReduce(ReduceBlock(Array(6f), 0, 1, 0, roundConfig, 3))
      roundConfig = roundConfig.copy(workerId = 2)
      expectReduce(ReduceBlock(Array(6f), 0, 2, 0, roundConfig, 3))
      roundConfig = roundConfig.copy(workerId = 3)
      expectReduce(ReduceBlock(Array(6f), 0, 3, 0, roundConfig, 3))

      roundConfig = roundConfig.copy(workerId = 0)
      worker ! ReduceBlock(Array(12f), 0, 0, 0, roundConfig, 3)
      worker ! ReduceBlock(Array(11f), 1, 0, 0, roundConfig, 3)
      worker ! ReduceBlock(Array(10f), 2, 0, 0, roundConfig, 3)
      roundConfig = roundConfig.copy(workerId = idx)
      expectMsg(CompleteAllreduce(0, roundConfig))
      worker ! ReduceBlock(Array(9f), 3, 0, 0, roundConfig, 3)
      expectNoMsg()
    }

    "missed reduce" in {
      val workerNum = 4
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 1f
      val thComplete = 0.75f
      val dataSize = 4
      val maxChunkSize = 100
      val workerPerNodeNum = 5

      val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = thReduce, thComplete = thComplete)
      val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

      val workerConfig = WorkerConfig(
        discoveryTimeout = 5.seconds,
        threshold = threshold,
        metaData = metaData)

      val worker = createNewWorker(workerConfig, createBasicDataSource(dataSize), sink)
      var roundConfig = RoundConfig(0, 0, self, workers, idx)

      println("===============start missing test!==============")
      roundConfig = roundConfig.copy(workerId = 0)
      worker ! StartAllreduce(roundConfig)

      roundConfig = roundConfig.copy(workerId = 0)
      expectScatter(ScatterBlock(Array(0f), 0, 0, 0, roundConfig))
      roundConfig = roundConfig.copy(workerId = 1)
      expectScatter(ScatterBlock(Array(1f), 0, 1, 0, roundConfig))
      roundConfig = roundConfig.copy(workerId = 2)
      expectScatter(ScatterBlock(Array(2f), 0, 2, 0, roundConfig))
      roundConfig = roundConfig.copy(workerId = 3)
      expectScatter(ScatterBlock(Array(3f), 0, 3, 0, roundConfig))

      roundConfig = roundConfig.copy(workerId = 0)
      worker ! ScatterBlock(Array(0f), 0, 0, 0, roundConfig)
      worker ! ScatterBlock(Array(2f), 1, 0, 0, roundConfig)
      worker ! ScatterBlock(Array(4f), 2, 0, 0, roundConfig)
      worker ! ScatterBlock(Array(6f), 3, 0, 0, roundConfig)

      roundConfig = roundConfig.copy(workerId = 0)
      expectReduce(ReduceBlock(Array(12f), 0, 0, 0, roundConfig, 4))
      roundConfig = roundConfig.copy(workerId = 1)
      expectReduce(ReduceBlock(Array(12f), 0, 1, 0, roundConfig, 4))
      roundConfig = roundConfig.copy(workerId = 2)
      expectReduce(ReduceBlock(Array(12f), 0, 2, 0, roundConfig, 4))
      roundConfig = roundConfig.copy(workerId = 3)
      expectReduce(ReduceBlock(Array(12f), 0, 3, 0, roundConfig, 4))

      roundConfig = roundConfig.copy(workerId = 0)
      worker ! ReduceBlock(Array(12f), 0, 0, 0, roundConfig, 4)
      expectNoMsg()
      worker ! ReduceBlock(Array(11f), 1, 0, 0, roundConfig, 4)
      expectNoMsg()
      worker ! ReduceBlock(Array(10f), 2, 0, 0, roundConfig, 4)
      //worker ! ReduceBlock(Array(9), 3, 0, 0)
      roundConfig = roundConfig.copy(workerId = idx)
      expectMsg(CompleteAllreduce(0, roundConfig))
    }

    "correctly force stop previous round" in {
      val workerNum = 2
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 1f
      val thComplete = 1f
      val dataSize = 6
      val maxChunkSize = 1
      val workerPerNodeNum = 5

      val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = thReduce, thComplete = thComplete)
      val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

      val workerConfig = WorkerConfig(
        discoveryTimeout = 5.seconds,
        threshold = threshold,
        metaData = metaData)

      val worker = createNewWorker(workerConfig, createBasicDataSource(dataSize), sink)
      var roundConfig = RoundConfig(0, 0, self, workers, idx)

      println("===============start missing test!==============")
      worker ! StartAllreduce(roundConfig)

      roundConfig = roundConfig.copy(workerId = 0)
      expectScatter(ScatterBlock(Array(0f), 0, 0, 0, roundConfig))
      expectScatter(ScatterBlock(Array(1f), 0, 0, 1, roundConfig))
      expectScatter(ScatterBlock(Array(2f), 0, 0, 2, roundConfig))
      roundConfig = roundConfig.copy(workerId = 1)
      expectScatter(ScatterBlock(Array(3f), 0, 1, 0, roundConfig))
      expectScatter(ScatterBlock(Array(4f), 0, 1, 1, roundConfig))
      expectScatter(ScatterBlock(Array(5f), 0, 1, 2, roundConfig))

      // three things are tested: 1. normal case, 2. half done 3. nothing done.
      roundConfig = roundConfig.copy(workerId = 0)
      worker ! ScatterBlock(Array(2f), 0, 0, 0, roundConfig)
      worker ! ScatterBlock(Array(3f), 1, 0, 0, roundConfig)

      roundConfig = roundConfig.copy(workerId = 0)
      expectReduce(ReduceBlock(Array(5f),0,0,0,roundConfig,2))
      roundConfig = roundConfig.copy(workerId = 1)
      expectReduce(ReduceBlock(Array(5f),0,1,0,roundConfig,2))

      roundConfig = roundConfig.copy(workerId = 0)
      worker ! ScatterBlock(Array(2f), 0, 0, 1, roundConfig)
      expectNoMsg()

      roundConfig = roundConfig.copy(round = 5, workerId = idx)
      worker ! StartAllreduce(roundConfig)

      roundConfig = roundConfig.copy(round = 0, workerId = 0)
      expectReduce(ReduceBlock(Array(2f),0,0,1,roundConfig,1))
      roundConfig = roundConfig.copy(workerId = 1)
      expectReduce(ReduceBlock(Array(2f),0,1,1,roundConfig,1))
      roundConfig = roundConfig.copy(workerId = 0)
      expectReduce(ReduceBlock(Array(0f),0,0,2,roundConfig,0))
      roundConfig = roundConfig.copy(workerId = 1)
      expectReduce(ReduceBlock(Array(0f),0,1,2,roundConfig,0))
      roundConfig = roundConfig.copy(workerId = 0)
      expectMsg(CompleteAllreduce(0,roundConfig))
    }
  }


  /**
    * Expect scatter block containing array value. This is needed because standard expectMsg will not be able to match
    * mutable Array and check for equality during assertion.
    */
  private def expectScatter(expected: ScatterBlock) = {
    receiveOne(remainingOrDefault) match {
      case s: ScatterBlock =>
        s.srcId shouldEqual expected.srcId
        s.destId shouldEqual expected.destId
        s.config shouldEqual expected.config
        s.value.toList shouldEqual expected.value.toList
        s.chunkId shouldEqual expected.chunkId
    }
  }

  /**
    * Expect reduce block containing array value. This is needed because standard expectMsg will not be able to match
    * mutable Array and check for equality during assertion.
    */
  private def expectReduce(expected: ReduceBlock) = {
    receiveOne(remainingOrDefault) match {
      case r: ReduceBlock =>
        r.srcId shouldEqual expected.srcId
        r.destId shouldEqual expected.destId
        r.config shouldEqual expected.config
        r.value.toList shouldEqual expected.value.toList
        r.chunkId shouldEqual expected.chunkId
        r.count shouldEqual expected.count
    }
  }

  private def createNewWorker(workerConfig: WorkerConfig, source: DataSource, sink: DataSink) = {

    system.actorOf(
      Props(
        classOf[AllreduceWorker],
        workerConfig,
        source,
        sink
      ),
      name = Random.alphanumeric.take(10).mkString
    )
  }

  private def initializeWorkersAsSelf(size: Int): Map[Int, ActorRef] = {

    (for {
      i <- 0 until size
    } yield (i, self)).toMap

  }

}
