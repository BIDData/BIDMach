package BIDMach.allreduce

import akka.actor.{ActorRef, ActorSystem, Props}
import akka.testkit.{ImplicitSender, TestKit}
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpecLike}

import scala.util.Random

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
      r.data.toList shouldBe expectedOutput(pos)
      r.count.toList shouldBe expectedCount(pos)
    }
  }

  "Flushed output of all reduce" must {

    val idx = 1
    val thReduce = 1f
    val thComplete = 1f
    val maxLag = 5
    val dataSize = 3
    val maxMsgSize = 2
    val workerNum = 2

    "reduce data over two completions" in {

      val generator = (idx: Int, iter: Int) => idx + iter.toFloat
      val source = createCustomDataSource(dataSize)(generator)

      val output1 = Array.range(0, dataSize).map(generator(_, 0)).map(_ * workerNum).toList
      val output2 = Array.range(0, dataSize).map(generator(_, 1)).map(_ * workerNum).toList

      val counts = List(2, 2, 2)

      // sink assertion
      val sink = assertiveDataSink(
        expectedOutput = List(output1, output2),
        expectedCount = List(counts, counts),
        List(0, 1))


      val worker = createNewWorker(source, sink)

      // Replace test actor with the worker itself, it can actually send message to self - not intercepted by testactor
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum).updated(idx, worker)

      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxMsgSize)

      worker ! StartAllreduce(0)
      worker ! ScatterBlock(Array(2f), srcId = 0, destId = 1, chunkId = 0, round=0)
      worker ! ReduceBlock(Array(0f, 2f), srcId = 0, destId = 1, chunkId =0, round=0, count=2)

      fishForMessage() {
        case CompleteAllreduce(1, 0) => true
        case _ => false
      }

      worker ! StartAllreduce(1)
      worker ! ScatterBlock(Array(3f), srcId = 0, destId = 1, chunkId = 0, round=1)
      worker ! ReduceBlock(Array(2f, 4f), srcId = 0, destId = 1, chunkId =0, round=1, count=2)

      fishForMessage() {
        case CompleteAllreduce(1, 1) => true
        case _ => false
      }
    }

  }

  "Early receiving reduce" must {

    val idx = 0
    val thReduce = 1f
    val thComplete = 0.8f
    val maxLag = 5
    val dataSize = 8
    val maxMsgSize = 2
    val workerNum = 4

    val worker = createNewWorker(source, sink)
    val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum).updated(idx, worker)

    val futureRound = 2

    "trigger scatter of round up to reduced message" in {

      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxMsgSize)

      worker ! ReduceBlock(Array(11f, 10f), 1, 0, 0, futureRound, count = 4)

      val scatters = receiveWhile() {
        case s: ScatterBlock => s
      }

      val scatterByRounds: Map[Int, Seq[ScatterBlock]] = scatters.groupBy(_.round)

      for (round <- 0 to futureRound) {
        scatterByRounds(round).size shouldBe workerNum - 1
      }

    }

    "send complete to master when reaching threshold" in {

      worker ! ReduceBlock(Array(10f, 20f), 2, 0, 0, futureRound, count = 4)
      worker ! ReduceBlock(Array(9f, 10f), 3, 0, 0, futureRound, count = 4)

      fishForMessage() {
        case c: CompleteAllreduce => {
          c.round shouldBe futureRound
          c.srcId shouldBe 0
          true
        }
        case s: ScatterBlock => {
          s.round shouldBe futureRound
          s.srcId shouldBe idx
          s.destId shouldNot be (idx)
          false
        }
      }
    }

    "no longer process scatter for early received round" in {

      for (i <- (idx + 1) until workerNum) {
        worker ! ScatterBlock(Array(2f, 2f), srcId = i, destId = idx, chunkId = 0, futureRound)
      }
      expectNoMsg()
    }

    "still process scatter for earlier rounds" in {

      val earlierRound = futureRound - 1

      for (i <- (idx + 1) until workerNum) {
        worker ! ScatterBlock(Array(2f, 2f), srcId = i, destId = idx, chunkId = 0, earlierRound)
      }

      val reduced = receiveWhile() {
        case r: ReduceBlock => r
      }

      reduced.map(_.round) shouldEqual Array.fill(workerNum - 1)(earlierRound).toList

    }
  }

  "Worker sending msg pattern" should {

    val idx = 1
    val thReduce = 1f
    val thComplete = 1f
    val maxLag = 5
    val dataSize = 8
    val maxMsgSize = 2
    val workerNum = 4

    val worker = createNewWorker(source, sink)
    val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum).updated(idx, worker)


    "only send one scatter to the other live peer" in {

      val theOtherPeerId = 0
      val incompleteWorkers = Map(theOtherPeerId -> workers(theOtherPeerId), idx -> workers(idx))

      worker ! InitWorkers(incompleteWorkers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxMsgSize)
      worker ! StartAllreduce(0)

      expectScatter(ScatterBlock(Array(0f, 1f), srcId = idx, destId = theOtherPeerId, 0, round=0))
      expectNoMsg()
    }

    "send all scatters after all peers joined starting from peers to the right" in {

      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxMsgSize)
      worker ! StartAllreduce(1)

      val peerList = List(2, 3, 0)
      for (whichPeer <- peerList) {
        expectScatter(ScatterBlock(Array(2f * whichPeer + 1, 2f * whichPeer + 2), srcId = idx, destId = whichPeer, 0, round = 1))
      }
    }

    "send reduced block starting from peers to the right" in {

      val peerList = List(2, 3, 0)
      for (whichPeer <- peerList) {
        worker ! ScatterBlock(Array(3f, 4f), srcId = whichPeer, destId = idx, 0, round = 1)
      }


      for (whichPeer <- peerList) {
        // multiple 4 of the scatter block
        expectReduce(ReduceBlock(Array(12f, 16f), srcId = idx, destId = whichPeer, chunkId =0, round = 1, count =4))
      }

    }

  }

  "Allreduce worker" must {

    "single-round allreduce" in {
      val worker = createNewWorker(source, sink)
      val workerNum = 4

      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 1f
      val thComplete = 0.75f
      val maxLag = 5
      val dataSize = 8
      val maxChunkSize = 2

      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxChunkSize)
      println("============start normal test!===========")
      worker ! StartAllreduce(0)

      // expect scattering to other nodes
      for (i <- 0 until 4) {
        expectScatter(ScatterBlock(Array(2f * i, 2f * i + 1), srcId = 0, destId = i, 0, 0))
      }

      // simulate sending scatter from other nodes
      for (i <- 0 until 4) {
        worker ! ScatterBlock(Array(2f * i, 2f * i), srcId = i, destId = 0, 0, 0)
      }

      // expect sending reduced result to other nodes
      expectReduce(ReduceBlock(Array(12f, 12f), 0, 0, 0, 0, 4))
      expectReduce(ReduceBlock(Array(12f, 12f), 0, 1, 0, 0, 4))
      expectReduce(ReduceBlock(Array(12f, 12f), 0, 2, 0, 0, 4))
      expectReduce(ReduceBlock(Array(12f, 12f), 0, 3, 0, 0, 4))

      // simulate sending reduced block from other nodes
      worker ! ReduceBlock(Array(12f, 15f), 0, 0, 0, 0, 4)
      worker ! ReduceBlock(Array(11f, 10f), 1, 0, 0, 0, 4)
      worker ! ReduceBlock(Array(10f, 20f), 2, 0, 0, 0, 4)
      worker ! ReduceBlock(Array(9f, 10f), 3, 0, 0, 0, 4)
      expectMsg(CompleteAllreduce(0, 0))
    }

    "uneven size sending to self first" in {

      val dataSize = 3

      val worker = createNewWorker(createBasicDataSource(dataSize), sink)
      val workerNum = 2

      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 1
      val thReduce = 1
      val thComplete = 1
      val maxLag = 1
      val maxChunkSize = 1

      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxChunkSize)
      println("============start normal test!===========")
      worker ! StartAllreduce(0)

      // expect scattering to other nodes
      expectScatter(ScatterBlock(Array(2f), srcId = 1, destId = 1, 0, 0))
      expectScatter(ScatterBlock(Array(0f), srcId = 1, destId = 0, 0, 0))
      expectScatter(ScatterBlock(Array(1f), srcId = 1, destId = 0, 1, 0))

    }

    "single-round allreduce with nasty chunk size" in {

      val dataSize = 6
      val worker = createNewWorker(createBasicDataSource(dataSize), sink)
      val workerNum = 2
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 0.9f
      val thComplete = 0.8f
      val maxLag = 5
      val maxChunkSize = 2

      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxChunkSize)
      println("============start normal test!===========")
      worker ! StartAllreduce(0)

      // expect scattering to other nodes
      expectScatter(ScatterBlock(Array(0f, 1f), srcId = 0, destId = 0, 0, 0))
      expectScatter(ScatterBlock(Array(2f), srcId = 0, destId = 0, 1, 0))
      expectScatter(ScatterBlock(Array(3f, 4f), srcId = 0, destId = 1, 0, 0))
      expectScatter(ScatterBlock(Array(5f), srcId = 0, destId = 1, 1, 0))

      // // simulate sending scatter block from other nodes
      worker ! ScatterBlock(Array(0f, 1f), srcId = 0, destId = 0, 0, 0)
      worker ! ScatterBlock(Array(2f), srcId = 0, destId = 0, 1, 0)
      worker ! ScatterBlock(Array(0f, 1f), srcId = 1, destId = 0, 0, 0)
      worker ! ScatterBlock(Array(2f), srcId = 1, destId = 0, 1, 0)


      // expect sending reduced result to other nodes
      expectReduce(ReduceBlock(Array(0f, 1f), srcId = 0, destId = 0, 0, 0, 1))
      expectReduce(ReduceBlock(Array(0f, 1f), srcId = 0, destId = 1, 0, 0, 1))
      expectReduce(ReduceBlock(Array(2f), srcId = 0, destId = 0, 1, 0, 1))
      expectReduce(ReduceBlock(Array(2f), srcId = 0, destId = 1, 1, 0, 1))

      // simulate sending reduced block from other nodes
      worker ! ReduceBlock(Array(0f, 2f), 0, 0, 0, 0, 1)
      worker ! ReduceBlock(Array(4f), 0, 0, 1, 0, 1)

      worker ! ReduceBlock(Array(6f, 8f), 1, 0, 0, 0, 1)
      expectMsg(CompleteAllreduce(0, 0))
      worker ! ReduceBlock(Array(10f), 1, 0, 1, 0, 1)


    }

    "single-round allreduce with nasty chunk size contd" in {
      val dataSize = 9
      val worker = createNewWorker(createBasicDataSource(dataSize), sink)
      val workerNum = 3
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 0.7f
      val thComplete = 0.7f
      val maxLag = 5
      val maxChunkSize = 1

      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxChunkSize)
      println("============start normal test!===========")
      worker ! StartAllreduce(0)

      // expect scattering to other nodes
      expectScatter(ScatterBlock(Array(0f), srcId = 0, destId = 0, 0, 0))
      expectScatter(ScatterBlock(Array(1f), srcId = 0, destId = 0, 1, 0))
      expectScatter(ScatterBlock(Array(2f), srcId = 0, destId = 0, 2, 0))
      expectScatter(ScatterBlock(Array(3f), srcId = 0, destId = 1, 0, 0))
      expectScatter(ScatterBlock(Array(4f), srcId = 0, destId = 1, 1, 0))
      expectScatter(ScatterBlock(Array(5f), srcId = 0, destId = 1, 2, 0))
      expectScatter(ScatterBlock(Array(6f), srcId = 0, destId = 2, 0, 0))
      expectScatter(ScatterBlock(Array(7f), srcId = 0, destId = 2, 1, 0))
      expectScatter(ScatterBlock(Array(8f), srcId = 0, destId = 2, 2, 0))

      // // simulate sending scatter block from other nodes
      worker ! ScatterBlock(Array(0f), srcId = 0, destId = 0, 0, 0)
      worker ! ScatterBlock(Array(1f), srcId = 0, destId = 0, 1, 0)
      worker ! ScatterBlock(Array(2f), srcId = 0, destId = 0, 2, 0)
      worker ! ScatterBlock(Array(0f), srcId = 1, destId = 0, 0, 0)
      worker ! ScatterBlock(Array(1f), srcId = 1, destId = 0, 1, 0)
      worker ! ScatterBlock(Array(2f), srcId = 1, destId = 0, 2, 0)
      worker ! ScatterBlock(Array(0f), srcId = 2, destId = 0, 0, 0)
      worker ! ScatterBlock(Array(1f), srcId = 2, destId = 0, 1, 0)
      worker ! ScatterBlock(Array(2f), srcId = 2, destId = 0, 2, 0)


      // expect sending reduced result to other nodes
      //expectNoMsg()
      expectReduce(ReduceBlock(Array(0f), srcId = 0, destId = 0, 0, 0, 2))
      expectReduce(ReduceBlock(Array(0f), srcId = 0, destId = 1, 0, 0, 2))
      expectReduce(ReduceBlock(Array(0f), srcId = 0, destId = 2, 0, 0, 2))
      expectReduce(ReduceBlock(Array(2f), srcId = 0, destId = 0, 1, 0, 2))
      expectReduce(ReduceBlock(Array(2f), srcId = 0, destId = 1, 1, 0, 2))
      expectReduce(ReduceBlock(Array(2f), srcId = 0, destId = 2, 1, 0, 2))
      expectReduce(ReduceBlock(Array(4f), srcId = 0, destId = 0, 2, 0, 2))
      expectReduce(ReduceBlock(Array(4f), srcId = 0, destId = 1, 2, 0, 2))
      expectReduce(ReduceBlock(Array(4f), srcId = 0, destId = 2, 2, 0, 2))


      // simulate sending reduced block from other nodes
      worker ! ReduceBlock(Array(0f), srcId = 0, destId = 0, 0, 0, 2)
      worker ! ReduceBlock(Array(3f), srcId = 0, destId = 0, 1, 0, 2)
      worker ! ReduceBlock(Array(6f), srcId = 0, destId = 0, 2, 0, 2)
      worker ! ReduceBlock(Array(9f), srcId = 1, destId = 0, 0, 0, 2)
      worker ! ReduceBlock(Array(12f), srcId = 1, destId = 0, 1, 0, 2)
      worker ! ReduceBlock(Array(15f), srcId = 1, destId = 0, 2, 0, 2)
      worker ! ReduceBlock(Array(18f), srcId = 2, destId = 0, 0, 0, 2)
      expectMsg(CompleteAllreduce(0, 0))
      worker ! ReduceBlock(Array(21f), srcId = 2, destId = 0, 1, 0, 2)
      worker ! ReduceBlock(Array(24f), srcId = 2, destId = 0, 2, 0, 2)
      expectNoMsg()
    }

    "multi-round allreduce" in {
      val worker = createNewWorker(source, sink)
      val workerNum = 4
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 0.8f
      val thComplete = 0.5f
      val maxLag = 5
      val dataSize = 8
      val maxChunkSize = 2

      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxChunkSize)
      println("===============start multi-round test!==============")
      for (i <- 0 until 10) {
        worker ! StartAllreduce(i)
        expectScatter(ScatterBlock(Array(0f + i, 1f + i), 0, 0, 0, i))
        expectScatter(ScatterBlock(Array(2f + i, 3f + i), 0, 1, 0, i))
        expectScatter(ScatterBlock(Array(4f + i, 5f + i), 0, 2, 0, i))
        expectScatter(ScatterBlock(Array(6f + i, 7f + i), 0, 3, 0, i))
        worker ! ScatterBlock(Array(0f + i, 1f + i), 0, 0, 0, i)
        worker ! ScatterBlock(Array(0f + i, 1f + i), 1, 0, 0, i)
        worker ! ScatterBlock(Array(0f + i, 1f + i), 2, 0, 0, i)
        worker ! ScatterBlock(Array(0f + i, 1f + i), 3, 0, 0, i)
        expectReduce(ReduceBlock(Array(0f + 3 * i, 1f * 3 + 3 * i), 0, 0, 0, i, 3))
        expectReduce(ReduceBlock(Array(0f + 3 * i, 1f * 3 + 3 * i), 0, 1, 0, i, 3))
        expectReduce(ReduceBlock(Array(0f + 3 * i, 1f * 3 + 3 * i), 0, 2, 0, i, 3))
        expectReduce(ReduceBlock(Array(0f + 3 * i, 1f * 3 + 3 * i), 0, 3, 0, i, 3))
        worker ! ReduceBlock(Array(1f, 2f), 0, 0, 0, i, 3)
        worker ! ReduceBlock(Array(1f, 2f), 1, 0, 0, i, 3)
        expectMsg(CompleteAllreduce(0, i))
        worker ! ReduceBlock(Array(1f, 2f), 2, 0, 0, i, 3)
        worker ! ReduceBlock(Array(1f, 2f), 3, 0, 0, i, 3)
        expectNoMsg();
      }
    }

    "multi-round allreduce v2" in {
      val worker = createNewWorker(source, sink)
      val workerNum = 2
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 0.6f
      val thComplete = 0.8f
      val maxLag = 5
      val dataSize = 8
      val maxChunkSize = 2

      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxChunkSize)
      println("===============start multi-round test!==============")
      for (i <- 0 until 10) {
        worker ! StartAllreduce(i)
        expectScatter(ScatterBlock(Array(0f + i, 1f + i), 0, 0, 0, i))
        expectScatter(ScatterBlock(Array(2f + i, 3f + i), 0, 0, 1, i))
        expectScatter(ScatterBlock(Array(4f + i, 5f + i), 0, 1, 0, i))
        expectScatter(ScatterBlock(Array(6f + i, 7f + i), 0, 1, 1, i))
        worker ! ScatterBlock(Array(0f + i, 1f + i), 0, 0, 0, i)
        worker ! ScatterBlock(Array(2f + i, 3f + i), 0, 0, 1, i)
        worker ! ScatterBlock(Array(10f + i, 11f + i), 1, 0, 0, i)
        worker ! ScatterBlock(Array(12f + i, 13f + i), 1, 0, 1, i)
        expectReduce(ReduceBlock(Array(0f * 1 + 1 * i, 1f * 1 + 1 * i), 0, 0, 0, i, 1))
        expectReduce(ReduceBlock(Array(0f * 1 + 1 * i, 1f * 1 + 1 * i), 0, 1, 0, i, 1))
        expectReduce(ReduceBlock(Array(2f * 1 + 1 * i, 3f * 1 + 1 * i), 0, 0, 1, i, 1))
        expectReduce(ReduceBlock(Array(2f * 1 + 1 * i, 3f * 1 + 1 * i), 0, 1, 1, i, 1))
        worker ! ReduceBlock(Array(1f, 2f), 0, 0, 0, i, 1)
        worker ! ReduceBlock(Array(1f, 2f), 0, 0, 1, i, 1)
        worker ! ReduceBlock(Array(1f, 2f), 1, 0, 0, i, 1)
        expectMsg(CompleteAllreduce(0, i))
        worker ! ReduceBlock(Array(1f, 2f), 1, 0, 1, i, 1)
        expectNoMsg()

      }
    }

    "missed scatter" in {
      val workerNum = 4
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 0.75f
      val thComplete = 0.75f
      val maxLag = 5
      val dataSize = 4
      val maxChunkSize = 2
      val worker = createNewWorker(createBasicDataSource(dataSize), sink)

      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxChunkSize)
      println("===============start outdated scatter test!==============")
      worker ! StartAllreduce(0)
      expectScatter(ScatterBlock(Array(0f), 0, 0, 0, 0))
      expectScatter(ScatterBlock(Array(1f), 0, 1, 0, 0))
      expectScatter(ScatterBlock(Array(2f), 0, 2, 0, 0))
      expectScatter(ScatterBlock(Array(3f), 0, 3, 0, 0))
      worker ! ScatterBlock(Array(0f), 0, 0, 0, 0)
      expectNoMsg()
      worker ! ScatterBlock(Array(2f), 1, 0, 0, 0)
      expectNoMsg()
      worker ! ScatterBlock(Array(4f), 2, 0, 0, 0)
      worker ! ScatterBlock(Array(6f), 3, 0, 0, 0)

      expectReduce(ReduceBlock(Array(6f), 0, 0, 0, 0, 3))
      expectReduce(ReduceBlock(Array(6f), 0, 1, 0, 0, 3))
      expectReduce(ReduceBlock(Array(6f), 0, 2, 0, 0, 3))
      expectReduce(ReduceBlock(Array(6f), 0, 3, 0, 0, 3))
      worker ! ReduceBlock(Array(12f), 0, 0, 0, 0, 3)
      worker ! ReduceBlock(Array(11f), 1, 0, 0, 0, 3)
      worker ! ReduceBlock(Array(10f), 2, 0, 0, 0, 3)
      expectMsg(CompleteAllreduce(0, 0))
      worker ! ReduceBlock(Array(9f), 3, 0, 0, 0, 3)
      expectNoMsg()
    }

    "future scatter" in {
      val workerNum = 4
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 0.75f
      val thComplete = 0.75f
      val maxLag = 5
      val dataSize = 4
      val maxChunkSize = 2
      val worker = createNewWorker(createBasicDataSource(dataSize), sink)
      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxChunkSize)
      println("===============start missing test!==============")
      worker ! StartAllreduce(0)
      expectScatter(ScatterBlock(Array(0f), 0, 0, 0, 0))
      expectScatter(ScatterBlock(Array(1f), 0, 1, 0, 0))
      expectScatter(ScatterBlock(Array(2f), 0, 2, 0, 0))
      expectScatter(ScatterBlock(Array(3f), 0, 3, 0, 0))

      worker ! ScatterBlock(Array(2f), 1, 0, 0, 0)
      worker ! ScatterBlock(Array(4f), 2, 0, 0, 0)
      worker ! ReduceBlock(Array(11f), 1, 0, 0, 0, 3)
      worker ! ReduceBlock(Array(10f), 2, 0, 0, 0, 3)
      // two of the messages is delayed, so now stall
      worker ! StartAllreduce(1) // master call it to do round 1
      worker ! ScatterBlock(Array(2f), 1, 0, 0, 1)
      worker ! ScatterBlock(Array(4f), 2, 0, 0, 1)
      worker ! ScatterBlock(Array(6f), 3, 0, 0, 1)

      expectScatter(ScatterBlock(Array(1f), 0, 0, 0, 1))
      expectScatter(ScatterBlock(Array(2f), 0, 1, 0, 1))
      expectScatter(ScatterBlock(Array(3f), 0, 2, 0, 1))
      expectScatter(ScatterBlock(Array(4f), 0, 3, 0, 1))
      expectReduce(ReduceBlock(Array(12f), 0, 0, 0, 1, 3))
      expectReduce(ReduceBlock(Array(12f), 0, 1, 0, 1, 3))
      expectReduce(ReduceBlock(Array(12f), 0, 2, 0, 1, 3))
      expectReduce(ReduceBlock(Array(12f), 0, 3, 0, 1, 3))
      // delayed message now get there
      worker ! ScatterBlock(Array(0f), 3, 0, 0, 0)
      worker ! ScatterBlock(Array(6f), 3, 0, 0, 0) // should be outdated
      expectReduce(ReduceBlock(Array(6f), 0, 0, 0, 0, 3))
      expectReduce(ReduceBlock(Array(6f), 0, 1, 0, 0, 3))
      expectReduce(ReduceBlock(Array(6f), 0, 2, 0, 0, 3))
      expectReduce(ReduceBlock(Array(6f), 0, 3, 0, 0, 3))
      println("finishing the reduce part")
      //worker ! ReduceBlock(Array(12), 0, 0, 1)

      worker ! ReduceBlock(Array(9f), 3, 0, 0, 0, 3)
      expectMsg(CompleteAllreduce(0, 0))
      worker ! ReduceBlock(Array(11f), 1, 0, 0, 1, 3)
      worker ! ReduceBlock(Array(10f), 2, 0, 0, 1, 3)
      worker ! ReduceBlock(Array(9f), 3, 0, 0, 1, 3)
      expectMsg(CompleteAllreduce(0, 1))
    }

    "missed reduce" in {
      val workerNum = 4
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 1f
      val thComplete = 0.75f
      val dataSize = 4
      val maxChunkSize = 100
      val maxLag = 5
      val worker = createNewWorker(createBasicDataSource(dataSize), sink)

      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxChunkSize)
      println("===============start missing test!==============")
      worker ! StartAllreduce(0)
      expectScatter(ScatterBlock(Array(0f), 0, 0, 0, 0))
      expectScatter(ScatterBlock(Array(1f), 0, 1, 0, 0))
      expectScatter(ScatterBlock(Array(2f), 0, 2, 0, 0))
      expectScatter(ScatterBlock(Array(3f), 0, 3, 0, 0))
      worker ! ScatterBlock(Array(0f), 0, 0, 0, 0)
      worker ! ScatterBlock(Array(2f), 1, 0, 0, 0)
      worker ! ScatterBlock(Array(4f), 2, 0, 0, 0)
      worker ! ScatterBlock(Array(6f), 3, 0, 0, 0)
      expectReduce(ReduceBlock(Array(12f), 0, 0, 0, 0, 4))
      expectReduce(ReduceBlock(Array(12f), 0, 1, 0, 0, 4))
      expectReduce(ReduceBlock(Array(12f), 0, 2, 0, 0, 4))
      expectReduce(ReduceBlock(Array(12f), 0, 3, 0, 0, 4))
      worker ! ReduceBlock(Array(12f), 0, 0, 0, 0, 4)
      expectNoMsg()
      worker ! ReduceBlock(Array(11f), 1, 0, 0, 0, 4)
      expectNoMsg()
      worker ! ReduceBlock(Array(10f), 2, 0, 0, 0, 4)
      //worker ! ReduceBlock(Array(9), 3, 0, 0)
      expectMsg(CompleteAllreduce(0, 0))
    }

    "delayed future reduce" in {
      val workerNum = 4
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 0.75f
      val thComplete = 0.75f
      val dataSize = 4
      val maxChunkSize = 100
      val maxLag = 5
      val worker = createNewWorker(createBasicDataSource(4), sink)

      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxChunkSize)
      println("===============start delayed future reduce test!==============")
      worker ! StartAllreduce(0)
      expectScatter(ScatterBlock(Array(0f), 0, 0, 0, 0))
      expectScatter(ScatterBlock(Array(1f), 0, 1, 0, 0))
      expectScatter(ScatterBlock(Array(2f), 0, 2, 0, 0))
      expectScatter(ScatterBlock(Array(3f), 0, 3, 0, 0))

      worker ! ScatterBlock(Array(2f), 1, 0, 0, 0)
      worker ! ScatterBlock(Array(4f), 2, 0, 0, 0)
      worker ! ScatterBlock(Array(6f), 3, 0, 0, 0)
      expectReduce(ReduceBlock(Array(12f), 0, 0, 0, 0, 3))
      expectReduce(ReduceBlock(Array(12f), 0, 1, 0, 0, 3))
      expectReduce(ReduceBlock(Array(12f), 0, 2, 0, 0, 3))
      expectReduce(ReduceBlock(Array(12f), 0, 3, 0, 0, 3))
      worker ! StartAllreduce(1) // master call it to do round 1
      worker ! ScatterBlock(Array(3f), 1, 0, 0, 1)
      worker ! ScatterBlock(Array(5f), 2, 0, 0, 1)
      worker ! ScatterBlock(Array(7f), 3, 0, 0, 1)
      // we send scatter value of round 1 to peers in case someone need it
      expectScatter(ScatterBlock(Array(1f), 0, 0, 0, 1))
      expectScatter(ScatterBlock(Array(2f), 0, 1, 0, 1))
      expectScatter(ScatterBlock(Array(3f), 0, 2, 0, 1))
      expectScatter(ScatterBlock(Array(4f), 0, 3, 0, 1))
      expectReduce(ReduceBlock(Array(15f), 0, 0, 0, 1, 3))
      expectReduce(ReduceBlock(Array(15f), 0, 1, 0, 1, 3))
      expectReduce(ReduceBlock(Array(15f), 0, 2, 0, 1, 3))
      expectReduce(ReduceBlock(Array(15f), 0, 3, 0, 1, 3))
      println("finishing the reduce part")
      // assertion: reduce t would never come after reduce t+1. (FIFO of message) otherwise would fail!
      worker ! ReduceBlock(Array(11f), 1, 0, 0, 0, 3)
      worker ! ReduceBlock(Array(11f), 1, 0, 0, 1, 3)
      worker ! ReduceBlock(Array(10f), 2, 0, 0, 0, 3)
      worker ! ReduceBlock(Array(10f), 2, 0, 0, 1, 3)
      worker ! ReduceBlock(Array(9f), 3, 0, 0, 0, 3)
      worker ! ReduceBlock(Array(9f), 3, 0, 0, 1, 3)
      expectMsg(CompleteAllreduce(0, 0))
      expectMsg(CompleteAllreduce(0, 1))
    }

  }

  "Catch up when message's round is more than maximal lag" should {

    "complete old rounds with current data and start the latest" in {
      val worker = createNewWorker(source, sink)
      val workerNum = 4
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 1
      val thComplete = 1
      val maxLag = 5
      val dataSize = 8
      val maxChunkSize = 2
      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxChunkSize)
      println("===============start simple catchup test!==============")
      for (i <- 0 to maxLag) {
        worker ! StartAllreduce(i)
        expectBasicSendingScatterBlock(i)

        simulateScatterBlocksFromPeers(worker, i)
        // intentionally missing self so no completion has been made
        worker ! ReduceBlock(Array(12.0f, 12.0f), 1, 0, 0, i, 4)
        worker ! ReduceBlock(Array(12.0f, 12.0f), 2, 0, 0, i, 4)
        worker ! ReduceBlock(Array(12.0f, 12.0f), 3, 0, 0, i, 4)
      }
      expectNoMsg()

      for (catchupRound <- List(6, 7, 8)) {
        worker ! StartAllreduce(catchupRound)

        // round 0, 1, 2 are completed
        val oldCompletionRound = catchupRound - (maxLag + 1)
        expectBasicSendingReduceBlock(oldCompletionRound)
        expectMsg(CompleteAllreduce(0, oldCompletionRound))

        // start latest
        expectBasicSendingScatterBlock(catchupRound)
      }
    }


    "return zero counts doing cold-catchup and start the latest" in {
      // extreme case of catch up, only for test use
      val workerNum = 4
      val worker = createNewWorker(source, sink)
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 1
      val thComplete = 1
      val maxLag = 5
      val dataSize = 8
      val maxChunkSize = 2

      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxChunkSize)

      // trigger the catchup instantly
      worker ! StartAllreduce(10)

      // currently we send zero-filled array with zero counts
      for (i <- 0 until maxLag) {
        expectReduce(ReduceBlock(Array(0f, 0f), 0, 0, 0, i, count= 0))
        expectReduce(ReduceBlock(Array(0f, 0f), 0, 1, 0, i, count=0))
        expectReduce(ReduceBlock(Array(0f, 0f), 0, 2, 0, i, count=0))
        expectReduce(ReduceBlock(Array(0f, 0f), 0, 3, 0, i, count=0))
        expectMsg(CompleteAllreduce(0, i))
      }

      // all iterations scatter
      for (i <- 0 to 10) {
        expectBasicSendingScatterBlock(i)
      }
    }

  }


  "Sanity Check" must {

    "multi-round allreduce v3" in {
      val workerNum = 3
      val workers: Map[Int, ActorRef] = initializeWorkersAsSelf(workerNum)
      val idx = 0
      val thReduce = 0.75f
      val thComplete = 0.75f
      val dataSize = 9
      val maxChunkSize = 2
      val maxLag = 5
      val worker = createNewWorker(createBasicDataSource(dataSize), sink)

      worker ! InitWorkers(workers, workerNum, self, idx, thReduce, thComplete, maxLag, dataSize, maxChunkSize)
      println("===============start delayed future reduce test!==============")
      worker ! StartAllreduce(0)
      expectScatter(ScatterBlock(Array(0f, 1f), 0, 0, 0, 0))
      expectScatter(ScatterBlock(Array(2f), 0, 0, 1, 0))
      expectScatter(ScatterBlock(Array(3f, 4f), 0, 1, 0, 0))
      expectScatter(ScatterBlock(Array(5f), 0, 1, 1, 0))
      expectScatter(ScatterBlock(Array(6f, 7f), 0, 2, 0, 0))
      expectScatter(ScatterBlock(Array(8f), 0, 2, 1, 0))

      worker ! ScatterBlock(Array(0f, 1f), 0, 0, 0, 0)
      worker ! ScatterBlock(Array(0f, 1f), 1, 0, 0, 0)
      worker ! ScatterBlock(Array(0f, 1f), 2, 0, 0, 0)
      worker ! ScatterBlock(Array(2f), 0, 0, 1, 0)
      worker ! ScatterBlock(Array(2f), 1, 0, 1, 0)
      worker ! ScatterBlock(Array(2f), 2, 0, 1, 0)

      expectReduce(ReduceBlock(Array(0f, 2f), 0, 0, 0, 0, 2))
      expectReduce(ReduceBlock(Array(0f, 2f), 0, 1, 0, 0, 2))
      expectReduce(ReduceBlock(Array(0f, 2f), 0, 2, 0, 0, 2))
      expectReduce(ReduceBlock(Array(4f), 0, 0, 1, 0, 2))
      expectReduce(ReduceBlock(Array(4f), 0, 1, 1, 0, 2))
      expectReduce(ReduceBlock(Array(4f), 0, 2, 1, 0, 2))

      worker ! StartAllreduce(1) // master call it to do round 1
      worker ! ScatterBlock(Array(10f, 11f), 1, 0, 0, 1)
      worker ! ScatterBlock(Array(12f), 1, 0, 1, 1)
      worker ! ScatterBlock(Array(10f, 11f), 2, 0, 0, 1)
      worker ! ScatterBlock(Array(12f), 2, 0, 1, 1)

      // we send scatter value of round 1 to peers in case someone need it
      expectScatter(ScatterBlock(Array(1f, 2f), 0, 0, 0, 1))
      expectScatter(ScatterBlock(Array(3f), 0, 0, 1, 1))
      expectScatter(ScatterBlock(Array(4f, 5f), 0, 1, 0, 1))
      expectScatter(ScatterBlock(Array(6f), 0, 1, 1, 1))
      expectScatter(ScatterBlock(Array(7f, 8f), 0, 2, 0, 1))
      expectScatter(ScatterBlock(Array(9f), 0, 2, 1, 1))

      expectReduce(ReduceBlock(Array(20f, 22f), 0, 0, 0, 1, 2))
      expectReduce(ReduceBlock(Array(20f, 22f), 0, 1, 0, 1, 2))
      expectReduce(ReduceBlock(Array(20f, 22f), 0, 2, 0, 1, 2))
      expectReduce(ReduceBlock(Array(24f), 0, 0, 1, 1, 2))
      expectReduce(ReduceBlock(Array(24f), 0, 1, 1, 1, 2))
      expectReduce(ReduceBlock(Array(24f), 0, 2, 1, 1, 2))
      println("finishing the reduce part")

      // assertion: reduce t would never come after reduce t+1. (FIFO of message) otherwise would fail!
      worker ! ReduceBlock(Array(11f, 11f), 1, 0, 0, 0, 2)
      worker ! ReduceBlock(Array(11f), 1, 0, 1, 1, 2)
      worker ! ReduceBlock(Array(11f, 11f), 1, 0, 0, 1, 2)
      worker ! ReduceBlock(Array(11f), 1, 0, 1, 0, 2)
      worker ! ReduceBlock(Array(11f, 11f), 2, 0, 0, 0, 2)
      worker ! ReduceBlock(Array(11f), 2, 0, 1, 1, 2)
      expectNoMsg()
      worker ! ReduceBlock(Array(11f, 11f), 2, 0, 0, 1, 2)
      expectMsg(CompleteAllreduce(0, 1))
      worker ! ReduceBlock(Array(11f), 2, 0, 1, 0, 2)
      expectMsg(CompleteAllreduce(0, 0))

    }
  }

  private def simulateScatterBlocksFromPeers(worker: ActorRef, i: Int) = {
    worker ! ScatterBlock(Array(1.0f * (i + 1), 1.0f * (i + 1)), 1, 0, 0, i)
    worker ! ScatterBlock(Array(2.0f * (i + 1), 2.0f * (i + 1)), 2, 0, 0, i)
    worker ! ScatterBlock(Array(4.0f * (i + 1), 4.0f * (i + 1)), 3, 0, 0, i)
  }

  private def expectBasicSendingScatterBlock(i: Int) = {
    expectScatter(ScatterBlock(Array(0f + i, 1f + i), 0, 0, 0, i))
    expectScatter(ScatterBlock(Array(2f + i, 3f + i), 0, 1, 0, i))
    expectScatter(ScatterBlock(Array(4f + i, 5f + i), 0, 2, 0, i))
    expectScatter(ScatterBlock(Array(6f + i, 7f + i), 0, 3, 0, i))
  }

  private def expectBasicSendingReduceBlock(i: Int) = {
    expectReduce(ReduceBlock(Array(7.0f * (i + 1), 7.0f * (i + 1)), 0, 0, 0, i, 3))
    expectReduce(ReduceBlock(Array(7.0f * (i + 1), 7.0f * (i + 1)), 0, 1, 0, i, 3))
    expectReduce(ReduceBlock(Array(7.0f * (i + 1), 7.0f * (i + 1)), 0, 2, 0, i, 3))
    expectReduce(ReduceBlock(Array(7.0f * (i + 1), 7.0f * (i + 1)), 0, 3, 0, i, 3))
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
        s.round shouldEqual expected.round
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
        r.round shouldEqual expected.round
        r.value.toList shouldEqual expected.value.toList
        r.chunkId shouldEqual expected.chunkId
        r.count shouldEqual expected.count
    }
  }

  private def createNewWorker(source: DataSource, sink: DataSink) = {
    system.actorOf(
      Props(
        classOf[AllreduceWorker],
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