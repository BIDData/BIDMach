package BIDMach.allreduce

import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import akka.testkit.{ImplicitSender, TestKit}
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec, WordSpecLike}

class AllreduceTrainerSpec extends TestKit(ActorSystem("MySpec")) with ImplicitSender
  with WordSpecLike with Matchers with BeforeAndAfterAll {
  val learner = AllreduceTrainer.leNetModel()
  val trainer = system.actorOf(
    Props(
      classOf[AllreduceTrainer],
      learner
    ),
    name = "trainer"
  )
  "trainer" must {
    "train" in{
      trainer ! StartTraining
      expectNoMsg()
    }
  }
}
