package BIDMach.allreduce

import akka.actor.{Actor, ActorRef, ActorSystem, Props, RootActorPath, Terminated}
import akka.cluster.ClusterEvent.MemberUp
import akka.cluster.{Cluster, Member}
import com.typesafe.config.ConfigFactory

import scala.concurrent.{Await, Future}
import scala.concurrent.duration._
import scala.language.postfixOps
import scala.concurrent.ExecutionContext.Implicits.global

case class MasterConfig(
						nodeNum: Int, 
						discoveryTimeout: FiniteDuration)

case class ThresholdConfig(
						thAllreduce: Float, 
						thReduce: Float, 
						thComplete: Float)

case class MetaDataConfig(
						dataSize: Int, 
						maxChunkSize: Int)

case class LineMasterConfig(
														 workerPerNodeNum: Int,
														 dim: Int,
														 maxRound: Int,
														 discoveryTimeout: FiniteDuration,
														 threshold: ThresholdConfig,
														 metaData: MetaDataConfig)

case class NodeConfig(dimNum: Int)

case class WorkerConfig(
						discoveryTimeout: FiniteDuration,
                        threshold: ThresholdConfig,
                        metaData: MetaDataConfig)

case class DimensionNodeConfig(
						dim: Int)