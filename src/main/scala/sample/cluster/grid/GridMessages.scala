package sample.cluster.grid

import akka.actor.ActorRef

//#messages
// master messages
final case class OrganizeGridWorker(text: String)
final case class GridOrganizationFailed(reason: String, job: OrganizeGridWorker)

// worker messages
final case class GridNeighborAddresses(addresses: Seq[ActorRef])
final case class HelloFromNeighbor(text: String)
case object GreetNeighbor
//#messages
