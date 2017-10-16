package BIDMach.allreduce

import akka.actor.ActorRef

//#messages
// master messages
final case class OrganizeGridWorker(text: String)
final case class GridOrganizationFailed(reason: String, job: OrganizeGridWorker)

// worker messages
//final case class GridNeighborAddresses(addresses: Seq[ActorRef])
final case class GridGroupAddresses(group: GridGroup, addresses: Set[ActorRef])
final case class HelloFromGroup(text: String)
case class GreetGroups()
//#messages
