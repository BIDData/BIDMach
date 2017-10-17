package BIDMach.allreduce

import akka.actor.ActorRef


// worker messages
final case class GridGroupAddresses(group: GridGroup, addresses: Set[ActorRef])
final case class HelloFromGroup(text: String)
final case class GreetGroups()
