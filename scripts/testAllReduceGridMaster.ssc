import BIDMach.allreduce._

import scala.concurrent.duration._

// Override the configuration of the port when specified as program argument
val port = "2551"
val nodeNum = 16
val masterConfig = GridMasterConfig(nodeNum = nodeNum, nodeResolutionTimeout = 5.seconds)

AllreduceGridMaster.startUp(port, masterConfig)