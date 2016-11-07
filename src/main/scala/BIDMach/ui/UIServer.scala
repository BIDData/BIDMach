package BIDMach.ui

import BIDMat.Mat
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach._
import BIDMach.models._

import play.api._
import play.api.routing.Router
import play.api.routing._
import play.api.routing.sird._
import play.api.mvc._
import play.core.server.{ServerConfig, NettyServer}
import play.api.mvc._
import play.api.libs.iteratee._
import play.api.libs.concurrent.Execution.Implicits.defaultContext

/**
 * The UIServer class for BIDMach Engine
 * It accepts a Learner and an Options as parameter and create the server for Interactive Machine Learning
 * 
 * 
 * */

class UIServer(nn: Learner,opts: BIDMat.Opts) extends LocalWebServer{
    override def routes = {
        case GET(p"/")=>
          controllers.Assets.at(path="/public", file="index.html")
        case GET(p"/assets/$file*")=>
          controllers.Assets.at(path="/public", file=file)
        case GET(p"/socket")=>
           WebSocket.using[String] { 
               request => 
           
                // Concurrent.broadcast returns (Enumerator, Concurrent.Channel)
                val (out, channel) = Concurrent.broadcast[String]

                // log the message to stdout and send response back to client
                val in = Iteratee.foreach[String] {
                  msg =>
                    println(msg)
                    // the Enumerator returned by Concurrent.broadcast subscribes to the channel and will
                    // receive the pushed messages
                    channel push("I received your message: " + msg)
                }
                (in,out)
              }
    }
    
}

object UIServer {
    def start(nn: Learner,opts: BIDMat.Opts) {
        Mat.checkMKL
        Mat.checkCUDA
        val server = new UIServer(nn,opts)
    }
    def main(args: Array[String]) {
        val (nn,opts) = KMeans.learner(rand(100,10000))
        start(nn,opts)
    }
}