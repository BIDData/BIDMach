package BIDMach.viz

import java.io.File

import akka.actor.{ActorSystem,ActorRef}
import akka.http.scaladsl.Http
import akka.http.scaladsl.Http.ServerBinding
import akka.http.scaladsl.model._
import akka.http.scaladsl.server.Directives._
import akka.stream.ActorMaterializer
import akka.http.scaladsl.model.ws._//{ TextMessage, Message }
import akka.util.ByteString;
import akka.stream.scaladsl.{ Source, Flow,Sink }
import akka.http.scaladsl._
import java.util.logging.Logger;
import akka.stream._
import scala.concurrent.Future
import scala.collection.mutable.ListBuffer

import scala.concurrent.duration._    

/**
  * A simple lightweight web server built upon Akka http framework 
  * http://doc.akka.io/docs/akka-http/current/scala/http/index.html
  * Please override the routes before using it
  * The web sever will be running as soon as you instantiate the class.
  * See the examples below in object WebServer, supporting WebSocket and static file serving
  **/
    
class WebServer(val port:Int = 8888, val callback: String=>Any = println(_)) {
    
    implicit val system = ActorSystem("BID-System")
    implicit val materializer = ActorMaterializer()
    implicit val executionContext = system.dispatcher
    System.setProperty("java.util.logging.SimpleFormatter.format", "%1$tH:%1$tM:%1$tS %4$s: %5$s%n");
    val logger = Logger.getLogger("BID-WebServer console logging")

    //Simple Websocket handler
    //See more WebSocket examples at http://doc.akka.io/docs/akka-http/10.0.9/scala/http/websocket-support.html
    val greeterWebSocketService =
      Flow[Message]
        .collect {
          case tm: TextMessage => TextMessage(Source.single("Hello ") ++ tm.textStream)
        }
    

    // A more complex WebSocket handler 
    // See reference https://github.com/johanandren/chat-with-akka-http-websockets/blob/akka-2.4.9/src/main/scala/chat/Server.scala
    // Use Flow, Source and Sink http://doc.akka.io/docs/akka-stream-and-http-experimental/1.0/scala/stream-flows-and-basics.html

    var senders = ListBuffer[ActorRef]()
            
    def handler(): Flow[Message, Message, _] = {
      val incomingMessages: Sink[Message,_] =
          Sink.foreach[Message](x=>x match {
                          case TextMessage.Strict(text)=>{
                              //println(text);
                              val reply = callback(text)
                              senders.foreach(_ ! reply)
                          }
                          case _ =>  {}
      })

      val outgoingMessages: Source[Message, _] =
        Source.actorRef[Any](10, OverflowStrategy.fail)
        .mapMaterializedValue { outActor =>
          // Save the actor so that we could send messages out
          senders += outActor
        }.map(
          // transform domain message to web socket message
          {
              case (outMsg: String) => TextMessage(outMsg)
              case (outMsg:Int) => TextMessage(outMsg.toString)
              case _ => BinaryMessage(ByteString("abcd"))
          }
      )
      // then combine both to a flow
      Flow.fromSinkAndSource(incomingMessages, outgoingMessages)
    }
    
    val baseDir = "src/main/resources/viz/"
    //Default route rules
    val route = 
      path("") {
        get {
            getFromFile(baseDir + "index.html")
            //getFromBrowseableDirectories("./")
        }
      }~
      path("action" / Remaining){ 
        str=>complete(callback(str).toString)
      }~
      path(Remaining) { id=> {
                            getFromFile(baseDir + id)
                       }
                      }~
      path("ws") {
        get {
          handleWebSocketMessages(handler)
        }
      }
            
    def findPort(startPort: Int): Int = {
        var p = startPort
        while (p<65536) {
            try {
               val socket = new java.net.ServerSocket(p)
               socket.close()
               return p
            }
            catch {
                case _:Throwable =>
            }
            p+=1
        }
        0
    }
    
    val availablePort = findPort(port)
    
    val bindingFuture = Http().bindAndHandle(route, "0.0.0.0", availablePort)
    logger.info("Server started at 0.0.0.0:" + availablePort)
  
  def send(msg: String) {
        senders.foreach(_ ! msg)
  }
        
  def close() {
      bindingFuture
          .flatMap(_.unbind()) // trigger unbinding from the port
          .onComplete(_ => system.terminate()) // and shutdown when done
      logger.info("Server closed")
  }
}
    
object WebServer {
  def main(args: Array[String]){
      val s = new WebServer
  }
}
