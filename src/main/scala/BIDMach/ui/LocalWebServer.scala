package BIDMach.ui

import java.io.File

import play.api.ApplicationLoader.Context
import play.api._
import play.api.routing.Router
import play.api.routing._
import play.api.routing.sird._
import play.api.mvc._
import play.core.server.{ServerConfig, NettyServer}
import play.api.mvc._
import play.api.libs.iteratee._
import play.api.libs.concurrent.Execution.Implicits.defaultContext


class LocalWebServer {
  val environment = new Environment(
    new File("."),
    getClass.getClassLoader,
    play.api.Mode.Dev
  )

  val context = ApplicationLoader.createContext(environment)

  //def routes:Router.Routes = new Router.Routes 
  //if (routes == null) throw new Exception("Routes is null")
  val components = new BuiltInComponentsFromContext(context) {
    override def router: Router = Router.from{
        case GET(p"/hello/$to") => Action {
            Results.Ok(s"Hello2 $to")
        }
        case GET(p"/ws")=>
           WebSocket.using[String] { request =>

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
        case GET(p"/assets/$file*")=>
          //Assets.versioned(path="/public", file=file)
          controllers.Assets.at(path="/public", file=file)
    }
  }

  val applicationLoader = new ApplicationLoader {
    override def load(context: Context): Application = components.application
  }

  val application = applicationLoader.load(context)

  Play.start(application)

  private object CausedBy {
    def unapply(e: Throwable): Option[Throwable] = Option(e.getCause)
  }

  private def startFindPort(port:Int): (Int, NettyServer) = {
    try {
      (port, NettyServer.fromApplication(application, ServerConfig(port = Some(port))))
    } catch {
      case CausedBy(e : java.net.BindException) => {
        startFindPort(port + 1)
      }
    }
  }

  val (port, server) = startFindPort(9000)

  def stop() = server.stop()

}

object  LocalWebServer {
    def main(arg:Array[String]) {
        val a = new LocalWebServer
    }
}
