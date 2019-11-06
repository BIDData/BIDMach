
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

Future {
  while (opts.resScale > 0.01f) {
    Thread.sleep(100*1000)
    opts.resScale = opts.resScale * 0.99f
  }
  opts.resScale = 0f
}
