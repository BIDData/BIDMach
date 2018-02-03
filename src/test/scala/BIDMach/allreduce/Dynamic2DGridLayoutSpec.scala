package BIDMach.allreduce

import org.scalatest.{Matchers, WordSpec}

class Dynamic2DGridLayoutSpec extends WordSpec with Matchers {
  "Dynamic 2D grid" should {
    var g = new Dynamic2DGridLayout()
    "Whitebox Addition" in {
      for (i <- 0 until 25) {
        println(s"${g} add ${i}")
        println(s"need to update on ${g.addNode(i)}")
        println(s"node map: ${g._node_map}")
        println("----")
      }
      println(g)
    }
    "Whitebox Random Deletion" in {
      var random = new scala.util.Random(42)
      val deletion = (0 until 25).toList
      val random_deletion = random.shuffle(deletion)
      for (d <- random_deletion) {
        println(s"${g} delete ${d}")
        println(s"need to update on ${g.removeNode(d)}")
        println(s"node map: ${g._node_map}")
        println("----")
      }
      print(g)
    }
    "Addition Performance" in {
      g = new Dynamic2DGridLayout((0 until 10000).toList)
    }
    "Deletion Performance" in {
      for (i <- 0 until 5000) {
        g.removeNode(i)
      }
    }
  }
}
