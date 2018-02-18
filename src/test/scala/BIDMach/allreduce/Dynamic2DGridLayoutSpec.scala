package BIDMach.allreduce

import org.scalatest.{Matchers, WordSpec}

class Dynamic2DGridLayoutSpec extends WordSpec with Matchers {
  "Dynamic 2D grid" should {
    var g = new Dynamic2DGridLayout()
    "Whitebox Addition" in {
      var master_config = g.currentMasterLayout()
      for (i <- 0 until 25) {
        println(s"${g} add ${i}")
        g.addNode(i)
        val new_master_config = g.currentMasterLayout()
        println(s"new master list: ${new_master_config}")
        println(s"diff: ${Dynamic2DGridLayout.calculate_difference(master_config, new_master_config)}")
        println(s"node map: ${g._node_map}")
        println("----")
        master_config = new_master_config
      }
      println(g)
    }
    "Whitebox Random Deletion" in {
      var random = new scala.util.Random(42)
      val deletion = (0 until 25).toList
      val random_deletion = random.shuffle(deletion)
      var master_config = g.currentMasterLayout()
      for (d <- random_deletion) {
        g.removeNode(d)
        val new_master_config = g.currentMasterLayout()
        println(s"${g} delete ${d}")
        println(s"new master list: ${new_master_config}")
        println(s"diff: ${Dynamic2DGridLayout.calculate_difference(master_config, new_master_config)}")
        println(s"node map: ${g._node_map}")
        println("----")
        master_config = new_master_config
      }
      print(g)
    }
    "Addition Performance" in {
      g = new Dynamic2DGridLayout((0 until 1000).toList)
    }
    "Deletion Performance" in {
      for (i <- 0 until 500) {
        g.removeNode(i)
      }
    }
  }
}
