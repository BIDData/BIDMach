package BIDMach.allreduce

import akka.actor.ActorRef

import scala.collection.mutable


class Dynamic2DGridLayout(nodes: List[Int]) {
  var _grid = new mutable.ArrayBuffer[mutable.ArrayBuffer[Option[Int]]]()
  _grid+=mutable.ArrayBuffer.fill(1)(Option.empty)
  var _count = 0
  var _filled = 0
  var _N = 1
  var _node_map = mutable.HashMap[Int, Tuple2[Int, Int]]()
  for (node <- nodes) {
    addNode(node)
  }

  def this(){
    this(List.empty)
  }

  private def getXNodes(x: Int): Set[Int] = {
    assert(0 <= x && x < _grid(0).length)
    (for (y <- 0 until _grid.length if _grid(y)(x).isDefined) yield _grid(y)(x).get).toSet
  }

  private def getYNodes(y: Int): Set[Int] = {
    assert(0 <= y && y < _grid.length)
    (for (x <- 0 until _grid(0).length if _grid(y)(x).isDefined) yield _grid(y)(x).get).toSet
  }

  private def setMasterNode(value: mutable.HashMap[Int, Tuple2[Set[Int], Set[Int]]], i: Int): Unit = {
    //exist to help handle n - 1's master properly
    if (i == _N - 1 && _grid.last.last.isEmpty) {
      assert(_grid.last(0).isDefined && _grid(0).last.isDefined)
      value(_grid.last(0).get) = (getYNodes(_N - 1), Set())
      value(_grid(0).last.get) = (Set(), getXNodes(_N - 1))
    } else {
      value(_grid(i)(i).get) = (getYNodes(i), getXNodes(i))
    }
  }

  private def getPos(filled: Int, n: Int): Tuple2[Int, Int] = {
    //pattern is (1,2),(2,1),(3,4),(4,3),...(n-1, n-2) -> (n-1,n-1) -> (0,1),(1,0),(2,3),(3,2)...
    val separate_loc = if (n % 2 == 1) n - 1 else n - 2
    if (filled < separate_loc) {
      val y = filled + 1
      val x = if (filled % 2 == 0) y + 1 else y - 1
      (y, x)
    } else if (filled == separate_loc) {
      (n - 1, n - 1)
    } else {
      val y = filled - (separate_loc + 1)
      val x = if (filled % 2 == 1) y + 1 else y - 1
      (y, x)
    }
  }

  def addNode(node: Int): mutable.Map[Int, Tuple2[Set[Int], Set[Int]]] = {
    //maintain grid, return update for{master_node:[col_neighbours], [row_neighbours]}} pair
    val ret = mutable.HashMap[Int, Tuple2[Set[Int], Set[Int]]]()
    if (_N * _N == _count) {
      _filled = 0
      for (i <- 0 until _N) {
        _grid(i) += Option.empty
      }
      _grid += mutable.ArrayBuffer.fill(_N + 1)(Option.empty)
      _N += 1
      for (i <- 0 until _N - 1) {
        _grid(i).update(_grid(i).length-1, _grid(i)(i + 1))
        _grid(_grid.length-1).update(i, _grid(i + 1)(i))
        _grid(i).update(i + 1, Option.empty)
        _grid(i + 1).update(i, Option.empty)
        if (_grid(i)(_grid(i).length-1).isDefined) {
          _node_map.update(_grid(i)(_grid(i).length-1).get, (i, _N - 1))
        }
        if (_grid(_grid.length-1)(i).isDefined) {
          _node_map.update(_grid(_grid.length-1)(i).get, (_N - 1, i))
        }
      }
      for (i <- 1 until _N - 1) {
        setMasterNode(ret, i)
      }
    }

    if (_N == 1) {
      _grid(0).update(0, Some(node))
      _node_map.update(node, (0, 0))
      ret.update(_grid(0)(0).get, (getYNodes(0), getXNodes(0)))
      _count += 1
    } else {
      val (y, x) = getPos(_filled, _N)
      _filled += 1
      _grid(y).update(x, Some(node))
      _node_map.update(node, (y, x))
      setMasterNode(ret, y)
      setMasterNode(ret, x)
      _count += 1
    }
    ret
  }

  private def remove(): mutable.HashMap[Int, Tuple2[Set[Int], Set[Int]]] ={
    //maintain grid, return  return update for{master_node:[col_neighbours], [row_neighbours]}} pair. literally reverse add node
    assert(_count>0)
    assert(_filled>0)
    val ret = mutable.HashMap[Int, Tuple2[Set[Int], Set[Int]]]()
    _count -=1
    if(_N==1){
      _node_map.remove(_grid(0)(0).get)
      _grid(0).update(0, Option.empty)
    }else{
      _filled-=1
      val (y,x) = getPos(_filled, _N)
      _node_map.remove(_grid(y)(x).get)
      _grid(y).update(x, Option.empty)
      if( _filled == 0){
        // compact afterwards
        for(i <-0 until _N - 1){
          _grid(i).update(i+1, _grid(i)(_grid(i).length-1))
          _grid(i+1).update(i, _grid(_grid.length-1)(i))
          _grid(i).update(_grid(i).length-1, Option.empty)
          _grid(_grid.length-1).update(i, Option.empty)
          if(_grid(i)(i+1).isDefined){
            _node_map.update(_grid(i)(i+1).get,(i,i+1))
          }
          if(_grid(i+1)(i).isDefined){
            _node_map.update(_grid(i+1)(i).get,(i+1,i))
          }
        }
        _grid.remove(_grid.size-1)
        for( i <- 0 until _N-1){
          _grid(i).remove(_grid(i).size-1)
        }
        _N-=1
        _filled = if (_N>1) 2 * _N -1 else 1
        for( i <- 1 until _N){
          setMasterNode(ret, i)
        }
      }else{
        setMasterNode(ret, y)
        setMasterNode(ret, x)
      }
    }
    ret
  }

  private def nextRemove() : Tuple2[Int, Int] ={
    assert(_count > 0)
    if(_N == 1){
      (0,0)
    }else if(_filled > 0){
      getPos(_filled-1,_N)
    }else{
      (_N - 1, getPos(2*(_N-1)-2,_N-1)._1)
    }
  }

  def removeLocation(location : Tuple2[Int, Int]): mutable.Map[Int, Tuple2[Set[Int], Set[Int]]] ={
    val remove_loc = nextRemove()
    if (remove_loc != location){
      val (ry,rx) = remove_loc
      val (y,x) = location
      val temp = _grid(ry)(rx)
      _grid(ry)(rx) = _grid(y)(x)
      _grid(y)(x) = temp
      _node_map.update(_grid(ry)(rx).get,(ry,rx))
      _node_map.update(_grid(y)(x).get, (y,x))
      val swapped_groups = Set(rx,ry,x,y)
      val new_update = remove()
      for(group <- swapped_groups){
        if(group < _N){
          setMasterNode(new_update, group)
        }
      }
      new_update
    }else{
      remove()
    }
  }

  def removeNode(node: Int): mutable.Map[Int, Tuple2[Set[Int], Set[Int]]] ={
    assert(_node_map.contains(node))
    removeLocation(_node_map(node))
  }

  override def toString: String = {
    var ret = new String()
    for(j <- 0 until _grid.length){
      ret += "["
      for(i <- 0 until _grid(0).length){
        if(_grid(j)(i).isDefined){
          ret += s"${_grid(j)(i).get} "
        }
        else{
          ret += s"_ "
        }
      }
      ret+="]\n"
    }
    ret
  }


}

object Dynamic2DGridLayout {
  def main(args: Array[String]): Unit = {
    var g = new Dynamic2DGridLayout()
    println("====== Whitebox Addition Test ======")
    for(i <- 0 until 25){
      println(s"${g} add ${i}")
      println(s"need to update on ${g.addNode(i)}")
      println(s"node map: ${g._node_map}")
      println("----")
    }
    println(g)
    println("===== Whitebox Random Deletion Test =====")
    var random = new scala.util.Random(42)
    val deletion = (0 until 25).toList
    val random_deletion = random.shuffle(deletion)
    for(d <- random_deletion){
      println(s"${g} delete ${d}")
      println(s"need to update on ${g.removeNode(d)}")
      println(s"node map: ${g._node_map}")
      println("----")
    }
    print(g)
    println("===== Addition Performance Test =====")
    g = new Dynamic2DGridLayout((0 until 10000).toList)
    println("===== Deletion Performance Test =====")
    for(i <- 0 until 5000){
      g.removeNode(i)
    }
    println("Test all Cleared.")
  }

}