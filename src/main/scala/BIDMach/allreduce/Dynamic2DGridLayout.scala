package BIDMach.allreduce

import akka.actor.ActorRef

import scala.collection.mutable

/**
  * Dynamic 2D Grid data structure to maintain balanced layout. Run Dynamic2DGridLayoutSpec to check how it works.
  * @param nodes initialize nodes indexes
 */
class Dynamic2DGridLayout(nodes: List[Int]) {
  type MasterLayout = Dynamic2DGridLayout.MasterLayout
  var _grid = new mutable.ArrayBuffer[mutable.ArrayBuffer[Option[Int]]]() // grid recording the nodes layout
  _grid+=mutable.ArrayBuffer.fill(1)(Option.empty)
  var _count = 0 //how many nodes are in the grid
  var _filled = 0 //how many locations for are filled. Once fully filled grid would expand
  var _N = 1 // size of the grid, always N*N except for count = 0 case.
  var _node_map = mutable.HashMap[Int, Tuple2[Int, Int]]() // node_idx => location in grid
  for (node <- nodes) {
    addNode(node)
  }

  def this(){
    this(List.empty)
  }

  /**
    * @param x the idx specified
    * @return return all idx current with given x idx.
    * Same for the getYNodes
    */
  private def getXNodes(x: Int): Set[Int] = {
    assert(0 <= x && x < _grid(0).length)
    (for (y <- 0 until _grid.length if _grid(y)(x).isDefined) yield _grid(y)(x).get).toSet
  }

  private def getYNodes(y: Int): Set[Int] = {
    assert(0 <= y && y < _grid.length)
    (for (x <- 0 until _grid(0).length if _grid(y)(x).isDefined) yield _grid(y)(x).get).toSet
  }

  /**
    * set master node of given index
    * @param value the layout to be handled
    * @param i the idx of row(or column) to handle
    */
  private def setMasterNode(value: MasterLayout, i: Int): Unit = {
    //exist to help handle n - 1's master properly
    if (i == _N - 1 && _grid.last.last.isEmpty) {
      assert(_grid.last(0).isDefined && _grid(0).last.isDefined)
      value(_grid.last(0).get) = (Some(getYNodes(_N - 1)), Option.empty)
      value(_grid(0).last.get) = (Option.empty, Some(getXNodes(_N - 1)))
    } else {
      value(_grid(i)(i).get) = (Some(getYNodes(i)), Some(getXNodes(i)))
    }
  }

  /**
    * Helper function to decide which next free position to take
    * @param filled
    * @param n
    * @return the next position to take
    */
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

  def addNode(node: Int): Unit = {
    //maintain grid
    val ret = mutable.HashMap[Int, Tuple2[Option[Set[Int]], Option[Set[Int]]]]()
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
    }

    if (_N == 1) {
      _grid(0).update(0, Some(node))
      _node_map.update(node, (0, 0))
      _count += 1
    } else {
      val (y, x) = getPos(_filled, _N)
      _filled += 1
      _grid(y).update(x, Some(node))
      _node_map.update(node, (y, x))
      _count += 1
    }
  }

  /**
    * This always remove the last added node, effectively reverse the process
    */
  private def remove(): Unit ={
    assert(_count>0)
    _count -=1
    if(_N==1){
      _node_map.remove(_grid(0)(0).get)
      _grid(0).update(0, Option.empty)
    }else{
      assert(_filled>0)
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
      }
    }
  }

  /**
    * @return which position to be removed next
    */
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

  /**
    * Remove specific location using swap and remove()
    * @param location
    */
  def removeLocation(location : Tuple2[Int, Int]): Unit ={
    val remove_loc = nextRemove()
    if (remove_loc != location){
      val (ry,rx) = remove_loc
      val (y,x) = location
      val temp = _grid(ry)(rx)
      _grid(ry)(rx) = _grid(y)(x)
      _grid(y)(x) = temp
      _node_map.update(_grid(ry)(rx).get,(ry,rx))
      _node_map.update(_grid(y)(x).get, (y,x))
    }
    remove()
  }

  /**
    * Remove node by finding its location
    * @param node node idx to be removed
    */
  def removeNode(node: Int): Unit ={
    assert(_node_map.contains(node))
    removeLocation(_node_map(node))
  }

  /**
    * Getter method to get current layout snapshot
    */
  def currentMasterLayout() : MasterLayout = {
    val ret : MasterLayout = mutable.HashMap[Int, Tuple2[Option[Set[Int]], Option[Set[Int]]]]()
    if (_count ==0){ // special case for count=0
      return ret
    }
    for( i <- 0 until _N){
      setMasterNode(ret, i)
    }
    ret
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

object Dynamic2DGridLayout{
  type MasterLayout = mutable.HashMap[Int, Tuple2[Option[Set[Int]], Option[Set[Int]]]]
  // Option.empty means "keep unchanged", and Set() means an empty peer list, so we are forced to use option here
  // to distinguish between the two situations

  /**
    * Given two layout, calculate difference to minimize update
    * @param old_layout
    * @param new_layout
    * @return only difference
    */
  def calculate_difference(old_layout: MasterLayout, new_layout: MasterLayout) : MasterLayout = {
    var diff : MasterLayout = new_layout.clone()
    for(master_id <- old_layout.keys){
      if(!new_layout.contains(master_id)){
        diff(master_id)=(Some(Set()),Some(Set()))
        if(old_layout(master_id)._1.isEmpty){
          diff(master_id) = (Option.empty, diff(master_id)._2)
        }
        if(old_layout(master_id)._2.isEmpty){
          diff(master_id) = (diff(master_id)._1, Option.empty)
        }
      }else{
        // remove master id that does not change
        if(new_layout(master_id)._1 == old_layout(master_id)._1){
          diff(master_id) = (Option.empty, diff(master_id)._2)
        }
        if(new_layout(master_id)._2 == old_layout(master_id)._2){
          diff(master_id) = (diff(master_id)._1, Option.empty)
        }
        // clear the entry if both doesn't change.
        if(diff(master_id)._1 == Option.empty && diff(master_id)._2== Option.empty){
          diff.remove(master_id)
        }
      }
    }
    diff
  }
}