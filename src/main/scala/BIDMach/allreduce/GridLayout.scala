package BIDMach.allreduce

case class GridGroup(val index: Integer, val dim: Integer){}

class GridLayout(val scale: Int, val dim: Int) {
  val total: Int = Math.pow(scale, dim).toInt
  type GridMember = Array[Int]
  private val members: Array[GridMember] = genMembers(dim)

  private def genMembers(dim: Int): Array[Array[Int]] =  {
    if(dim == 0){
      Array(Array())
    }
    else{
      var ret: Array[Array[Int]] = Array()
      for(x <- genMembers(dim -1)){
        for(i <- 0 until scale){
          ret :+= (x:+i)
        }
      }
      ret
    }
  }

  def members(group: GridGroup):Array[Int]={
    //given a group, return the members it have, O(total) naiive implemetation
    var ret = Array[Int]()
    for((member, i) <- members.zipWithIndex){
      if(member(group.dim) == group.index){
        ret :+= i.asInstanceOf[Int]
      }
    }
    ret
  }

  def groups(member_idx:Integer):Array[GridGroup]= {
    //given a member_idx, return the groups it belong to
    assert(0 <= member_idx && member_idx < total)
    var ret = Array[GridGroup]();
    val member = members(member_idx);
    for (i <- 0 until dim) {
      ret :+= new GridGroup(member(i), i)
    }
    ret
  }
}

