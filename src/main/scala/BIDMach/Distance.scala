package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

abstract class Distance(val opts:Distance.Options = new Distance.Options) {

  def dists(a:FMat):FMat
  def dists(a:FMat, b:FMat):FMat

}

object Distance {
  class Options {
          
  }
}


//Euclidean Distance
class euclidDistance(override val opts:euclidDistance.Options = new euclidDistance.Options) extends Distance(opts) { 
  
  override def dists(a:FMat):FMat = {
    val dd = if (Mat.hasCUDA > 0) a xTG a else a xT a;
    val d1 = getdiag(dd)
    dd ~ dd * 2.0f
    dd ~ d1 - dd
    dd ~ dd + (d1.t)
    max(dd, 0f, dd)
    sqrt(dd, dd)
    dd
  }

  override def dists(a:FMat, b:FMat):FMat = {
    val aa = getdiag(if (Mat.hasCUDA > 0) a xTG a else a xT a);
    val bb = getdiag(if (Mat.hasCUDA > 0) b xTG b else b xT b);
    val ab = if (Mat.hasCUDA > 0) a xTG b else a xT b;
    ab ~ ab * 2.0f
    ab ~ aa - ab + (bb.t)
    max(ab, 0f, ab)
    sqrt(ab, ab)

    ab
  }
}

object euclidDistance  {
  class Options extends Distance.Options {

  }
}


//Cosangle Distance
class cosangleDistance(override val opts:cosangleDistance.Options = new cosangleDistance.Options) extends Distance(opts) { 
  
  override def dists(a:FMat):FMat = {
    val dd = if (Mat.hasCUDA > 0) a xTG a else a xT a;
    var d1 = getdiag(dd)
    sqrt(d1, d1)
    d1 =  if (Mat.hasCUDA > 0) 0.0f+(d1 xTG d1) else 0.0f+(d1 xT d1);
    dd ~ 1 - dd / d1
    dd
  }

  override def dists(a:FMat, b:FMat):FMat = {
    val aa = getdiag(if (Mat.hasCUDA > 0) a xTG a else a xT a);
    val bb = getdiag(if (Mat.hasCUDA > 0) b xTG b else b xT b);
    val ab = if (Mat.hasCUDA > 0) a xTG b else a xT b;
  
    sqrt(aa, aa)
    sqrt(bb, bb)

    val dd = if (Mat.hasCUDA > 0) 0.0f+(aa xTG bb) else 0.0f+(aa xT bb);
    ab ~ 1 - ab / dd
    ab
  }
}

object cosangleDistance  {
  class Options extends Distance.Options {

  }
}


//Correlation Distance
class correlationDistance(override val opts:correlationDistance.Options = new correlationDistance.Options) extends Distance(opts) { 
  
  override def dists(a:FMat):FMat = {
    val mu = mean(a,1)
    a ~ a - mu
    val dd = if (Mat.hasCUDA > 0) a xTG a else a xT a;
    var d1 = getdiag(dd)
    sqrt(d1, d1)
    d1 =  if (Mat.hasCUDA > 0) 0+(d1 xTG d1) else 0+(d1 xT d1);
    dd ~ 1 - dd / d1
    dd
  }

  override def dists(a:FMat, b:FMat):FMat = {
    val mua = mean(a,1)
    a ~ a - mua
    val mub = mean(b,1)
    b ~ b - mub

    val aa = getdiag(if (Mat.hasCUDA > 0) a xTG a else a xT a);
    val bb = getdiag(if (Mat.hasCUDA > 0) b xTG b else b xT b);
    val ab = if (Mat.hasCUDA > 0) a xTG b else a xT b;
  
    sqrt(aa, aa)
    sqrt(bb, bb)

    val dd = if (Mat.hasCUDA > 0) 0.0f+(aa xTG bb) else 0.0f+(aa xT bb);
    ab ~ 1 - ab / dd
    ab
  }
}

object correlationDistance  {
  class Options extends Distance.Options {

  }
}

