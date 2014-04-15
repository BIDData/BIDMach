package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.Learner

object RForest {
  import jcuda.runtime._
  import jcuda.runtime.JCuda._
  import jcuda.runtime.cudaError._
  import jcuda.runtime.cudaMemcpyKind._
  import scala.util.hashing.MurmurHash3._
  import edu.berkeley.bid.CUMACH
  
  def rhash(v1:Int, v2:Int, nb:Int):Int = {
    mix(v1, v2) % nb
  }
  
  def packFields(itree:Int, irfeat:Int, inode:Int, ivfeat:Int, icat:Int, fieldlengths:IMat):Long = {
    icat.toLong + 
    ((ivfeat.toLong + ((inode.toLong + ((irfeat.toLong + (itree.toLong 
        << fieldlengths(1))) 
        << fieldlengths(2))) 
        << fieldlengths(3)))
        << fieldlengths(4))
  }
   
  def treePack(fdata:FMat, fbounds:FMat, treenodes:IMat, cats:SMat, nr:Int, fieldlengths:IMat) {
    val nfeats = fdata.nrows
    val nsamples = fdata.ncols
    val ntrees = treenodes.nrows
    val ncats = cats.nrows
    val nnzcats = cats.nnz
    val out = new Array[Long](ntrees * nr * nnzcats)
    var i = 0
    var ioff = Mat.ioneBased
    val nifeats = 2 ^ fieldlengths(3)
    while (i < ntrees) {
       var j = 0
       while (j < nr) {
         var k = 0
         while (k < nsamples) {
           val inode = treenodes(i, k)
           val ifeat = rhash(inode, j, nfeats)
           val vfeat = fdata(ifeat, k)
           val ivfeat = math.min(nifeats-1, math.floor((vfeat - fbounds(ifeat,0))/(fbounds(ifeat,1) - fbounds(ifeat,0))*nifeats).toInt)
           var jc = cats.jc(k) - ioff
           val jcn = cats.jc(k+1) - ioff
           while (jc < jcn) {
             out(j + nr * (i + ntrees * jc)) = packFields(i, j, inode, ivfeat, cats.data(jc).toInt, fieldlengths)
             jc += 1
           }
           k += 1
         }
         j += 1
       }
       i += 1
    }
  }
  
}