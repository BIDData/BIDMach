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
  import edu.berkeley.bid.CUMACH
   
  def treeProd(treesArray:Mat, feats:Mat, treePos:Mat, oTreeVal:Mat) {
    val nrows = feats.nrows;
    val ncols = feats.ncols;
    val ns = treesArray.nrows - 1;
    val ntrees = treePos.nrows;
    if (treePos.ncols != ncols || oTreeVal.ncols != ncols) {
      throw new RuntimeException("treeProd mismatch in ncols")
    }
    if (oTreeVal.nrows != ntrees) {
      throw new RuntimeException("treeProd mismatch in ntrees")
    }
    val tstride = (ns + 1) * (treesArray.ncols / ntrees);
    (treesArray, feats, treePos, oTreeVal) match {
      case (tA : GIMat, fs : GMat, tP : GIMat, oTV : GMat) => treeProd(tA, fs, tP, oTV, nrows, ncols, ns, tstride, ntrees)
      case (tA : GIMat, fs : GMat, tP : GIMat, oTI : GIMat) => treeSteps(tA, fs, tP, oTI, nrows, ncols, ns, tstride, ntrees, 1)
    }
  }
  
  def treeSearch(treesArray:Mat, feats:Mat, treePos:Mat, oTreeVal:Mat) {
    val nrows = feats.nrows;
    val ncols = feats.ncols;
    val ns = treesArray.nrows - 1;
    val ntrees = treePos.nrows;
    if (treePos.ncols != ncols || oTreeVal.ncols != ncols) {
      throw new RuntimeException("treeProd mismatch in ncols")
    }
    if (oTreeVal.nrows != ntrees) {
      throw new RuntimeException("treeProd mismatch in ntrees")
    }
    val tsize = (treesArray.ncols / ntrees);
    val tstride = (ns + 1) * tsize;
    val tdepth = (math.round(math.log(tsize)/math.log(2))).toInt
    (treesArray, feats, treePos, oTreeVal) match {
      case (tA : GIMat, fs : GMat, tP : GIMat, oTI : GIMat) => treeSteps(tA, fs, tP, oTI, nrows, ncols, ns, tstride, ntrees, tdepth)
    }
  }
  
  def treeProd(treesArray:GIMat, feats:GMat, treePos:GIMat, oTreeVal:GMat, nrows:Int, ncols:Int, ns: Int, tstride:Int, ntrees: Int) {
    val err = CUMACH.treeprod(treesArray.data, feats.data, treePos.data, oTreeVal.data, nrows, ncols, ns, tstride, ntrees);
    Mat.nflops += 1L * ncols * ntrees * ns
    if (err != 0) throw new RuntimeException("treeProd error %d: " + cudaGetErrorString(err) format err);
  }

  def treeSteps(treesArray:GIMat, feats:GMat, treePos:GIMat, oTreeI:GIMat, nrows:Int, ncols:Int, ns: Int, tstride:Int, ntrees: Int, tdepth:Int) {
    val err = CUMACH.treesteps(treesArray.data, feats.data, treePos.data, oTreeI.data, nrows, ncols, ns, tstride, ntrees, tdepth);
    Mat.nflops += 1L * ncols * ntrees * ns * tdepth
    if (err != 0) throw new RuntimeException("treeProd error %d: " + cudaGetErrorString(err) format err);
  }
  
}