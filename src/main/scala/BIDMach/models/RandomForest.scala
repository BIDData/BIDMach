package BIDMach.models

// Random Forest model.
// Computes a matrix representing the Forest.
// The matrix encodes a 3 x nnodes x ntrees array

import BIDMat.{SBMat,CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
//import BIDMat.Solvers._
//import BIDMat.Sorting._
import edu.berkeley.bid.CUMAT
import BIDMach.models.RandForest._
import scala.util.hashing.MurmurHash3
import jcuda._

class RandomForest(override val opts:RandomForest.Opts) extends Model(opts) {
  var nnodes:Int = 0;
  
  def init() = {
    nnodes = math.pow(2, opts.depth + 1).toInt;
    val modelmat = if (opts.useGPU && Mat.hasCUDA > 0) gizeros(2 * nnodes, opts.ntrees) else izeros(2 * nnodes, opts.ntrees)
    modelmats = Array{modelmat};
  } 
  
  def doblock(gmats:Array[Mat], ipass:Int, i:Long) = {
    val sdata = gmats(0);
    val cats = gmats(1);
    val nnodes = gmats(2);
  }
  
  def evalblock(mats:Array[Mat], ipass:Int):FMat = {
    val sdata = gmats(0);
    row(0)
  } 
}

object RandomForest {
  
  
  def rhash(v1:Int, v2:Int, v3:Int, nb:Int):Int = {
    math.abs(MurmurHash3.mix(MurmurHash3.mix(v1, v2), v3) % nb)
  }
  
  def packFields(itree:Int, inode:Int, jfeat:Int, ifeat:Int, ivfeat:Int, icat:Int, fieldlengths:IMat):Long = {
    icat.toLong + 
    ((ivfeat.toLong + 
        ((ifeat.toLong + 
            ((jfeat.toLong + 
                ((inode.toLong + 
                    (itree.toLong << fieldlengths(INode))
                    ) << fieldlengths(JFeat))
                ) << fieldlengths(IFeat))
          ) << fieldlengths(IVFeat))
      ) << fieldlengths(ICat))
  }
  
  def extractAbove(fieldNum : Int, packedFields : Long, FieldShifts : IMat) : Int = {
    (packedFields >>> FieldShifts(fieldNum)).toInt
  }

  def extractField(fieldNum : Int, packedFields : Long, FieldShifts : IMat, FieldMasks : IMat) : Int = {
    (packedFields >>> FieldShifts(fieldNum)).toInt & FieldMasks(fieldNum) 
  }

  def getFieldShifts(fL : IMat) : IMat = {
    val out = izeros(1, fL.length)
    var i = fL.length - 2
    while (i >= 0) {
      out(i) = out(i+1) + fL(i+1)
      i -= 1
    }
    out
  }

  def getFieldMasks(fL : IMat) : IMat = {
    val out = izeros(1, fL.length)
    var i = 0
    while (i < fL.length) {
      out(i) = (1 << fL(i)) - 1
      i += 1
    }
    out
  }
  
  def treePack(idata:IMat, treenodes:IMat, cats:SMat, out:Array[Long], nsamps:Int, fieldlengths:IMat) = {
    val nfeats = idata.nrows
    val nitems = idata.ncols
    val ntrees = treenodes.nrows
    val ncats = cats.nrows
    val nnzcats = cats.length
    val ionebased = Mat.ioneBased
    var icol = 0
    while (icol < nitems) {
      var jci = cats.jc(icol) - ionebased
      val jcn = cats.jc(icol+1) - ionebased
      var itree = 0
      while (itree < ntrees) {
        val inode = treenodes(itree, icol)
        var jfeat = 0
        while (jfeat < nsamps) {
          val ifeat = rhash(itree, inode, jfeat, nfeats)
          val ivfeat = idata(ifeat, icol)
          var j = jci
          while (j < jcn) {
            out(jfeat + nsamps * (itree + ntrees * j)) = packFields(itree, inode, jfeat, ifeat, ivfeat, cats.ir(j) - ionebased, fieldlengths)
            j += 1
          }
          jfeat += 1
        }
        itree += 1
      }
      icol += 1
    }
    out
  }
  
  def treeStep(idata:IMat, tnodes:IMat,  trees:IMat, nifeats:Int, nnodes:Int)  {
    val nfeats = idata.nrows;
    val nitems = idata.ncols;
    val ntrees = tnodes.nrows;
    var icol = 0;
    while (icol < nitems) {
      var itree = 0;
      while (itree < ntrees) {
        val inode = tnodes(itree, icol);
        val ifeat = trees(2 * inode, itree);
        val ithresh = trees(1 + 2 * inode, itree);
        if (ifeat < 0) {
          tnodes(itree, icol) = ithresh;
        } else {
          val ivfeat = idata(ifeat, icol);
          if (ivfeat > ithresh) {
            tnodes(itree, icol) = 2 * inode + 2;
          } else {
            tnodes(itree, icol) = 2 * inode + 1;
          }
        }
        itree += 1;
      }
      icol += 1;
    }
  }
  
  def treeWalk(idata:IMat, tnodes:IMat,  trees:IMat, nifeats:Int, nnodes:Int, depth:Int)  {
    val nfeats = idata.nrows;
    val nitems = idata.ncols;
    val ntrees = tnodes.nrows;
    var icol = 0;
    while (icol < nitems) {
      var itree = 0;
      while (itree < ntrees) {
        var inode = 0;
        var id = 0;
        while (id <= depth) {
          val ifeat = trees(2 * inode, itree);
          val ithresh = trees(1 + 2 * inode, itree);
          if (ifeat < 0) {
            inode = ithresh;
            id = depth;
          } else {
            val ivfeat = idata(ifeat, icol);
            if (ivfeat > ithresh) {
              inode = 2 * inode + 2;
            } else {
              inode = 2 * inode + 1;
            }
          }
          id += 1;
        }
        tnodes(itree, icol) = inode;
        itree += 1;
      }
      icol += 1;
    }
  }
  
  trait Opts extends Model.Opts { 
     var depth = 1;
     var ntrees = 10;
  }
  
  class Options extends Opts {}
}
/*
/**********
 * fdata = nfeats x n 
 * fbounds = nfeats x 2
 * treenodes = ntrees x n
 * cats = ncats x n // labels 1 or greater
 * ntrees = ntrees
 * nsamps = nsamps
 * fieldLengths = 1 x 5 or 5 x 1
 * depth = depth 
 **********/
class RandomForest(fdata : Mat, cats : Mat, ntrees : Int, depth : Int, nsamps : Int, useGPU : Boolean) {
	val ITree = 0; val INode = 1; val IRFeat = 2; val IVFeat = 3; val ICat = 4

	var fbounds : Mat = mini(fdata, 2) \ maxi(fdata, 2)
	var fieldLengths : Mat = fdata.iones(1, 5)
	val itree = (Math.log(ntrees)/ Math.log(2)).toInt + 1; val inode = depth + 1; 
	val irfeat = (Math.log(fdata.nrows)/ Math.log(2)).toInt + 1; // TODO? errr.... not sure about this.....
	val ncats = cats.nrows
	var icat : Int = (Math.log(ncats)/ Math.log(2)).toInt + 1 // todo fix mat element access

	val ivfeat = Math.min(10,  64 - itree - inode - irfeat - icat); 
	fieldLengths <-- (itree\inode\irfeat\ivfeat\icat) 
	val n = fdata.ncols
	val treenodes = fdata.izeros(ntrees, fdata.ncols)
	val treesMetaInt = fdata.izeros(4, (ntrees * (math.pow(2, depth).toInt - 1))) // irfeat, threshold, cat, isLeaf
	treesMetaInt(2, 0->treesMetaInt.ncols) = (ncats) * iones(1, treesMetaInt.ncols)
	treesMetaInt(3, 0->treesMetaInt.ncols) = (-1) * iones(1, treesMetaInt.ncols)

	var FieldMaskRShifts : Mat = null;  var FieldMasks : Mat = null
	(fieldLengths) match {
		case (fL : IMat) => {
			FieldMaskRShifts = RandForest.getFieldMaskRShifts(fL); FieldMasks = RandForest.getFieldMasks(fL)
		}
	}
	
	def train {
		var totalTrainTime = 0f
		(fdata, fbounds, treenodes, cats, nsamps, fieldLengths, treesMetaInt, depth, FieldMaskRShifts, FieldMasks) match {
			case (fd : FMat, fb : FMat, tn : IMat, cts : SMat, nsps : Int, fL : IMat, tMI : IMat, d : Int, fMRS : IMat, fM : IMat) => {
				var d = 0
				while (d <  depth) {
					println("d: " + d)
					val treePacked : Array[Long] = RandForest.treePack(fd, fb, tn, cts, nsps, fL)
					RForest.sortLongs(treePacked, useGPU)
					RForest.updateTreeData(treePacked, fL, ncats, tMI, depth, d == (depth - 1), fMRS, fM)
					if (!(d == (depth - 1))) {
						RForest.treeSteps(tn , fd, fb, fL, tMI, depth, ncats, false)
					}
					d += 1
				}
			}
		}
	}

	// returns 1 * n
	def classify(tfdata : Mat) : Mat = {
		var totalClassifyTime : Float = 0f
		val treenodecats = tfdata.izeros(ntrees, tfdata.ncols)
		flip
		(tfdata, fbounds, treenodecats, fieldLengths, treesMetaInt, depth, ncats) match {
			case (tfd : FMat, fb : FMat, tnc : IMat, fL : IMat, tMI : IMat, depth : Int, ncts : Int) => {
				RForest.treeSearch(tnc, tfd, fb, fL, tMI, depth, ncts)
			}
		}
		val out = RForest.voteForBestCategoriesAcrossTrees(treenodecats.t, ncats) // ntrees * n
		out
	}


} 
*/