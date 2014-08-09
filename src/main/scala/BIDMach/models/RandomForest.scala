package BIDMach.models

import BIDMat.{SBMat,CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Sorting._
import edu.berkeley.bid.CUMAT
import BIDMach.models.RandForest._
import jcuda._
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