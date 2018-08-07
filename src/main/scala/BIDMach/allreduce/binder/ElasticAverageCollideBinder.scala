package BIDMach.allreduce.binder

import java.util.ArrayDeque
import java.util.concurrent.atomic.AtomicInteger
import java.util.logging.Logger
import scala.util.Random

import BIDMach.allreduce.binder.AllreduceBinder.{DataSink, DataSource}
//import BIDMach.models.Model
import BIDMach.updaters.Grad
import BIDMat.{Mat, FMat, GMat}


/**
  * Linearize input model mats, and elastic-average update to the same model.
  * Perform momentum exchange among several nodes in a cluster, preserving total energy of the nodes.
  *
  * @param model
  * @param alphaFromIter
  */
// FIXME: should get rndseed, node num and # nodes from worker
class ElasticAverageCollideBinder(updater: Grad, alphaFromIter: Int => Float, hardness: Float, rndseed: Long, inode: Int,
                                  nnodes: Int, logger: Logger) extends AllreduceBinder {

  val model = updater.model
  // Keeping track of elastic updates
  var tic = System.currentTimeMillis()
  val reduceCount = new AtomicInteger()
  
  val random = new Random(rndseed)
  // TODO: make these GMats when applicable
  val rawRandVecs = new Array[Array[FMat]](nnodes)
  val randVecs = new Array[Array[FMat]](nnodes)
  val randVecSqNorms = new Array[Array[Float]](nnodes)
  var rvOffset = 0
  // TODO: think about GMats too
  val aelem = FMat(1, 1)
  
  // TODO: make this more efficient by making use of functionality in SciFunctions etc.
  def genRandomVector(out: FMat) = {
    var i = 0
    val len = out.length
    while (i < len) {
      out.data(i) = random.nextGaussian().toFloat
    }
  }
  
  def dotprod(a:Mat, b:Mat):Float = {
    aelem ~ a.contents dot b.contents
    aelem.dv.toFloat;
  }
  
  // TODO: is synchronization necessary to get updater momentum lengths
  def initRandVecs = {
    if (rawRandVecs(0) eq null) {
      for (i <- 0 until nnodes) {
        rawRandVecs(i) = new Array(updater.momentum.length)
        
        for ((pm, j) <- updater.momentum.iterator.zipWithIndex) {
          val fmat = FMat.make(pm.dims)
          genRandomVector(fmat.contents())
          pm match {
            case _: GMat => rawRandVecs(i)(j) = GMat(fmat)
            case _: FMat => rawRandVecs(i)(j) = fmat
          }
        }
      }
      
      for (i <- 0 until nnodes) {
        randVecs(i) = new Array(updater.momentum.length)
        randVecSqNorms(i) = new Array(updater.momentum.length)
        for (j <- 0 until updater.momentum.length) {
          randVecs(i)(j) = rawRandVecs(i)(j) - rawRandVecs((i + 1) % nnodes)(j)
          randVecSqNorms(i)(j) = dotprod(randVecs(i)(j), randVecs(i)(j))
        }
      }
    }
  }
  
  def rotateRndVecs = {
    val prevOffset = (rvOffset + nnodes - 1) % nnodes
    
    for (randMat <- rawRandVecs(rvOffset)) {
      randMat match {
        case gmat: GMat =>
          val fmat = FMat.make(randMat.dims)
          genRandomVector(fmat)
          gmat <-- fmat
        case fmat: FMat => genRandomVector(fmat)
      }
    }
    
    for (offset <- Array(prevOffset, rvOffset)) {
      val nextOffset = (offset + 1) % nnodes
      for ((v1, v2) <- randVecs(offset) zip randVecs(nextOffset)) {
        v1 ~ v1 - v2
      }
      for ((v, i) <- randVecs(offset).iterator.zipWithIndex) {
        randVecSqNorms(offset)(i) = dotprod(v, v)
      }
    }
    
    rvOffset += 1
    if (rvOffset == nnodes) rvOffset = 0
  }

  override lazy val totalDataSize: Int = {
    var ret = 0
    updater.momentum.synchronized {
      // Momentum mats
      for (p <- updater.momentum) ret += p.length
      // Squared magnitudes of momentum mats
      ret += updater.momentum.length
      // Dot product of momentum mats and random mats
      ret += updater.momentum.length
    }
    // Model mats
    model.modelmats.synchronized {
      for (mat <- model.modelmats) ret += mat.length
    }
    ret
  }

  override def dataSource: DataSource = inputRequest => {
    initRandVecs

    val ret: Array[Float] = new Array[Float](totalDataSize)
    var current = totalDataSize
    val myRandVecs = randVecs((rvOffset + inode) % nnodes)
    
    // TODO: do we need to lock on the model and updater mats

    // backward traversing model mats, assuming forward traversal by the training model
    for (mm <- model.modelmats.reverseIterator) {
      current -= mm.length
      mm match {
        case gmat: GMat => GMat.GPUtoCPUarraycopy(gmat.pdata, 0, ret, current, gmat.length, "ElasticAverageBinder dataSource")
        case fmat: FMat => System.arraycopy(fmat.contents().data, 0, ret, current, fmat.length)
      }
    }

    // dot product of momentum and random vectors
    // backward traversing update mats, assuming forward traversal by updater
    for ((pm, r) <- updater.momentum.reverseIterator zip myRandVecs.reverseIterator) {
      current -= 1
      ret(current) = dotprod(pm, r)
    }
    
    // squared norm of momentums
    for (pm <- updater.momentum.reverseIterator) {
      current -= 1
      ret(current) = dotprod(pm, pm)
    }
    
    // backward traversing update mats, assuming forward traversal by updater
    for (pm <- updater.momentum.reverseIterator) {
      current -= pm.length
      pm match {
        case gmat: GMat => GMat.GPUtoCPUarraycopy(gmat.pdata, 0, ret, current, gmat.length, "ElasticAverageBinder dataSource")
        case fmat: FMat => System.arraycopy(fmat.contents().data, 0, ret, current, fmat.length)
      }
    }
    
    assert(current == 0, "current should be zero after iteration")

    AllReduceInput(ret)

  }



  override def dataSink: DataSink = reducedOutput => {

    reduceCount.synchronized {
      val currentCount: Int = reduceCount.getAndIncrement()
      val updateCounts = 10
      if (currentCount % updateCounts == 0) {
        val toc = System.currentTimeMillis()
        if (currentCount > 0) {
          logger.info(f"elastic_updates/s=${updateCounts/((toc - tic) / 1.0e3)}%2.2f, total_updates=$currentCount")
        }
        tic = toc
      }
    }
    val reducedData = reducedOutput.data

    assert(reducedData.length == totalDataSize, "Reduced output should be same length as input")

    // backward traversing model mats, assuming forward traversal by the training model
    // using while instead of for loop due to performance
    var current = totalDataSize
    val alpha = alphaFromIter(reducedOutput.iteration)

    for (mm <- model.modelmats.reverseIterator) {
      current -= mm.length
      mm.synchronized {
        mm match {
          case gmat: GMat =>
            val gReduced = GMat.make(gmat.dims)
            GMat.CPUtoGPUarraycopy(reducedData, current, gReduced.pdata, 0, gmat.length, "ElasticAverageCollideBinder dataSink")
            gReduced ~ gReduced / aelem.set(nnodes)
            gmat ~ gmat * aelem.set(1 - alpha)
            gReduced ~ gReduced * aelem.set(alpha)
            gmat ~ gReduced + gmat
            gReduced.free()
          case fmat: FMat =>
            val fReduced = FMat.make(fmat.dims)
            System.arraycopy(reducedData, current, fReduced.contents().data, 0, fmat.length)
            fReduced ~ fReduced / aelem.set(nnodes)
            fmat ~ fmat * aelem.set(1 - alpha)
            fReduced ~ fReduced * aelem.set(alpha)
            fmat ~ fReduced + fmat
        }
      }
    }
    
    val sumPmR = new Array[Float](updater.modelmats.length)
    current -= updater.modelmats.length
    System.arraycopy(reducedData, current, sumPmR, 0, updater.modelmats.length)
    
    val sumPmPm = new Array[Float](updater.modelmats.length)
    current -= updater.modelmats.length
    System.arraycopy(reducedData, current, sumPmPm, 0, updater.modelmats.length)
    
    val meanP = new Array[Mat](updater.modelmats.length)
    for (i <- updater.modelmats.length - 1 to 0 by -1) {
      current -= updater.modelmats(i).length
      val pbar = updater.modelmats(i) match {
        case _: GMat =>
          val pbar = GMat.make(updater.modelmats(i).dims)
          GMat.CPUtoGPUarraycopy(reducedData, current, pbar.pdata, 0, updater.modelmats(i).length, "ElasticAverageCollideBinder dataSink")
          pbar
        case _: FMat =>
          val pbar = FMat.make(updater.modelmats(i).dims)
          System.arraycopy(reducedData, current, pbar.contents().data, 0, updater.modelmats(i).length)
          pbar
      }
      pbar ~ pbar / aelem.set(nnodes)
      meanP(i) = pbar
    }
    
    assert(current == 0, "current should be zero after iteration")
    
    for (j <- updater.modelmats.length - 1 to 0 by -1) {
      // TODO: not hold the lock for 1293579813753 years, but also avoid data races
      updater.modelmats(j) synchronized {
        val x = meanP(j) - updater.modelmats(j)
        x ~ x * aelem.set(hardness)
        x ~ x + updater.modelmats(j)
        
        val sumXR = (1 - hardness) * sumPmR(j)
        val sumXXminusPmPm = hardness * (hardness - 2) * (sumPmPm(j) - nnodes * dotprod(meanP(j), meanP(j)))
        
        val twoSumXR = 2 * sumXR
        val sumRR = randVecSqNorms.map(_(j)).reduce(_ + _)
        // Discriminant should always be positive for any hardness in [0, 1]
        val discr = twoSumXR*twoSumXR - 4*sumRR*sumXXminusPmPm
        val epsilon = 1e-36f
        val beta = if (Mat.myrand.nextFloat() < 0.5f) {
          (-twoSumXR + math.sqrt(discr).toFloat) / (2 * sumRR + epsilon)
        } else {
          (-twoSumXR - math.sqrt(discr).toFloat) / (2 * sumRR + epsilon)
        }
        
        updater.modelmats(j) ~ x - aelem.set(beta) * randVecs((rvOffset + inode) % nnodes)(j)
      }
    }
    
    rotateRndVecs
  }

}

