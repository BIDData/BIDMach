import BIDMat.{Mat, DMat, FMat, IMat, BMat, CMat, SMat, SDMat, GMat, GSMat, HMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import BIDMach.{Learner,LogisticModel,LinearRegModel,ADAGradUpdater}

{
var a = new Array[Int](1)
jcuda.runtime.JCuda.cudaGetDeviceCount(a)
if (a(0) > 0) jcuda.runtime.JCuda.initialize
Mat.hasCUDA = a(0)
}
printf("%d CUDA device%s found\n", Mat.hasCUDA, if (Mat.hasCUDA == 1) "" else "s")
