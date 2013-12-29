import BIDMat.{BMat,CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import BIDMach.Learner
import BIDMach.datasources.{MatDataSource,FilesDataSource,SFilesDataSource}
import BIDMach.models.{LDAModel,NMFModel}

Mat.checkMKL
Mat.checkCUDA

