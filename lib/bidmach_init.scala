import BIDMat.{SBMat,CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import BIDMach.Learner
import BIDMach.datasources.{MatDS,FilesDS,SFilesDS}
import BIDMach.models.{LDA,NMF,ALS,LDAgibbs,GLM}

Mat.checkMKL
Mat.checkCUDA

