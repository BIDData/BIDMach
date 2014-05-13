import BIDMat.{CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SBMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import BIDMach.Learner
import BIDMach.datasources.{MatDS,FilesDS,SFilesDS}
import BIDMach.models.{KMeans,GLM,LDA,NMF,SFA,FM,LDAgibbs}
import BIDMach.causal.{IPTW}

Mat.checkMKL
Mat.checkCUDA

