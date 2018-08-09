import BIDMat.{BMat,CMat,CSMat,DMat,Dict,FMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,HMat,IDict,Image,IMat,LMat,Mat}
import BIDMat.{Quaternion,SMat,SBMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import BIDMach.{Learner,ParLearner}
import BIDMach.models.{Click,FM,GLM,KMeans,KMeansw,LDA,LDAgibbs,Model,NMF,SFA,RandomForest,SVD}
import BIDMach.networks.{Net}
import BIDMach.datasources.{DataSource,MatSource,FileSource,SFileSource}
import BIDMach.datasinks.{DataSink,MatSink}
import BIDMach.mixins.{CosineSim,Perplexity,Top,L1Regularizer,L2Regularizer}
import BIDMach.updaters.{ADAGrad,Batch,BatchNorm,Grad,IncMult,IncNorm,Telescoping,Updater}
import BIDMach.causal.{IPTW}
import BIDMat.Mat.console_publish

Mat.checkMKL(false)
Mat.checkCUDA(true)

