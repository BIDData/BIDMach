package BIDMach.allreduce

import BIDMat.{BMat,CMat,CSMat,DMat,Dict,FMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,HMat,IDict,Image,IMat,LMat,Mat,SMat,SBMat,SDMat,TMat}
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

import org.scalatest.{Matchers, WordSpec}

class LeNetSpec extends WordSpec with Matchers{
  "LeNet" should {
    "train properly" in {
      //Mat.checkMKL(true)
      //Mat.checkCUDA(true)
      //System.setProperty("java.library.path", "C:\\Users\\zhang\\Documents\\GitHub\\BIDMach\\lib\\;"+System.getProperty("java.library.path"));
      //System.out.println(System.getProperty("java.library.path"))
      val traindir = "./data/MNIST/"

      val train0 =       loadIDX(traindir+"train-images-idx3-ubyte.gz").reshapeView(1,28,28,60000);
      val trainlabels0 = loadIDX(traindir+"train-labels-idx1-ubyte.gz").reshapeView(1,60000);
      val test =         loadIDX(traindir+"t10k-images-idx3-ubyte.gz").reshapeView(1,28,28,10000);
      val testlabels =   loadIDX(traindir+"t10k-labels-idx1-ubyte.gz").reshapeView(1,10000);

      val rp = randperm(60000);
      val train = train0(?,?,?,rp);
      val trainlabels = trainlabels0(?,rp);

      val mt = train.mean(irow(3));
      train ~ train - mt;
      test ~ test - mt;

      val (nn,opts) = Net.learner(train,trainlabels);

      //val convt = jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION
      val convt = 0

      opts.batchSize= 64
      opts.npasses = 2

      opts.lrate = 1e-3f
      opts.texp = 0.3f
      opts.pstep = 0.1f
      opts.hasBias = true;
      opts.tensorFormat = Net.TensorNCHW;
      //opts.autoReset = false;

      import BIDMach.networks.layers.Node._;

      val in = input();

      //val conv1 = conv(in)(w=5,h=5,nch=20,stride=1,pad=0,initv=0.01f,convType=convt);
      //val pool1 = pool(conv1)(w=2,h=2,stride=2);

      //val conv2 = conv(pool1)(w=5,h=5,nch=20,stride=1,pad=0,convType=convt);
      //val pool2 = pool(conv2)(w=2,h=2,stride=2);

      val fc3 = linear(in)(outdim=500,initv=3e-2f);
      val relu3 = relu(fc3)();

      val fc4 = linear(relu3)(outdim=10,initv=5e-2f);
      //val out = softmaxout(fc4)();
      val out = softmaxout(fc4)(scoreType=1);

      val nodes = (in     \ null   on
        //conv1  \ pool1  on
        //conv2  \ pool2  on
        fc3    \ relu3  on
        fc4    \ out    ).t



      opts.nodemat = nodes;
      val model = nn.model.asInstanceOf[Net];
      nn.train;
    }
  }
}
