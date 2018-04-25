import BIDMach.Learner
import BIDMach.allreduce.AllreduceNode.{getBasicConfigs, startNodeAfterIter}
import BIDMach.allreduce.binder.ElasticAverageBinder
import BIDMach.allreduce.AllreduceNode
import BIDMach.datasources.FileSource
import BIDMach.networks.Net
import BIDMach.networks.layers._
import BIDMach.updaters.Grad
import BIDMat.SciFunctions

val traindir = "./data/ImageNet/train/";
//val traindir = "/home/jfc/data/ImageNet/2012/BIDMach/train/";
val testdir = "./data/ImageNet/val/";
val traindata = traindir+"partNCHW%04d.bmat.lz4";
val trainlabels = traindir+"label%04d.imat.lz4";
val testdata = testdir+"partNCHW%04d.bmat.lz4";
val testlabels = testdir+"label%04d.imat.lz4";
val testpreds = testdir+"pred%04d.fmat.lz4";

SciFunctions.setseed(4)

class MyOpts extends Learner.Options with Net.Opts with FileSource.Opts with Grad.Opts;
val opts = new MyOpts;
val ds = FileSource(traindata, trainlabels, opts);
val net = new Net(opts);
val grad = new Grad(opts);
val nn = new Learner(ds, net, null, grad, null, opts);


def lr_update(ipass:Float, istep:Float, frac:Float):Float = {
  val lr = if (ipass < 20) {
    1e-2f
  } else if (ipass < 40) {
    1e-3f
  } else 1e-4f;
  lr
}

opts.logfile = "logAlexnet_cluster=16_alpha=0_1.txt";
opts.batchSize= 128;
opts.npasses = 80;
//opts.nend = 10;
opts.lrate = 1e-4f;
opts.texp = 0f;
opts.pstep = 0.05f
opts.hasBias = true;
opts.l2reg = 0.0005f;
opts.vel_decay = 0.9f;
opts.lr_policy = lr_update _;
opts.tensorFormat = Net.TensorNCHW;
opts.useCache = false;
opts.convType = Net.CrossCorrelation;
opts.inplace = Net.BackwardCaching;
opts.inplace = Net.InPlace;

:silent

val means = ones(3\256\256\opts.batchSize) *@ loadFMat(traindir+"means.fmat.lz4");

{
  import BIDMach.networks.layers.Node._;

  Net.initDefaultNodeSet;

  val in =        input();
  val meanv =     const(means);
  val din =       in - meanv;
  val scalef =    const(row(0.01f));
  //val sdin =      din *@ scalef;
  //val fin =       format(in)();
  val cin =       cropMirror(din)(sizes=irow(3,227,227,0), randoffsets=irow(0,28,28,-1));
  //val min =       randmirror(cin)();

  val conv1 =     conv(cin)(w=11,h=11,nch=96,stride=4,initfn=Net.gaussian,initv=0.01f,initbiasv=0f);
  val relu1 =     relu(conv1)();
  val pool1 =     pool(relu1)(w=3,h=3,stride=2);
  val norm1 =     LRNacross(pool1)(dim=5,alpha=0.0001f,beta=0.75f);

  val conv2 =     conv(norm1)(w=5,h=5,nch=256,stride=1,pad=2,initfn=Net.gaussian,initv=0.01f,initbiasv=1f);
  val relu2 =     relu(conv2)();
  val pool2 =     pool(relu2)(w=3,h=3,stride=2);
  val norm2 =     LRNacross(pool2)(dim=5,alpha=0.0001f,beta=0.75f);

  val conv3 =     conv(norm2)(w=3,h=3,nch=384,pad=1,initfn=Net.gaussian,initv=0.01f,initbiasv=0f);
  val relu3 =     relu(conv3)();

  val conv4 =     conv(relu3)(w=3,h=3,nch=384,pad=1,initfn=Net.gaussian,initv=0.01f,initbiasv=1f);
  val relu4 =     relu(conv4)();

  val conv5 =     conv(relu4)(w=3,h=3,nch=256,pad=1,initfn=Net.gaussian,initv=0.01f,initbiasv=1f);
  val relu5 =     relu(conv5)();
  val pool5 =     pool(relu5)(w=3,h=3,stride=2);

  val fc6 =       linear(pool5)(outdim=4096,initfn=Net.gaussian,initv=0.01f,initbiasv=1f);
  val relu6 =     relu(fc6)();
  val drop6 =     dropout(relu6)(0.5f);

  val fc7 =       linear(drop6)(outdim=4096,initfn=Net.gaussian,initv=0.01f,initbiasv=1f);
  val relu7  =    relu(fc7)();
  val drop7 =     dropout(relu7)(0.5f);

  val fc8  =      linear(drop7)(outdim=1000,initfn=Net.gaussian,initv=0.01f,initbiasv=1f);
  val out =       softmaxout(fc8)(scoreType=1,lossType=1);

  opts.nodeset=Net.getDefaultNodeSet
}
def loss = {net.layers(net.layers.length-1).asInstanceOf[SoftmaxOutputLayer]};

val sgd = nn.updater.asInstanceOf[Grad];



nn.launchTrain;

// All-reduce
val nodeConfig = getBasicConfigs().copy(elasticRate = 0.1f)
val binder = new ElasticAverageBinder(nn.model, (x: Int) => nodeConfig.elasticRate, nn.myLogger)
AllreduceNode.startNodeAfterIter(nn, iter = 0, nodeConfig, binder)

println("Examine the 'nn' variable to track learning state.\n");

//nn.train;

//val (mm, mopts) =  Net.predLabels(net, testdata, testlabels);
//mopts.batchSize= opts.batchSize;
//mopts.autoReset = false;
//mm.predict;

//println("Accuracy = %f" format mean(mm.results(0,?),2).v);
:silent





