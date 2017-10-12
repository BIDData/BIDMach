package BIDMach.viz;
import BIDMat.{BMat,Mat,SBMat,CMat,DMat,FMat,FFilter,IMat,HMat,GDMat,GFilter,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models.Model;
import BIDMach.networks.Net;
import BIDMach.networks.layers._;
import BIDMach.datasources._;
import BIDMach.updaters._
import scala.collection.mutable.ListBuffer;
import scala.concurrent.Future;
import scala.concurrent.ExecutionContext.Implicits.global

/***
    Synthesizing the image
**/

class Synthesis(val name: String = "Input",val modelname: String = "cifar") extends Visualization{
    val plot = new Plot(name);
    var iter = 10
    var lrate = 10f;
    var _lrate: Mat = null;
    var langevin = 0.1f
    var _langevin: Mat = null;
    val zero = irow(0);
    val ten = irow(0->10);
    var _net: Net = null;
    var D: Net = null;
    var updater:Grad = null;
    val dscores : ListBuffer[Float] = new ListBuffer[Float];
    val gscores : ListBuffer[Float] = new ListBuffer[Float];
    var accClassifier: Mat = null;
    var accDiscriminator: Mat = null;
    var dWeight = 1f;
    var _dWeight: Mat = null;        
    var noise: Mat = null;
    var done = false
        
        
    def check(model:Model, mats:Array[Mat]) = 1  
        
    def setInputGradient(net:Net) {
        for (i <- 0 until net.input_layers.length) {
            val layer = net.input_layers(i);
            if (layer.deriv.asInstanceOf[AnyRef] == null) 
                layer.deriv = layer.output.zeros(layer.output.dims);            
            layer.deriv.clear;                          
        }
        net.opts.compute_input_gradient = true;
    }
       
    override def init(model:Model, mats:Array[Mat]) = {
        interval  = 100;
        val net = model.asInstanceOf[Net];
        _net = net;
        _lrate = net.layers(0).output.zeros(1,1);
        _langevin = net.layers(0).output.zeros(1,1);  
        _dWeight = net.layers(0).output.zeros(1,1);  
        accClassifier = net.layers(0).output.zeros(net.layers(0).output.dims);
        accDiscriminator = net.layers(0).output.zeros(net.layers(0).output.dims);
        noise = net.layers(0).output.zeros(net.layers(0).output.dims);
        
        val (_D,_updater) = if (modelname == "cifar") Synthesis.buildCifarDiscriminator(); else Synthesis.buildMnistDiscriminator()
            
        D = _D;updater = _updater;
        
        setInputGradient(net);
        setInputGradient(D);
        
        plot.add_slider("lrate",(x:Int)=>{lrate=x/10f});
        plot.add_slider("iter",(x:Int)=>{iter=x+1});
        plot.add_slider("langevin",(x:Int)=>{langevin=x/10f});
    }

    override def doUpdate(model:Model, mats:Array[Mat], ipass:Int, pos:Long) = {        
        val net = model.asInstanceOf[Net];
        val layer = net.layers(0).asInstanceOf[InputLayer];
        val srcImg = utils.filter2img(layer.output(?,?,?,zero)/256f-0.5f,net.opts.tensorFormat);
        for(t<-0 until iter){
            net.forward;
            net.setderiv()
            net.backward(ipass, pos);
            _lrate(0,0) = lrate;
            _langevin(0,0) = langevin;
            layer.output~layer.output + (layer.deriv *@ _lrate);
            val dims = layer.output.dims;
            val s = dims(1)
            val d = layer.output.reshapeView(dims(1),dims(2),dims(0),dims(3))
            if (t % 2 == 0) {
                d(0->(s-1),?,?,0) = (d(0->(s-1),?,?,0) + d(1->s,?,?,0))*0.5f
                d(?,0->(s-1),?,0) = (d(?,0->(s-1),?,0) + d(?,1->s,?,0))*0.5f
            }
            else {
                d(1->s,?,?,0) = (d(0->(s-1),?,?,0) + d(1->s,?,?,0))*0.5f
                d(?,1->s,?,0) = (d(?,0->(s-1),?,0) + d(?,1->s,?,0))*0.5f
            }
            val img = srcImg \ utils.filter2img(layer.output(?,?,?,zero)/256f-0.5f,net.opts.tensorFormat)
            plot.plot_image(img)
        }
    }
    
    def generate(model:Model,random: Boolean = true) = {
        val net = model.asInstanceOf[Net];
        reset(random);
        if (!random)
            net.output_layers(0).target<--row(irow(0->net.datasource.opts.batchSize).data.map(_%10))        
        D.layers(0).output<--net.layers(0).output;
        D.output_layers(0).target(?)=1;
        accClassifier(?) = 0;
        accDiscriminator(?) = 0;
        for(t<-0 until iter){
            net.forward;
            net.layers(0).deriv.clear;
            net.setderiv()
            net.backward(0, 0);
            D.forward;
            D.layers(0).deriv.clear;
            D.setderiv()
            D.backward(0, 0);
            _lrate(0,0) = lrate;
            _langevin(0,0) = langevin;
            accClassifier ~ (accClassifier * 0.9f) + ((net.layers(0).deriv *@ net.layers(0).deriv) * 0.1f);
            accDiscriminator ~ (accDiscriminator * 0.9f) + ((D.layers(0).deriv *@ D.layers(0).deriv) * 0.1f);
            net.layers(0).deriv ~ net.layers(0).deriv / ((accClassifier+1e-7f)^0.5f);
            D.layers(0).deriv ~ D.layers(0).deriv / ((accDiscriminator+1e-7f)^0.5f);
            _dWeight(0,0) = dWeight;
            val grad = net.layers(0).deriv + (_dWeight *@ D.layers(0).deriv)
            normrnd(0,langevin,noise);
            grad ~ grad + noise;
            net.layers(0).output~net.layers(0).output + (grad *@ _lrate);
            max(net.layers(0).output,0,net.layers(0).output);
            min(net.layers(0).output,255,net.layers(0).output);            
            val dims = net.layers(0).output.dims;
            val s = dims(1);
            val d = net.layers(0).output.reshapeView(dims(1),dims(2),dims(0),dims(3))
            if (t % 2 == 0) {
                d(0->(s-1),?,?,0) = (d(0->(s-1),?,?,0) + d(1->s,?,?,0))*0.5f
                d(?,0->(s-1),?,0) = (d(?,0->(s-1),?,0) + d(?,1->s,?,0))*0.5f
            }
            else {
                d(1->s,?,?,0) = (d(0->(s-1),?,?,0) + d(1->s,?,?,0))*0.5f
                d(?,1->s,?,0) = (d(?,0->(s-1),?,0) + d(?,1->s,?,0))*0.5f
            }
                
            D.layers(0).output<--net.layers(0).output
        }
        net.layers(0).output
    }
    
    def trainD() = {
        val ds = D.datasource;
        done = true
        for(t<-0 until 1){
            ds.reset;
            var here = 0
            while (ds.hasNext && done) {
                here += ds.opts.batchSize
                val batch = ds.next
                batch(1)(?) = 1;
                D.layers(0).deriv.clear;
                D.dobatchg(batch,here,0);
                updater.update(0,here,0);     
                val dloss = mean(D.output_layers(0).output(1,?)).dv.toFloat;                
                val dscore = mean(D.output_layers(0).score).dv.toFloat
                dscores+=dscore
                    
                val data = generate(_net);
                D.output_layers(0).target(?) = 0;
                D.layers(0).deriv.clear;
                D.forward;D.setderiv();D.backward(0, 0);
                val gloss = mean(D.output_layers(0).output(1,?)).dv.toFloat;
                val gscore = mean(D.output_layers(0).score).dv.toFloat
                gscores+=gscore;
                updater.update(0,here,0);
                if (here/ds.opts.batchSize % 20 == 0 ) {
                    println("Trained %d samples. Real samples score: %.3f, Generate samples score: %.3f".format(here,dscore,gscore));
                    val img = utils.filter2img(D.layers(0).output(?,?,?,ten)/256f-0.5f,D.opts.tensorFormat);
                    plot.plot_image(img)
                }
            }
        }        
    }
    
    def launchTrain ={
        Future{trainD()}
    }
    
    def stop = {
        done = false
    }
    
    def reset(random:Boolean = true) {
        val data = _net.layers(0).output;
        if (random){
            val scale = 256f//maxi(data).dv.toFloat
            data<--rand(data.nrows,data.ncols).reshapeView(data.dims)*scale
        }
        else{
            _net.datasource.reset;
            val d = _net.datasource.next;
            data<--d(0);
            _net.output_layers(0).target<--d(1)
                
        }
    }
}

object Synthesis {
    def load(net:Net,fname:String) {
        for (i <- 0 until net.modelmats.length) {
            val data = loadMat(fname+"modelmat%02d.lz4" format i);
            net.modelmats(i)<--data
        }
    }
    
    def buildCifarDiscriminator() = {
        class MyOpts extends Net.Opts with FileSource.Opts with ADAGrad.Opts;
        val datadir = "/code/BIDMach/data/CIFAR10/parts/"
        val trainfname = datadir + "trainNCHW%d.fmat.lz4";
        val labelsfname = datadir + "labels%d.imat.lz4";
        val opts = new MyOpts;
        val ds = FileSource(trainfname, labelsfname, opts);
        val updater = new ADAGrad(opts)
        opts.batchSize = 100;
        opts.hasBias = true;
        opts.tensorFormat = Net.TensorNCHW;
        opts.convType = Net.CrossCorrelation;
        opts.lrate = 1e-4f;
        opts.vel_decay = 0.9f
        opts.gsq_decay = 0.99f
        opts.texp = 0.0f
        Mat.useCache = true;
        Mat.useGPUcache = true;

        val net = new Net(opts);
        {
            import BIDMach.networks.layers.Node._;
            val convt = jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION
            val in = input;
            val scalef = constant(row(0.01f));
            val inscale = in *@ scalef

            val conv1 = conv(inscale)(w=5,h=5,nch=32,stride=1,pad=0,initv=1f,convType=convt);
            val pool1 = pool(conv1)(w=2,h=2,stride=2);
            //val norm1 = batchNormScale(pool1)();
            val relu1 = relu(pool1)();

            val conv2 = conv(relu1)(w=5,h=5,nch=32,stride=1,pad=0,convType=convt);
            val pool2 = pool(conv2)(w=2,h=2,stride=2);
            //val norm2 = batchNormScale(pool2)();
            val relu2 = relu(pool2)();

            val conv3 = conv(relu2)(w=5,h=5,nch=32,stride=1,pad=2,convType=convt);
            val pool3 = pool(conv3)(w=3,h=3,stride=2);
            val fc3 =   linear(pool3)(outdim=2,initv=3e-2f);
            val out =   softmaxout(fc3)(scoreType=1); 

            val nodes = (in     \ scalef \ inscale on
                         conv1  \ pool1  \ relu1  on
                         conv2  \ pool2  \ relu2  on
                         conv3  \ pool3  \ null   on
                         fc3    \ out    \ null   ).t
            opts.nodemat = nodes;
        }        
        ds.init
        net.bind(ds)
        net.init;
        updater.init(net)
        (net,updater)
    }
    
    def buildMnistDiscriminator() = {
        class MyOpts extends Net.Opts with MatSource.Opts with ADAGrad.Opts;
        val traindir = "/code/BIDMach/data/MNIST/"
        val train0 = loadIDX(traindir+"train-images-idx3-ubyte").reshapeView(1,28,28,60000);
        val trainlabels0 = loadIDX(traindir+"train-labels-idx1-ubyte").reshapeView(1,60000);
        val opts = new MyOpts;
        val ds = new MatSource(Array(train0, trainlabels0), opts);
        val updater = new ADAGrad(opts);
        opts.batchSize = 100;
        opts.hasBias = true;
        opts.tensorFormat = Net.TensorNCHW;
        opts.convType = Net.CrossCorrelation;
        opts.lrate = 1e-4f;
        opts.vel_decay = 0.9f;
        opts.gsq_decay = 0.99f;
        opts.texp = 0.0f;
        Mat.useCache = true;
        Mat.useGPUcache = true;

        val net = new Net(opts);
        {
            import BIDMach.networks.layers.Node._;
            val convt = jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION

            val in = input;

            val conv1 = conv(in)(w=5,h=5,nch=6,stride=1,pad=2,initv=0.01f,convType=convt);
            val pool1 = pool(conv1)(w=2,h=2,stride=2,pad=0);
            val bns1 = batchNormScale(pool1)();
            val relu1 = relu(bns1)();

            val conv2 = conv(relu1)(w=5,h=5,nch=16,stride=1,pad=2,convType=convt);   
            val pool2 = pool(conv2)(w=2,h=2,stride=2,pad=0);
            val bns2 = batchNormScale(pool2)();
            val relu2 = relu(bns2)();

            val fc3 = linear(relu2)(outdim=120,initv=4e-2f);
            val relu3 = relu(fc3)();

            val fc4 = linear(relu3)(outdim=84,initv=1e-1f);
            val relu4  = relu(fc4)();

            val fc5  = linear(relu4)(outdim=2,initv=1e-1f);
            val out = softmaxout(fc5)(scoreType=1);

            val nodes = (in     \ null    \ null   \ null    on
                         conv1  \ pool1   \ bns1   \ relu1   on
                         conv2  \ pool2   \ bns2   \ relu2   on
                         fc3    \ relu3   \ null   \ null    on
                         fc4    \ relu4   \ null   \ null    on
                     fc5    \ out     \ null   \ null).t

            opts.nodemat = nodes;
        }        
        ds.init
        net.bind(ds)
        net.init;
        updater.init(net)
        (net,updater)
    }
    
    def buildImageNetDiscriminator() = {
        class MyOpts extends Net.Opts with FileSource.Opts// with Grad.Opts;
        val traindir = "/code/BIDMach/data/ImageNet/train/";
        val traindata = traindir+"partNCHW%04d.bmat.lz4";
        val trainlabels = traindir+"label%04d.imat.lz4";
        val opts = new MyOpts;
        val ds = FileSource(traindata, trainlabels, opts);
        opts.batchSize = 1;
        opts.hasBias = true;
        opts.tensorFormat = Net.TensorNCHW;
        opts.convType = Net.CrossCorrelation;
        Mat.useCache = true;
        Mat.useGPUcache = true;            
            
        val net = new Net(opts);
        {
            import BIDMach.networks.layers.Node._;
            Net.initDefaultNodeSet;

            val means = ones(3\256\256\opts.batchSize) *@ loadFMat(traindir+"means.fmat.lz4");
            val in =        input;
            val meanv =     const(means);
            val din =       in - meanv;
            val scalef =    const(row(0.01f));
            val cin =       cropMirror(din)(sizes=irow(3,227,227,0), randoffsets=irow(0,28,28,-1));

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
        ds.init
        net.bind(ds)
        net.init;
        load(net,"models/alex32p/");
        net            
    }
}
    
    
    