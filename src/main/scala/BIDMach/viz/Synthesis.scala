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
    //val ten = irow(0->100);
    var _net: Net = null;
    var D: Net = null;
    var updater:Grad = null;
    val dscores : ListBuffer[Float] = new ListBuffer[Float];
    val gscores : ListBuffer[Float] = new ListBuffer[Float];
    var accClassifier: Mat = null;
    var accDiscriminator: Mat = null;
    var momentum: Mat = null;
    var vWeight = 0.9f;
    var _vWeight: Mat = null;
    var dWeight = 0.5f;
    var _dWeight: Mat = null;
    var scale = 64f;
    var _scale: Mat = null;
    var noise: Mat = null;
    var done = false;
    var ipass = 0;
    var trans : Mat=>Mat = null;
    var gsteps : ListBuffer[Float] = new ListBuffer[Float];
    var gdata : Mat = null;
    var trainDis = true;
    
        
        
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
        _vWeight = net.layers(0).output.zeros(1,1);  
        _scale = net.layers(0).output.zeros(1,1);  
        accClassifier = net.layers(0).output.zeros(net.layers(0).output.dims);
        accDiscriminator = net.layers(0).output.zeros(net.layers(0).output.dims);
        momentum = net.layers(0).output.zeros(net.layers(0).output.dims);
        noise = net.layers(0).output.zeros(net.layers(0).output.dims);
        gdata = net.layers(0).output.zeros(net.layers(0).output.dims);
        
        val (_D,_updater) = modelname match {
            case "cifar" => Synthesis.buildCifarDiscriminator(); 
            case "mnist" => Synthesis.buildMnistDiscriminator();
            case "imagenet" => Synthesis.buildImageNetDiscriminator();
        }
            
        D = _D;updater = _updater;
        
        setInputGradient(net);
        setInputGradient(D);
        
        plot.add_slider("lrate",(x:Int)=>{lrate=x/10f},10);
        plot.add_slider("iter",(x:Int)=>{iter=(x+1)*10},1000);
        plot.add_slider("langevin",(x:Int)=>{langevin=x/10f},10);
        plot.add_slider("discriminatorWeight",(x:Int)=>{dWeight=x/100f},1);
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
    
    def mcmc(model:Model,targetScore:Float = 0.75f,p:Boolean = false,assignTarget: Boolean = true) = {
        val net = model.asInstanceOf[Net];
        if (assignTarget){
            net.output_layers(0).target match {
                case t:IMat=> t<--irow(irow(0->net.datasource.opts.batchSize).data.map(_%10));
                case t:FMat=> t<-- row(irow(0->net.datasource.opts.batchSize).data.map(_%10));
            }
        }
        D.layers(0).output<--gdata;
        net.layers(0).output<--gdata;
        D.output_layers(0).target(?)=1;
//        accClassifier(?) = 0;
//        accDiscriminator(?) = 0;
        //var curScore = 0f;
        D.layers(D.layers.length-3) match {
            case dl:DropoutLayer=>dl.opts.frac = 1f;
            case _=>                
        }
//        gsteps.clear
        var curScore = 0f
        var t = 0;
//        while(curScore < targetScore && t < iter){
        while(t < iter) {
            net.forward;
            net.layers(0).deriv.clear;
            net.setderiv()
            net.backward(0, 0);
            D.forward;
            D.layers(0).deriv.clear;
            D.setderiv()
            D.backward(0, 0);
            curScore = mean(D.output_layers(0).output(1,?)).dv.toFloat;  
            val logit = D.layers(D.layers.length-2)
            val margin = mean(logit.output(1,?)-logit.output(0,?)).dv.toFloat;  
            _lrate <-- lrate// /((t+1f)^0.5f);
            _langevin(0,0) = langevin;
            _vWeight(0,0) = vWeight
            accClassifier ~ (accClassifier * 0.9f) + ((net.layers(0).deriv *@ net.layers(0).deriv) * 0.1f);
            accDiscriminator ~ (accDiscriminator * 0.9f) + ((D.layers(0).deriv *@ D.layers(0).deriv) * 0.1f);
            net.layers(0).deriv ~ net.layers(0).deriv / ((accClassifier+1e-8f)^0.5f);
            D.layers(0).deriv ~ D.layers(0).deriv / ((accDiscriminator+1e-8f)^0.5f);
            _dWeight(0,0) = dWeight;
            val grad = (net.layers(0).deriv *@ (1f - _dWeight)) + (_dWeight *@ D.layers(0).deriv)
            normrnd(0,langevin,noise);
            grad ~ grad + noise;
            grad ~ grad * 0.1f;
            momentum ~ momentum * 0.9f;
            momentum ~ momentum + grad
            net.layers(0).output~net.layers(0).output + (momentum *@ _lrate);
            max(net.layers(0).output,0,net.layers(0).output);
            min(net.layers(0).output,255,net.layers(0).output);
            val dims = net.layers(0).output.dims;
            val s = dims(1);
            val d = net.layers(0).output.reshapeView(dims(1),dims(2),dims(0),dims(3))
            if (t % 2 == 0) {
                d(0->(s-1),?,?,0) = (d(0->(s-1),?,?,0)*0.9f + d(1->s,?,?,0)*0.1f)//*0.5f
                d(?,0->(s-1),?,0) = (d(?,0->(s-1),?,0)*0.9f + d(?,1->s,?,0)*0.1f)//*0.5f
            }
            else {
                d(1->s,?,?,0) = (d(0->(s-1),?,?,0)*0.9f + d(1->s,?,?,0)*0.1f)//*0.5f
                d(?,1->s,?,0) = (d(?,0->(s-1),?,0)*0.9f + d(?,1->s,?,0)*0.1f)//*0.5f
            }
            t += 1   
            D.layers(0).output<--net.layers(0).output;
            if (p && t%10 == 0) {
                println(curScore,margin);
                _scale(0,0) = scale;
                val da = D.layers(0).output / _scale;
                val img = utils.filter2img(da-0.5f,D.opts.tensorFormat);
                plot.plot_image(img)   
            }
            gsteps+=margin
        }
        D.layers(D.layers.length-3) match {
            case dl:DropoutLayer=>dl.opts.frac = 0.5f
            case _=>                
        }
        D.clearUpdatemats
        gdata<--net.layers(0).output;
        gdata
    }
    
    def trainD() = {
        val ds = D.datasource;
        done = false
        while(!done){
            ds.reset;
            var here = 0
            ipass += 1
            while (ds.hasNext && !done) {
                //val data = if (here/ds.opts.batchSize % 2 == 0) generate(_net,targetScore = 0.7f); else generate(_net,targetScore = 0.2f);
                val data = mcmc(_net,targetScore = 0.7f);
                
                val da = FMat(data(?,?,?,?));
                _scale(0,0) = scale;
                val da2 = if (trans == null) da/_scale else trans(da)
                val img = utils.filter2img(da2-0.5f,D.opts.tensorFormat);
                plot.plot_image(img)
                                        
                val gscore = mean(D.output_layers(0).output(1,?)).dv.toFloat;
                gscores+=gscore;
                
                /*batch(1)(?) = 1;
                batch(0)(?,?,?,0->(batchSize/2)) = cpu(data(?,?,?,0->(batchSize/2)))
                batch(1)(?,0->(batchSize/2)) = 0;                
                D.layers(0).deriv.clear;
                D.dobatchg(batch,here,0);*/
                if (trainDis) {
                    val batchSize = ds.opts.batchSize;
                    here += batchSize;
                    val batch = ds.next;
                    D.layers(0).output(?,?,?,(batchSize/2)->batchSize) = batch(0)(?,?,?,0->(batchSize/2))
                    D.output_layers(0).target(?) = 1
                    D.output_layers(0).target(?,0->(batchSize/2)) = 0;
                    D.layers(0).deriv.clear;
                    D.forward;D.setderiv();D.backward(0, 0);
                    updater.update(ipass,here,0);
                    val dscore = mean(D.output_layers(0).score).dv.toFloat
                    dscores+=dscore;
                    println("Trained %d samples. Real samples score: %.3f, Generate samples score: %.3f".format(here,dscore,gscore));
                }
                else
                    println("Generate samples score: %.3f".format(gscore));
                
                /*val dloss = mean(D.output_layers(0).output(1,?)).dv.toFloat;                
                val dscore = mean(D.output_layers(0).score).dv.toFloat
                dscores+=dscore
                    
                D.output_layers(0).target(?) = 0;
                D.layers(0).deriv.clear;
                D.forward;D.setderiv();D.backward(0, 0);
                val gloss = mean(D.output_layers(0).output(1,?)).dv.toFloat;
                val gscore = mean(D.output_layers(0).score).dv.toFloat
                gscores+=gscore;
                updater.update(0,here,0);*/
                
            }
        }        
    }
    
    def launchTrain ={
        Future{trainD()}
    }
    
    def stop = {
        done = true
    }
    
    def reset(random:Boolean = true) {
        val data = gdata//_net.layers(0).output;
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
        opts.texp = 0.1f
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
        opts.texp = 0.1f
        Mat.useCache = true;
        Mat.useGPUcache = true;

        val net = new Net(opts);
        {
            import BIDMach.networks.layers.Node._;
            val convt = jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION

            val in = input;
            //val rin = relu(in)()
            //val scalef = constant(row(1f/256));
            //val inscale = rin *@ scalef

            val conv1 = conv(in)(w=5,h=5,nch=32,stride=1,pad=2,initfn=Net.gaussian, initv=0.1f,convType=convt,initbiasv=0.1f);
            val pool1 = pool(conv1)(w=2,h=2,stride=2,pad=0);
            //val bns1 = batchNormScale(pool1)();
            val relu1 = relu(pool1)();

            val conv2 = conv(relu1)(w=5,h=5,nch=64,stride=1,pad=2,convType=convt,initfn=Net.gaussian,initv=0.1f, initbiasv=0.1f);   
            val pool2 = pool(conv2)(w=2,h=2,stride=2,pad=0);
            //val bns2 = batchNormScale(pool2)();
            val relu2 = relu(pool2)();

            val fc3 = linear(relu2)(outdim=1024,initfn=Net.gaussian,initv=0.1f, initbiasv=0.1f);
            val relu3 = relu(fc3)();

//            val fc4 = linear(relu3)(outdim=84,initv=1e-1f);
//            val relu4  = relu(fc4)();
            
            val drop6 =  dropout(relu3)(0.5f);

            val fc5  = linear(drop6)(outdim=2,initfn=Net.gaussian,initv=0.1f, initbiasv=0.1f);
            val out = softmaxout(fc5)(scoreType=1);

            val nodes = (in     \ null    \ null   \ null    on
                         conv1  \ pool1   \ null   \ relu1   on
                         conv2  \ pool2   \ null   \ relu2   on
                         fc3    \ relu3   \ null   \ null    on
                         drop6    \ null   \ null    \ null    on
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
        class MyOpts extends Net.Opts with FileSource.Opts with ADAGrad.Opts;
        val traindir = "/code/BIDMach/data/ImageNet/train/";
        val traindata = traindir+"partNCHW%04d.bmat.lz4";
        val trainlabels = traindir+"label%04d.imat.lz4";
        val opts = new MyOpts;
        val ds = FileSource(traindata, trainlabels, opts);
        val updater = new ADAGrad(opts);
        opts.batchSize = 1;
        opts.hasBias = true;
        opts.tensorFormat = Net.TensorNCHW;
        opts.convType = Net.CrossCorrelation;
        opts.lrate = 1e-4f;
        opts.vel_decay = 0.9f;
        opts.gsq_decay = 0.99f;
        opts.texp = 0.1f
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

            val fc8  =      linear(drop7)(outdim=2,initfn=Net.gaussian,initv=0.01f,initbiasv=1f);
            val out =       softmaxout(fc8)(scoreType=1,lossType=1);

            opts.nodeset=Net.getDefaultNodeSet
        }
        ds.init
        net.bind(ds)
        net.init;
        updater.init(net)
        //load(net,"models/alex32p/");
        (net,updater)
    }
}