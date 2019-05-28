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
    Use activation maximization to synthesis images based on a particular neuron.
*/

class Synthesis(val modelname: String = "cifar",val opts:Synthesis.Opts = new Synthesis.Options) extends Visualization{
    val plot = new ImageArray("Input Synthesis");
    val plot2 = new ImageArray("Enlarge Image");    
    val zero = irow(0);

    var _net: Net = null;
    var D: Net = null;        //Discriminator
    var updater:Grad = null;  //Updater for the discriminator
    
    val dscores : ListBuffer[Float] = new ListBuffer[Float];
    val gscores : ListBuffer[Float] = new ListBuffer[Float];
    var accClassifier: Mat = null;
    var accDiscriminator: Mat = null;
    var momentum: Mat = null;
    var noise: Mat = null;

    var done = false;
    var ipass = 0;
    var mcmcSteps = 0;
    var resetFlag = false;            
    var gsteps : ListBuffer[Float] = new ListBuffer[Float];
    var gdata : Mat = null;
    val vizs = new ListBuffer[Visualization];

    // Helper matrix for caching purpose
    var _lrate: Mat = null;
    var _langevin: Mat = null;
    var _vWeight: Mat = null;
    var _dWeight: Mat = null;
    var _scale: Mat = null;
    var _base: Mat = null;    
    var _l2lambda: Mat = null;
    var _dissimilarity: Mat = null;
    var _wClip: Mat = null;
    var _averagingWeight: Mat = null;
    var averaging: Mat = null;
    var _mask: Mat = null;
    var _averagingModelmats: Array[Mat] = null;
    var selecter = irow(0);
        
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
        val net = model.asInstanceOf[Net];
        _net = net;
        _lrate = net.layers(0).output.zeros(1,1);
        _langevin = net.layers(0).output.zeros(1,1);  
        _dWeight = net.layers(0).output.zeros(1,1);  
        _vWeight = net.layers(0).output.zeros(1,1);  
        _scale = net.layers(0).output.zeros(1,1);  
        _base = net.layers(0).output.zeros(1,1);  
        _l2lambda = net.layers(0).output.zeros(1,1);  
        _dissimilarity = net.layers(0).output.zeros(1,1); 
        _wClip = net.layers(0).output.zeros(1,1); 
        _averagingWeight = net.layers(0).output.zeros(1,1); 
        accClassifier = net.layers(0).output.zeros(net.layers(0).output.dims);
        accDiscriminator = net.layers(0).output.zeros(net.layers(0).output.dims);
        momentum = net.layers(0).output.zeros(net.layers(0).output.dims);
        noise = net.layers(0).output.zeros(net.layers(0).output.dims);
        gdata = net.layers(0).output.zeros(net.layers(0).output.dims);
        gdata<--rand(gdata.nrows,gdata.ncols).reshapeView(gdata.dims)*256f;
        averaging = net.layers(0).output.zeros(net.layers(0).output.dims);
        averaging<--gdata;
        interval  = opts.updateInterval;
        accClassifier(?) = 0;
        accDiscriminator(?) = 0;
        momentum(?) = 0;

        _mask = net.layers(0).output.zeros(net.layers(0).output.dims);
        if (_mask.dims(3) == 100)
            for(i<-0 until 10)
                _mask(?,?,?,(i*10)->(i+1)*10) = 0.1f*(i+1)
        
        _averagingModelmats = new Array[Mat](net.modelmats.length);
        for(i<-0 until net.modelmats.length) {
            _averagingModelmats(i) = net.modelmats(i).zeros(net.modelmats(i).dims)
            _averagingModelmats(i) <-- net.modelmats(i)
        }
           
        val batchSize = model.datasource.opts.batchSize
        val (_D,_updater) = modelname match {
            case "cifar" => Synthesis.buildCifarDiscriminator(opts.realImagesPath,batchSize); 
            case "mnist" => Synthesis.buildMnistDiscriminator(opts.realImagesPath,batchSize);
            case "imagenet" => Synthesis.buildImageNetDiscriminator(opts.realImagesPath,batchSize);
        }            
        D = _D;updater = _updater;
        
        setInputGradient(net);
        setInputGradient(D);        
        if (opts.pretrainedDiscriminatorPath != null)
            Synthesis.load(D,opts.pretrainedDiscriminatorPath)

        show(net.layers(0).output);
        
            //val convLayers = List(5,9,13,15,17)
        val convLayers = net.layers.zipWithIndex.filter(_._1.isInstanceOf[ConvLayer]).map(_._2);

        val filterBox = plot.add_combobox(irow(0->256).data.map("ConvFilter "+_),
                              (i:Int,v:String)=>{
                                  if (convLayers.contains(opts.endLayer) && net.layers(opts.endLayer).deriv.dims(0)>i) {
                                      opts.derivFunc = (a:Layer)=>{
                                          val m = a.deriv;m.set(0f);
                                          val rm = m.reshapeView(m.dims(1),m.dims(2),m.dims(0),m.dims(3));
                                          rm(m.dims(1)/2,m.dims(2)/2,i,0->(m.dims(3)/2))=1f;
                                          rm(m.dims(1)/2,m.dims(2)/2,i,(m.dims(3)/2)->m.dims(3))=1f;      
                                      //Weird bug, 4d slice for GMat can't support bigger than 64 items if the last index is ?
                                      }
                                      resetFlag = true
                                  }
                              });
        plot.add_combobox(irow(1->(convLayers.length+1)).data.map("Conv"+_),
                              (i:Int,v:String)=>{
                                  opts.endLayer = convLayers(i);
                                  opts.derivFunc = (a:Layer)=>{
                                      val m = a.deriv;m.set(0f);
                                      var ind = irow(irow(0->m.dims(3)).data.map(_%m.dims(0))) + irow(0->m.dims(3)) * m.dims(0)
                                      val rm = m.reshapeView(m.dims(1),m.dims(2),m.dims(0)*m.dims(3));
                                      rm(m.dims(1)/2,m.dims(2)/2,ind(0->m.dims(3)/2))=1f;
                                      rm(m.dims(1)/2,m.dims(2)/2,ind((m.dims(3)/2)->m.dims(3)))=1f;
//                                      rm(m.dims(1)/2,m.dims(2)/2,0,)=1f
                                      //Weird bug, 4d slice for GMat can't support bigger than 64 items if the last index is ?
                                  }
                                  resetFlag = true;
                                  filterBox.setSelectedItem("ConvFilter 0")
                              });
        plot.add_combobox(irow(0->100).data.map("Show result "+_),
                              (i:Int,v:String)=>{opts.detailed = i})
        plot.add_slider("iter",(x:Int)=>{opts.iter=(x+1);opts.iter},1,0);
        plot.add_slider("scale",(x:Int)=>{opts.scale=x/10f;opts.scale},10,2);
        plot.add_slider("base",(x:Int)=>{opts.base=x*4;opts.base},32,0);
        plot.add_slider("lrate",(x:Int)=>{opts.lrate=(10f^(x/20f-4))(0);opts.lrate},60,4);
        plot.add_slider("noise",(x:Int)=>{opts.langevin=(10f^(x/20f-4))(0);opts.langevin},60,4);
        plot.add_slider("L2 norm",(x:Int)=>{opts.l2lambda=(10f^(x/20f-4))(0);opts.l2lambda},60,4);
        //plot.add_slider("Dissimilarity",(x:Int)=>{opts.dissimilarity=(10f^(x/20f-7))(0);opts.dissimilarity},0,7);
        plot.add_slider("discriminatorWeight",(x:Int)=>{opts.dWeight=x/100f;opts.dWeight},0);        
        plot.add_slider("averagingTime",(x:Int)=>{opts.averagingTime=math.exp(x/16f).toFloat;opts.averagingTime},0,2); 
        //plot.add_slider("modelAveragingTime",(x:Int)=>{opts.modelAveragingTime=math.exp(x/10).toFloat;opts.averagingTime},0,2);        
    }

    override def doUpdate(model:Model, mats:Array[Mat], ipass:Int, pos:Long) = {        
        //resetFlag = true
        
        for(i<-0 until model.modelmats.length){
            _averagingWeight(0,0) = 1-1f/opts.modelAveragingTime
            _averagingModelmats(i) ~ _averagingModelmats(i) *@ _averagingWeight;
            _averagingModelmats(i) ~ _averagingModelmats(i) + (model.modelmats(i) *@ (1f - _averagingWeight))
        }
        val tmp = model.modelmats;
        model._modelmats = _averagingModelmats
        val data = mcmc(_net);
        model._modelmats = tmp;
        if (opts.displayAveraging)
            show(averaging)
        else
            show(data)
        /**val net = model.asInstanceOf[Net];
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
        }*/
    }
    
    def backward(net:Net,end:Int,setDeriv:Layer=>Unit = null) {
        var i = end;
        if (i < 0) 
            i += net.layers.length;
        if (setDeriv == null)
            net.layers(i).deriv.set(1f);
        else
            setDeriv(net.layers(i))
        while (i>=1) {
            if (opts.guidebp && (net.layers(i).deriv != null) )
                net.layers(i).deriv~net.layers(i).deriv*@(net.layers(i).output>=0);                        
            net.layers(i).backward(0, 0);
            if (opts.guidebp) {
                if (net.layers(i).input != null) {
                    max(net.layers(i).inputDeriv,0,net.layers(i).inputDeriv);
                }
            }
            i -= 1;
        }
    }
    
    def show(data:Mat) = {
        _scale(0,0) = opts.scale;
        _base(0,0) = opts.base;
        val data_ = if (modelname=="imagenet") data(?,?,?,0->4) 
            else 
               data               
        val da = (data_ *@ _scale + _base);
        plot.plot_image(da)
        selecter(0,0) = opts.detailed
        val d2 = if (modelname=="imagenet")  data(?,?,?,selecter) else 
            mean(data.reshapeView(data.nrows*10,10),2).reshapeView(data.dims(0),data.dims(1),data.dims(2),10);
        plot2.plot_image(d2 *@ _scale + _base)
            
    }
    
    def mcmc(model:Model,targetScore:Float = 0.75f,p:Boolean = false,assignTarget: Boolean = true) = {
        if (mcmcSteps % opts.resetInterval == 0) {
            reset()
        }
        mcmcSteps += 1;
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
        //var curScore = 0f;
        D.layers(D.layers.length-3) match {
            case dl:DropoutLayer=>dl.opts.frac = 1f;
            case _=>                
        }
//        gsteps.clear
        var curScore = 0f
        var t = 0;
        //println("here")
//        while(curScore < targetScore && t < iter){
        while(t < opts.iter) {
            if (resetFlag) {
                val d = D.layers(0).output;
                d<--rand(d.nrows,d.ncols).reshapeView(d.dims)*256f;
                net.layers(0).output<--d;
                resetFlag = false;             
            }
            if (opts.resetAveraging) {
                averaging<--net.layers(0).output;
                opts.resetAveraging = false;
            }
            net.forward;
            net.layers(0).deriv.clear;
            net.setderiv();
            backward(net, opts.endLayer,opts.derivFunc);
            //net.backward(0, 0);
                
            D.forward;
            D.layers(0).deriv.clear;
            D.setderiv()
            D.backward(0, 0);
            //println("here2")
            curScore = mean(D.output_layers(0).output(1,?)).dv.toFloat;  
            val logit = D.layers(D.layers.length-2)
            val margin = mean(logit.output(1,?)-logit.output(0,?)).dv.toFloat;  
            _lrate <-- opts.lrate * 256f// /((t+1f)^0.5f);
            _langevin(0,0) = opts.langevin;
            _vWeight(0,0) = opts.vWeight
            accClassifier ~ (accClassifier * 0.9f) + ((net.layers(0).deriv *@ net.layers(0).deriv) * 0.1f);
            accDiscriminator ~ (accDiscriminator * 0.9f) + ((D.layers(0).deriv *@ D.layers(0).deriv) * 0.1f);
            net.layers(0).deriv ~ net.layers(0).deriv / ((accClassifier+1e-8f)^0.5f);
            D.layers(0).deriv ~ D.layers(0).deriv / ((accDiscriminator+1e-8f)^0.5f);
            _dWeight(0,0) = opts.dWeight;
            val grad = (net.layers(0).deriv *@ (1f - _dWeight)) + (_dWeight *@ D.layers(0).deriv)
            normrnd(0,opts.langevin/(1f+t*t),noise);
            _l2lambda(0,0) = opts.l2lambda*2f/256f;
            //noise ~ noise *@ _mask;
            grad ~ grad + noise;
            grad ~ grad - (net.layers(0).output *@ _l2lambda);
            val img = net.layers(0).output.reshapeView(net.layers(0).output.nrows,net.layers(0).output.ncols);
            _dissimilarity(0,0) = opts.dissimilarity/(img.ncols-1);
            val tmp = (sum(img,2)-img)*@ _dissimilarity                
            grad ~ grad - (tmp.reshapeView(grad.dims))
            grad ~ grad * 0.1f;
            momentum ~ momentum * 0.9f;
            momentum ~ momentum + grad
            net.layers(0).output~net.layers(0).output + (momentum *@ _lrate);
            if (opts.clipping){
                max(net.layers(0).output,0,net.layers(0).output);
                min(net.layers(0).output,255,net.layers(0).output);
            }
            val dims = net.layers(0).output.dims;
            val s = dims(1);
            val d = net.layers(0).output.reshapeView(dims(1),dims(2),dims(0),dims(3))
            val smooth_id  = 0  // Use ? if want to smooth all the images.
            if (t % 2 == 0) {
                d(0->(s-1),?,?,smooth_id) = (d(0->(s-1),?,?,smooth_id)*0.5f + d(1->s,?,?,smooth_id)*0.5f)// *0.5f
                d(?,0->(s-1),?,smooth_id) = (d(?,0->(s-1),?,smooth_id)*0.5f + d(?,1->s,?,smooth_id)*0.5f)// *0.5f
            }
            else {
                d(1->s,?,?,smooth_id) = (d(0->(s-1),?,?,smooth_id)*0.5f + d(1->s,?,?,smooth_id)*0.5f)// *0.5f
                d(?,1->s,?,smooth_id) = (d(?,0->(s-1),?,smooth_id)*0.5f + d(?,1->s,?,smooth_id)*0.5f)// *0.5f
            }
            t += 1
            D.layers(0).output<--net.layers(0).output;
            if (p && t%10 == 0) {
                println(curScore,margin);
                if (opts.displayAveraging)
                    show(averaging)
                else
                    show(D.layers(0).output)
            }
            gsteps+=margin
        }
        _averagingWeight(0,0) = 1-1f/opts.averagingTime;
        averaging ~ averaging *@ _averagingWeight;
        averaging ~ averaging + (net.layers(0).output *@ ( 1 - _averagingWeight ) )

        D.layers(D.layers.length-3) match {
            case dl:DropoutLayer=>dl.opts.frac = 0.5f
            case _=>
        }
        net.clearUpdatemats
        D.clearUpdatemats
        gdata<--net.layers(0).output;
        gdata
    }
    
    def start() = {
        val ds = D.datasource;
        done = false
        while(!done){
            ds.reset;
            var here = 0
            ipass += 1
            while (ds.hasNext && !done) {
                //val data = if (here/ds.opts.batchSize % 2 == 0) generate(_net,targetScore = 0.7f); else generate(_net,targetScore = 0.2f);
                val data = mcmc(_net,targetScore = 0.7f);
                
                if (opts.displayAveraging)
                    show(averaging)
                else
                    show(data)
                                        
                val gscore = mean(D.output_layers(0).output(1,?)).dv.toFloat;
                gscores+=gscore;
                
                /**batch(1)(?) = 1;
                batch(0)(?,?,?,0->(batchSize/2)) = cpu(data(?,?,?,0->(batchSize/2)))
                batch(1)(?,0->(batchSize/2)) = 0;                
                D.layers(0).deriv.clear;
                D.dobatchg(batch,here,0); */
                if (opts.trainDis) {
                    val batchSize = ds.opts.batchSize;
                    here += batchSize;
                    val batch = ds.next;
                    val data = batch(0) match {
                        case m:FMat=>m;
                        case m:BMat=>unsignedFloat(m,true);
                    }
                    D.layers(0).output(?,?,?,(batchSize/2)->batchSize) = data(?,?,?,0->(batchSize/2))
                    D.output_layers(0).target(?) = 1
                    D.output_layers(0).target(?,0->(batchSize/2)) = 0;
                    D.layers(0).deriv.clear;
                    D.clearUpdatemats
                    D.forward;D.setderiv();D.backward(0, 0);
                    updater.update(ipass,here,0);
                    if (opts.wClip > 0){
                        _wClip(0,0) = opts.wClip;
                        for(i<-0 until D.modelmats.length)
                            min(D.modelmats(i),_wClip,D.modelmats(i))
                        _wClip(0,0) = -opts.wClip;
                        for(i<-0 until D.modelmats.length)
                            max(D.modelmats(i),_wClip,D.modelmats(i))
                    }
                    val dscore = mean(D.output_layers(0).score).dv.toFloat
                    dscores+=dscore;
                    vizs.foreach(_.update(D,batch,ipass,here))
                    if (opts.printInfo)
                        println("Trained %d samples. Real samples score: %.3f, Generate samples score: %.3f".format(here,dscore,gscore));
                }
                else
                    if (opts.printInfo)
                        println("Generate samples score: %.3f".format(gscore));
                
                /**val dloss = mean(D.output_layers(0).output(1,?)).dv.toFloat;                
                val dscore = mean(D.output_layers(0).score).dv.toFloat
                dscores+=dscore
                    
                D.output_layers(0).target(?) = 0;
                D.layers(0).deriv.clear;
                D.forward;D.setderiv();D.backward(0, 0);
                val gloss = mean(D.output_layers(0).output(1,?)).dv.toFloat;
                val gscore = mean(D.output_layers(0).score).dv.toFloat
                gscores+=gscore;
                updater.update(0,here,0); */                
            }
        }        
    }
    
    def launch ={
        Future{start()}
    }
    
    def stop = {
        done = true
    }
    
    def startTrain = {
        done = false;
        opts.trainDis = true;
        opts.resetInterval = 10;
        launch
    }
    
    def startGenerate = {
        done = false;
        opts.trainDis = false;
        opts.resetInterval = 1000000000;
        launch
    }
    
    def reset(random:Boolean = true) {
        val data = gdata//_net.layers(0).output;
        if (random){
            val scale = 256f//maxi(data).dv.toFloat
            data<--rand(data.nrows,data.ncols).reshapeView(data.dims)*scale;
        }
        else{
            _net.datasource.reset;
            val batch = _net.datasource.next 
            val d = batch(0) match {
                        case m:FMat=>m;
                        case m:BMat=>unsignedFloat(m,true);
                    }
            data<--d;
            _net.output_layers(0).target<--batch(1)
                
        }
    }
    
    def plot(v:Visualization) {
        vizs += v;
    }
}

object Synthesis {
    trait Opts extends Model.Opts {
        var iter = 100
        var lrate = 10f;
        var langevin = 0.1f
        var vWeight = 0.9f;
        var dWeight = 0.5f;
        var scale = 1f;
        var base = 0f;
        var l2lambda = 0f;
        var dissimilarity = 0f;
        var trans : Mat=>Mat = null;
        var trainDis = true;
        var resetInterval = 1000000000;
        var updateInterval = 100;
        var endLayer = -1; // -1 means the last layer, -2 means the second-to-last, 0 means the first.
        var derivFunc: Layer=>Unit = null;
        var guidebp = false;
        var printInfo = true;
        var realImagesPath:String = null;
        var pretrainedDiscriminatorPath:String = null;  
        var wClip  = -1f;
        var clipping = false;
        var detailed = 1;
        var resetAveraging = false;
        var displayAveraging = true;
        var averagingTime = 1f
        var modelAveragingTime = 1f
      }
    
    class Options extends Opts {}
    
    def load(net:Net,path:String) {
        for (i <- 0 until net.modelmats.length) {
            val data = loadMat(path+"/modelmat%02d.lz4" format i);
            net.modelmats(i)<--data
        }
    }
    
    def buildCifarDiscriminator(datadir:String,batchSize:Int) = {
        class MyOpts extends Net.Opts with FileSource.Opts with ADAGrad.Opts;
        //val datadir = "/code/BIDMach/data/CIFAR10/parts/"
        val trainfname = datadir + "trainNCHW%d.fmat.lz4";
        val labelsfname = datadir + "labels%d.imat.lz4";
        val opts = new MyOpts;
        val ds = FileSource(trainfname, labelsfname, opts);
        val updater = new ADAGrad(opts)
        opts.batchSize = batchSize;
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
            val convt = jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
            Net.initDefaultNodeSet;
            val in = input;
            val scalef = constant(row(0.01f))(true);
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

            /**val nodes = (in     \ scalef \ inscale on
                         conv1  \ pool1  \ relu1  on
                         conv2  \ pool2  \ relu2  on
                         conv3  \ pool3  \ null   on
                         fc3    \ out    \ null   ).t
            opts.nodemat = nodes; */
            opts.nodeset=Net.getDefaultNodeSet
        }        
        ds.init
        net.bind(ds)
        net.init;
        updater.init(net)
        (net,updater)
    }
    
    def buildMnistDiscriminator(traindir:String,batchSize: Int) = {
        class MyOpts extends Net.Opts with MatSource.Opts with ADAGrad.Opts;
        //val traindir = "/code/BIDMach/data/MNIST/"
        val train0 = loadIDX(traindir+"train-images-idx3-ubyte").reshapeView(1,28,28,60000);
        val trainlabels0 = loadIDX(traindir+"train-labels-idx1-ubyte").reshapeView(1,60000);
        val opts = new MyOpts;
        val ds = new MatSource(Array(train0, trainlabels0), opts);
        val updater = new ADAGrad(opts);
        opts.batchSize = batchSize;
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
            Net.initDefaultNodeSet;
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

            /*val nodes = (in     \ null    \ null   \ null    on
                         conv1  \ pool1   \ null   \ relu1   on
                         conv2  \ pool2   \ null   \ relu2   on
                         fc3    \ relu3   \ null   \ null    on
                         drop6    \ null   \ null    \ null    on
                     fc5    \ out     \ null   \ null).t
            opts.nodemat = nodes; */
            opts.nodeset=Net.getDefaultNodeSet
        }        
        ds.init
        net.bind(ds)
        net.init;
        updater.init(net)
        (net,updater)
    }
    
    def buildImageNetDiscriminator(traindir:String,batchSize:Int) = {
        class MyOpts extends Net.Opts with FileSource.Opts with ADAGrad.Opts;
//        val traindir = "/code/BIDMach/data/ImageNet/train/";
        val traindata = traindir+"partNCHW%04d.bmat.lz4";
        val trainlabels = traindir+"label%04d.imat.lz4";
        val opts = new MyOpts;
        val ds = FileSource(traindata, trainlabels, opts);
        
/**        class MyOpts extends Net.Opts with MatSource.Opts with ADAGrad.Opts;
        val traindir = "/code/BIDMach/data/ImageNet/train/";
        val opts = new MyOpts;
        val ds = new MatSource(Array(loadBMat("data/ImageNet/classes/dataNCHW1.bmat.lz4"),IMat(1,100)+1),opts)*/
        val updater = new ADAGrad(opts);
        opts.batchSize = batchSize;
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
            val meanv =     const(means)(true);
            val din =       in - meanv;
            val scalef =    const(row(0.01f))(true);
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
