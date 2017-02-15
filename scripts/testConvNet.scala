/* Quanlai Li
* 2017-02-08
*/

:silent
import BIDMat.TMat

/* options */

val opts = new Net.FDSopts;

//TF is using MNIST data
val fn1 = mdir+"trainsortedx%02d.smat.lz4";
val fn2 = mdir+"trainlabel%02d.fmat.lz4";
opts.fnames = List(FileSource.simpleEnum(fn1,1,0),FileSource.simpleEnum(fn2,1,0));


// TF settings 
//training iterations = 200000
//display step = 10

//n_input = 784 # MNIST data input (img shape: 28*28)
//n_classes = 10 # MNIST total classes (0-9 digits)
//dropout = 0.75 # Dropout, probability to keep units

opts.eltsPerSample = 500;
opts.nend = 90
opts.batchSize= 128 //same with TF
opts.npasses = 4

opts.lrate = 0.001f //same with TF
opts.texp = 0.3f
opts.pstep = 0.001f
opts.aopts = opts
//opts.reg1weight = 0.0001
//opts.hasBias = true
opts.links = iones(1,1);
opts.nweight = 1e-4f
opts.lookahead = 0
opts.autoReset = false



import BIDMach.networks.layers._
import BIDMach.networks.layers.Node
import BIDMach.networks.layers.Node._

/* neural network design */

val tshape = 0.2f
val tms = Net.powerShape(tshape)_;
val shape = irow(200,120,80,50,1);
val in = input;
val dropout = 0.75; //same

//convolution layer
val conv1= new ConvolutionNode{inputs(0) = in; outdim = shape(0); hasBias = opts.hasBias; aopts = opts.aopts; tmatShape = tms};

//convolution layer
val conv2=convolution(conv1)(outdim = shape(1), hasBias = opts.hasBias, aopts = opts.aopts);

//fully connected layer
//reshape conv2 output to fit fully connected layer input
var fc1 = convolution(conv2)(outdim = shape(2), hasBias = opts.hasBias, aopts = opts.aopts);

//Apply Dropout
var drop = dropout(fc1, dropout);

//Output, class prediction
val out = glm(drop)(opts.links);
val layers = Array(in, conv1, conv2, fc1, drop, out)

opts.nodeset = new NodeSet(layers);
opts.what;
println(tshape.toString);
println(shape.toString);
val zzero = gzeros(1,1)

val nn = new Learner(new FileSource(opts),
		     new Net(opts),
		     null,
		     new Batch, 
		     null,
		     opts);


/* train */

val model = nn.model.asInstanceOf[Net]
nn.train

/* test */

val testdata = loadSMat(mdir+"trainsortedx%02d.smat.lz4" format opts.nend);
val testlabels = loadFMat(mdir+"trainlabel%02d.fmat.lz4" format opts.nend);

val (mm, mopts) = Net.predictor(model, testdata);
mm.predict

/* benchmark */

val preds=FMat(mm.preds(0))

val ll = DMat(ln(preds *@ testlabels + (1-preds) *@ (1-testlabels)))
val rc = roc(preds, testlabels, 1-testlabels, 1000);

:silent

(mean(ll), mean(rc))