
import BIDMach.networks.TransformerLT
import BIDMach.networks.layers._

val ddir = "/code/BIDMach/data/wikitext/"
val fname = ddir + "train/part%04d.imat.lz4"

val (nn, opts) = TransformerLT.learner(fname);

opts.lrate = 2e-4f
opts.seqlength = 1024
opts.batchSize = opts.seqlength
opts.npasses = 40
opts.degree = 64
opts.depth = 4
opts.nheads = 8
opts.dim = 512
opts.indim = opts.dim
opts.outdim = opts.dim
opts.dropout=0.9f;
opts.texp = 0f
opts.gsq_decay = 0.999f;
//opts.nvocab = 32
opts.scoreType = SoftmaxOutputLayer.CrossEntropyScore
opts.pstep = 0.01f
opts.useCache = false
opts.useGPUcache = true

val tt = nn.model.asInstanceOf[TransformerLT]

//nn.train
nn.launchTrain
Thread.sleep(2000)


val net = tt.txNets(0)
val fe = tt.frontEnd
val be = tt.backEnd

