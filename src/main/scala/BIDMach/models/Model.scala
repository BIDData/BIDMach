package BIDMach.models

import BIDMat.{BMat,Mat,SBMat,CMat,CSMat,DMat,FMat,FFilter,GMat,GFilter,GDMat,GIMat,GLMat,GSMat,GSDMat,HMat,IMat,JSON,LMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.datasinks._
import scala.collection.mutable.ListBuffer

/**
 * Abstract class with shared code for all models
 *
 * Models are saved as separate files into a directory. The model save pathname should contain a trailing "/" and name this parent directory.
 */

abstract class Model(val opts:Model.Opts = new Model.Options) extends Serializable {

  var datasource:DataSource = null;

  var datasink:DataSink = null;

  var _modelmats:Array[Mat] = null;
  
  var _lr_scales:FMat = null;

  var parent_model:Model = null;
  
  var elastic_tmp:Mat = null;
  
  var allreduce_tmp:Mat = null;

  def modelmats:Array[Mat] = {
    if (_modelmats != null) {
      _modelmats
    } else if (parent_model != null) {
      parent_model._modelmats
    } else {
      null
    }
  }
  
  def lr_scales:FMat = {
    if (_modelmats != null) {
      _lr_scales
    } else if (parent_model != null) {
      parent_model._lr_scales
    } else {
      null
    }
  }

  def setmodelmats(a:Array[Mat]) = {
    _modelmats = a;
    _lr_scales = ones(1, _modelmats.length);
  }

  var updatemats:Array[Mat] = null;

  // For Allreduce: the local indices
  var indexmats:Array[Mat] = null;

  // For Allreduce: cached local matrices:
  var sendmats:Array[Mat] = null;

  var recvmats:Array[FMat] = null;
  var maxmats:Array[Mat] = null;

  var mats:Array[Mat] = null;

  var gmats:Array[Mat] = null;

  var omats:Array[Mat] = null;

  var ogmats:Array[Mat] = null;

  var useGPU = false;

  var useDouble = false;

  var putBack = -1;

  var refresh = true;

  var runtimes:FMat = null;

  def mergeModelFn(models:Array[Model], mm:Array[Mat], um:Array[Mat], istep:Long, elastic_weight:Float = 1f, weights:FMat = null):Unit = {
    val mlen = models(0).modelmats.length;
    val thisGPU = getGPU;
    for (j <- 0 until mlen) {
      mm(j).clear
      for (i <- 0 until models.length) {
        if (useGPU && i < Mat.hasCUDA) setGPU(i);
      	um(j) <-- models(i).modelmats(j);
      	if (weights.asInstanceOf[AnyRef] != null) {
      	  um(j) ~ um(j) *@ weights(i)
      	}
      	mm(j) ~ mm(j) + um(j);
      }
      if (weights.asInstanceOf[AnyRef] == null) {
      	mm(j) ~ mm(j) * (1f/models.length);
      }
      for (i <- 0 until models.length) {
        if (elastic_weight != 1f) {
        	um(j) <-- models(i).modelmats(j);
          um(j) ~ um(j) *@ ((1f-elastic_weight)/elastic_weight);
          um(j) ~ um(j) + mm(j);
          um(j) ~ um(j) *@ elastic_weight;
          models(i).modelmats(j) <-- um(j);
        } else {
        	models(i).modelmats(j) <-- mm(j);
        }
    	}
    }
    setGPU(thisGPU);
  }

  def mergeModelPassFn(models:Array[Model], mm:Array[Mat], um:Array[Mat], ipass:Int) {}
  
  def preAllreduce(mm:Array[Mat], istep:Long, weights:FMat = null, imodel:Int, nmodels:Int):Unit = {
    val mlen = modelmats.length;
    val wt = if (weights.asInstanceOf[AnyRef] != null) {
      weights(imodel);
    } else {
      1f / nmodels;
    }
    for (i <- 0 until mlen) {
    	if (modelmats(i).asInstanceOf[AnyRef] != null) {
    		mm(i) ~ modelmats(i) *@ wt; 
    	}
    }
  }
  
  def sizeTo(tmp:Mat, m:Mat):Mat = {
    tmp match {
    case gm:GMat => new GMat(m.dims.data, gm.pdata, tmp.length);
    case gm:GDMat => new GDMat(m.dims.data, gm.pdata, tmp.length);
    case gm:GIMat => new GIMat(m.dims.data, gm.pdata, tmp.length);
    case gm:GLMat => new GLMat(m.dims.data, gm.pdata, tmp.length);
    case fm:FMat => new FMat(m.dims.data, fm.data);
    case fm:DMat => new DMat(m.dims.data, fm.data);
    case fm:IMat => new IMat(m.dims.data, fm.data);
    case fm:LMat => new LMat(m.dims.data, fm.data);
    }
  }
  
  def postAllreduce(mm:Array[Mat], istep:Long, elastic_weight:Float):Unit = {
    val mlen = modelmats.length;
    if (elastic_tmp.asInstanceOf[AnyRef] == null) {
      val maxlen = modelmats.map((x) => if (x.asInstanceOf[AnyRef] != null) x.length else 0).reduce((x,y) => math.max(x,y));
      elastic_tmp = modelmats(0).zeros(1, maxlen);
    }
    for (i <- 0 until mlen) {
    	if (modelmats(i).asInstanceOf[AnyRef] != null) {
    		val tmp = sizeTo(elastic_tmp, modelmats(i));
    		tmp ~ mm(i) - modelmats(i);
    		tmp ~ tmp *@ elastic_weight;
    		modelmats(i) ~ modelmats(i) + tmp;
    	}
    }
  }
  
  def allreduce(mms:Array[Array[Mat]], models:Array[Model]):Unit = {
    val nthreads = mms.length;
    val nmodels = mms(0).length;
    val toMats = new Array[GMat](nthreads);
    val fromMats = new Array[GMat](nthreads);
    if (useGPU) {
    	val thisGPU = getGPU;
    	for (j <- 0 until nthreads) {
    	  setGPU(j);
    		if (models(j).allreduce_tmp.asInstanceOf[AnyRef] == null) {
    			val maxlen = models(j).modelmats.map((x) => if (x.asInstanceOf[AnyRef] != null) x.length else 0).reduce((x,y) => math.max(x,y));
    			models(j).allreduce_tmp = models(j).modelmats(0).zeros(1, maxlen);
    		}
    	} 
    	setGPU(thisGPU);
    }
  	for (i <- 0 until nmodels) {
  	  mms(0)(i) match {
  	    case b:GMat => {
  	      for (j <- 0 until nthreads) {
  	        fromMats(j) = mms(j)(i).asInstanceOf[GMat];
  	        toMats(j) = sizeTo(models(j).allreduce_tmp, fromMats(j)).asInstanceOf[GMat];
  	        ncclAllReduce(fromMats, toMats);
  	        fromMats(j) <-- toMats(j);
  	      }
  	    }
  	    case a:FMat => {
  	      for (j <- 1 until nthreads) {
  	        mms(j)(i) ~ mms(j)(i) + mms(j-1)(i);
  	      }
  	      for (j <- 0 until (nthreads-1)) {
  	        mms(j)(i) <-- mms(nthreads-1)(i);
  	      }
  	    }
  	  }
  	}
  }

  def copyTo(mod:Model) = {
    mod.datasource = datasource;
    mod._modelmats = modelmats;
    mod.updatemats = updatemats;
//    mod.mats = mats;
    mod.gmats = gmats;
    mod.omats = omats;
    mod.ogmats = ogmats;
  }

  def copyFrom(mod:Model) = {
    setmodelmats(new Array[Mat](mod.modelmats.length));
    for (i <- 0 until modelmats.length) {
      modelmats(i) = mod.modelmats(i);
    }
  }

  def saveMetaData(fname:String) = {}

  def deleteMetaData(fname:String) = {}

  def loadMetaData(fname:String) = {}

  def guardOptions():AnyRef = null

  def unguardOptions(guard:AnyRef) = { }

  /**
   * Save the model to a given path. This is normally a directory (which is created if needed).
   * Otherwise the model and metadata filenames are concatenated to form the save file paths.
   */

  def save(fname:String) = {
    import java.io._
    val metadataname = new File(fname+"options.json");
    val parentdir = metadataname.getParentFile();
    if (parentdir != null) parentdir.mkdirs();
    val pw = new PrintWriter(metadataname);
    val guard = guardOptions();
    pw.print(JSON.toJSON(opts, true));
    pw.close;
    val out  = new FileOutputStream(fname+"options.ser")
    val output = new ObjectOutputStream(out);
    output.writeObject(opts);
    output.close;
    for (i <- 0 until modelmats.length) {
      val mat = modelmats(i);
      saveMat(fname+"modelmat%02d.lz4" format i, cpu(mat));
    }
    saveMetaData(fname);
    unguardOptions(guard);
  }

  def delete(fname:String) = {
    import java.io._
    val metadataname = new File(fname+"options.json");
    val serialname = new File(fname+"options.ser")
    val parentdir = metadataname.getParentFile();
    metadataname.delete();
    serialname.delete();
    for (i <- 0 until modelmats.length) {
      val matfile = new File(fname+"modelmat%02d.lz4" format i);
      matfile.delete();
    }
    deleteMetaData(fname);
    parentdir.delete();
  }

  def load(fname:String, useJson:Boolean=false) = {
	import java.io._
    import BIDMat.JSON
    if (modelmats != null && modelmats.length > 0) {
    	for (i <- 0 until modelmats.length) {
    		modelmats(i) = loadMat(fname+"modelmat%02d.lz4" format i);
    	}
    } else {
      var n = 0;
      var mlist = new ListBuffer[Mat]();
      while ((new File(fname+"modelmat%02d.lz4" format n)).exists) {
        mlist += loadMat(fname+"modelmat%02d.lz4" format n);
        n += 1;
      }
      setmodelmats(mlist.toArray);
    }
    if (useJson) { 
	  val fr = new BufferedReader(new FileReader(fname+"options.json"));
      val strbuf = new StringBuffer;
      var line:String = null;
      while ({line = fr.readLine(); line != null}) {
        strbuf.append(line).append("\n");
      }
      fr.close();
      val newopts = JSON.fromJSON(strbuf.toString).asInstanceOf[Model.Opts];
	  opts.copyFrom(newopts);
    } else { 
	  val in = new FileInputStream(fname+"options.ser");
	  val input = new ObjectInputStream(in);
	  val newopts = input.readObject.asInstanceOf[Model.Opts];
	  input.close;
	  opts.copyFrom(newopts);
    }
  }

  def bind(ds:DataSource):Unit = {
	  datasource = ds;
	  mats = datasource.next;
	  datasource.reset;
	  putBack = datasource.opts.putBack;
	  useGPU = opts.useGPU && Mat.hasCUDA > 0;
	  useDouble = opts.useDouble;
	  gmats = new Array[Mat](mats.length);
  }

  def bind(ds:DataSink):Unit = {
	  datasink = ds;
	  omats = datasink.omats;
	  ogmats = new Array[Mat](omats.length);
  }

  def init():Unit

  def dobatch(mats:Array[Mat], ipass:Int, here:Long)                                       // Calculate an update for the updater

  def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat              // Scores (log likelihoods)

  def logging(gmats:Array[Mat],ipass:Int, here:Long) = {
    if (opts.logFuncs!=null){
        val res = opts.logFuncs.map(f=>f(this,gmats));
        if (opts.logDataSink != null){
            opts.logDataSink.omats = res.flatten
            opts.logDataSink.setnmats(res.length)
            opts.logDataSink.put
        }
    }
  }

  def dobatchg(amats:Array[Mat], ipass:Int, here:Long) = {
    copyMats(amats, gmats);
    dobatch(gmats, ipass, here);
    logging(gmats, ipass, here);
  }

  def evalbatchg(amats:Array[Mat], ipass:Int, here:Long):FMat = {
    copyMats(amats, gmats)
    val v = evalbatch(gmats, ipass, here)
    if (omats != null) {
      for (i <- 0 until omats.length) {
        omats(i) = cpu(ogmats(i));
      }
    }
	v
  }

  def snapshot(len0:Int, avg:Boolean, matIdx:Int) = {
    if (sendmats == null) sendmats = new Array[Mat](modelmats.length)

    val mm = modelmats(matIdx)
    val mmnr = mm.nrows
    val mmnc = mm.ncols
    val avgOffset = if (avg) 1 else 0
    if (sendmats(matIdx) == null) sendmats(matIdx) = FMat(mmnr + avgOffset, mmnc)
    val len = math.min(len0, mmnc)
    val sendmat = sendmats(matIdx).view(mmnr + avgOffset, len)
    val smnr = sendmat.nrows
    val smnc = sendmat.ncols

    if (avg) sendmat(0, ?) = 1
    mm.synchronized {
      sendmat(avgOffset -> smnr, ?) = cpu(mm(?, 0 -> len))
    }
  }

  def addStep(len0:Int, avg:Boolean, matIdx:Int) = {
    if (avg && maxmats == null) maxmats = new Array[Mat](modelmats.length)

    val mm = modelmats(matIdx)
    val mmnr = mm.nrows
    val mmnc = mm.ncols
    val avgOffset = if (avg) 1 else 0
    if (avg && maxmats(matIdx) == null) maxmats(matIdx) = FMat(1, mmnc)
    val len = math.min(len0, mmnc);
    val recvmat = recvmats(matIdx).view(mmnr + avgOffset, len)
    val rmnr = recvmat.nrows
    val rmnc = recvmat.ncols
    val sendmat = sendmats(matIdx).view(mmnr + avgOffset, len)

    if (avg) {
      val maxmat = maxmats(matIdx).view(1, len)
      recvmat.rowslice(0, 1, maxmat)
      recvmat ~ recvmat / max(maxmat, 1f)
    }
    recvmat ~ recvmat - sendmat
    val smview = sendmat.view(mmnr, len)
    recvmat.rowslice(avgOffset, mmnr+avgOffset, smview)
    mm.synchronized {
      val mmview = mm.view(mmnr, len)

      var prenorm:Double = 0.0
      if (opts.trace > 2) prenorm = norm(mmview)

      mmview ~ mmview + smview

      if (opts.trace > 2) println("mat %d: pre-mm norm: %f, smview norm: %f, new-mm norm: %f" format (
        matIdx, prenorm, norm(smview), norm(mmview)))
    }
  }

  def elasticStep(len0:Int, avg:Boolean, alpha:Float, matIdx:Int) = {
    if (avg && maxmats == null) maxmats = new Array[Mat](modelmats.length)

    val mm = modelmats(matIdx)
    val mmnr = mm.nrows
    val mmnc = mm.ncols
    val avgOffset = if (avg) 1 else 0
    if (avg && maxmats(matIdx) == null) maxmats(matIdx) = FMat(1, mmnc)
    val len = math.min(len0, mmnc);
    val recvmat = recvmats(matIdx).view(mmnr + avgOffset, len)
    val rmnr = recvmat.nrows
    val rmnc = recvmat.ncols
    val sendmat = sendmats(matIdx).view(mmnr + avgOffset, len)

    if (avg) {
      val maxmat = maxmats(matIdx).view(1, len)
      recvmat.rowslice(0, 1, maxmat)
      recvmat ~ recvmat / max(maxmat, 1f)
    }
    recvmat ~ recvmat - sendmat
    val smview = sendmat.view(mmnr, len)
    recvmat.rowslice(avgOffset, mmnr+avgOffset, smview)
    mm.synchronized {
      val mmview = mm.view(mmnr, len)

      var prenorm:Double = 0.0
      if (opts.trace > 2) prenorm = norm(mmview)

      mmview ~ mmview + alpha*smview

      if (opts.trace > 2) println("mat %d: alpha %f, pre-mm norm: %f, smview norm: %f, new-mm norm: %f" format (
        matIdx, alpha, prenorm, norm(smview), norm(mmview)))
    }
  }

  def elasticStep(len0:Int, avg:Boolean, ee:Double, matIdx:Int):Unit =
    elasticStep(len0, avg, ee.toFloat, matIdx)

  def copyMats(from:Array[Mat], to:Array[Mat]) = {
    for (i <- 0 until from.length) {
      if (useGPU) {
        if (useDouble) {
         	to(i) = from(i) match {
        	case aa:GDMat => aa
        	case aa:GMat => GDMat(aa)
        	case aa:FMat => GDMat(aa)
        	case aa:IMat => GIMat(aa)
        	case aa:DMat => GDMat(aa)
        	case aa:BMat => GDMat(unsignedFloat(aa, true))
        	case aa:SMat => GSDMat(aa)
        	}
        } else {
        	to(i) = from(i) match {
        	case aa:GMat => aa
        	case aa:GDMat => GMat(aa)
        	case aa:FFilter => GFilter(aa)
        	case aa:FMat => GMat(aa)
        	case aa:DMat => GMat(aa)
        	case aa:IMat => GIMat(aa)
        	case aa:BMat => GMat(unsignedFloat(aa, true))
        	case aa:SMat => GSMat(aa)
        	}
        }
      } else {
      	if (useDouble) {
         	to(i) = from(i) match {
        	case aa:FMat => DMat(aa)
        	case aa:SMat => SDMat(aa)
        	case aa:DMat => DMat(aa);
        	case aa:SDMat => SDMat(aa);
        	case aa:IMat => IMat(aa);
        	case aa:LMat => LMat(aa);
        	case aa:BMat => DMat(unsignedFloat(aa,  true))
        	}
      	} else {
         	to(i) = from(i) match {
        	case aa:FMat => aa
        	case aa:SMat => aa
        	case aa:DMat => FMat(aa);
        	case aa:SDMat => SMat(aa);
        	case aa:IMat => IMat(aa);
        	case aa:LMat => LMat(aa);
        	case aa:BMat => unsignedFloat(aa,  true);
        	}
      	}
      }
    }
  }

  def updatePass(ipass:Int) = {}

  def convertMat(a:Mat):Mat = {
  	Model.convertMat(a, useGPU, opts.useDouble).asInstanceOf[Mat];
  }
  
  def convertIMat(a:IMat):IMat = {
    Model.convertMat(a, useGPU, opts.useDouble).asInstanceOf[IMat];
  }

  def combineModels(ipass:Int, model: Model):Model = this;
  def combineModels(model: Model):Model = combineModels(0, model);
  
  def clear = {}                        // Clear any model tmp matrices

  def wrapUp(ipass:Int):Unit = {}
}


object Model {
	trait Opts extends BIDMat.Opts{
	  var nzPerColumn:Int = 0;
	  var startBlock = 8000;
	  var useGPU = true;
	  var useDouble = false;
	  var doubleScore = false;
	  var doVariance = false;
	  var dim = 256;
	  var debug = 0;
	  var doAllReduce = false;
	  var logFuncs : Array[(Model,Array[Mat]) => Array[Mat]] = null;
	  var logDataSink : DataSink = null;
	  var naturalLambda = 0f
    var trace = 0
  }

	class Options extends Opts {}

  def convertMat(a:Mat, useGPU:Boolean, useDouble:Boolean):Mat = {
	   a match {
      case g:GMat => if (useGPU) {
      	if (useDouble) {
      		GDMat(g);
      	} else {
      	  g
      	}
      } else {
      	if (useDouble) {
      	  DMat(FMat(g));
      	} else {
      		FMat(g);
      	}
      }
      case g:GDMat => if (useGPU) {
      	if (useDouble) {
      		g;
      	} else {
      	  GMat(g)
      	}
      } else {
      	if (useDouble) {
      	  DMat(g);
      	} else {
      		FMat(g);
      	}
      }
      case g:GSMat => if (useGPU) {
      	if (useDouble) {
      		GSDMat(g);
      	} else {
      	  g;
      	}
      } else {
      	if (useDouble) {
      	  SDMat(SMat(g));
      	} else {
      		SMat(g);
      	}
      }
      case f:FFilter =>
      if (useGPU) {
      	GFilter(f);
      } else {
      	f;
      }
      case f:FMat =>
      if (useGPU) {
      	if (useDouble) {
      		GDMat(f);
      	} else {
      		GMat(f);
      	}
      } else {
      	if (useDouble) {
      		DMat(f);
      	} else {
      		f
      	}
      }
      case i:IMat =>
      if (useGPU) {
        GIMat(i);
      } else {
        i;
      }
      case s:SMat =>
      if (useGPU) {
        GSMat(s);
      } else {
        s;
      }
      case tt:TMat => new TMat(tt.nrows, tt.ncols, tt.y, tt.x, tt.tiles.map(convertMat(_, useGPU, useDouble).asInstanceOf[Mat]));
    }
  }
}
