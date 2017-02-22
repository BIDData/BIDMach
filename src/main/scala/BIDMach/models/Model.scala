package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,FND,GMat,GDMat,GIMat,GSMat,GSDMat,GND,HMat,IMat,JSON,LMat,ND,SMat,SDMat,TMat}
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

  var parent_model:Model = null;

  def modelmats:Array[Mat] = {
    if (_modelmats != null) {
      _modelmats
    } else if (parent_model != null) {
      parent_model._modelmats
    } else {
      null
    }
  }

  def setmodelmats(a:Array[Mat]) = {
    _modelmats = a;
  }

  var updatemats:Array[Mat] = null;

  // For Allreduce: the local indices
  var indexmat:Mat = null;

  // For Allreduce: cached local matrices:
  var sendmat:Mat = null;

  var recvmat:Mat = null;

  var mats:Array[Mat] = null;

  var gmats:Array[Mat] = null;

  var omats:Array[Mat] = null;

  var ogmats:Array[Mat] = null;

  var useGPU = false;

  var useDouble = false;

  var putBack = -1;

  var refresh = true;

  var runtimes:FMat = null;

  def mergeModelFn(models:Array[Model], mm:Array[Mat], um:Array[Mat], istep:Long):Unit = {
    val mlen = models(0).modelmats.length;
    val thisGPU = getGPU;
    for (j <- 0 until mlen) {
      mm(j).clear
      for (i <- 0 until models.length) {
        if (useGPU && i < Mat.hasCUDA) setGPU(i);
      	um(j) <-- models(i).modelmats(j);
      	mm(j) ~ mm(j) + um(j);
      }
      mm(j) ~ mm(j) * (1f/models.length);
      for (i <- 0 until models.length) {
      	models(i).modelmats(j) <-- mm(j);
    	}
    }
    setGPU(thisGPU);
  }

  def mergeModelPassFn(models:Array[Model], mm:Array[Mat], um:Array[Mat], ipass:Int) {}

  def copyTo(mod:Model) = {
    mod.datasource = datasource;
    mod._modelmats = modelmats;
    mod.updatemats = updatemats;
    mod.mats = mats;
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

  def loadMetaData(fname:String) = {}

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
    pw.print(JSON.toJSON(opts, true));
    pw.close;
    val out  = new FileOutputStream(fname+"options.ser")
    val output = new ObjectOutputStream(out);
    output.writeObject(opts);
    output.close;
    for (i <- 0 until modelmats.length) {
      val mat = modelmats(i);
      val f = new File(fname+"modelmat%02d.lz4" format i);
      saveMat(fname+"modelmat%02d.lz4" format i, cpu(mat));
    }
    saveMetaData(fname);
  }

  def load(fname:String) = {
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
	  if (new File(fname+"options.ser").exists) {
	  	val in = new FileInputStream(fname+"options.ser");
	  	val input = new ObjectInputStream(in);
	  	val newopts = input.readObject.asInstanceOf[Model.Opts];
	  	input.close;
	  	/*    val fr = new BufferedReader(new FileReader(fname+"options.json"));
    val strbuf = new StringBuffer;
    var line:String = null;
    while ({line = fr.readLine(); line != null}) {
      strbuf.append(line).append("\n");
    }
    val newopts = JSON.fromJSON(strbuf.toString).asInstanceOf[Model.Opts]; */
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

  def snapshot(len:Int, avg:Boolean) = {
  	val len0 = math.min(len, modelmats(0).ncols);
  	modelmats(0).synchronized {
  		sendmat = cpu(modelmats(0).colslice(0, len0));
  	}
  	if (avg) {
  		sendmat = ones(1, len0) on sendmat;
  	}
  }

  def addStep(len:Int, avg:Boolean) = {
  	val len0 = math.min(len, modelmats(0).ncols);
  	if (avg) recvmat = recvmat / max(recvmat(0,?), 1f);
  	recvmat = recvmat - sendmat;
  	val nr = modelmats(0).nrows;
  	modelmats(0).synchronized {
  		val head = modelmats(0).view(nr, len0);
  		val chead = sendmat.view(nr, len0);
  		chead <-- head;
  		chead ~ chead + (if (avg) recvmat(1 -> (nr+1), ?) else recvmat);
  		head <-- chead;
  	}
  }

  def elasticStep(len:Int, avg:Boolean, ee:Float) = {
  	val len0 = math.min(len, modelmats(0).ncols);
  	if (avg) recvmat = recvmat / max(recvmat(0,?), 1f);
  	recvmat = recvmat - sendmat;
  	val nr = modelmats(0).nrows;
  	modelmats(0).synchronized {
  		val head = modelmats(0).view(nr, len0);
  		val chead = sendmat.view(nr, len0);
  		chead <-- head;
  		chead ~ chead * (1 - ee) + (if (avg) recvmat(1 -> (nr+1), ?) else recvmat) * ee;
  		head <-- chead;
  	}
  }

  def copyMats(from:Array[Mat], to:Array[Mat]) = {
    for (i <- 0 until from.length) {
      if (useGPU) {
        if (useDouble) {
         	to(i) = from(i) match {
        	case aa:FMat => GDMat(aa)
        	case aa:IMat => GIMat(aa)
        	case aa:DMat => GDMat(aa)
        	case aa:SMat => GSDMat(aa)
        	case aa:GDMat => aa
        	case aa:GMat => GDMat(aa)
        	}
        } else {
        	to(i) = from(i) match {
        	case aa:FMat => GMat(aa)
        	case aa:DMat => GMat(aa)
        	case aa:IMat => GIMat(aa)
        	case aa:SMat => GSMat(aa)
        	case aa:GMat => aa
        	case aa:GDMat => GMat(aa)
        	}
        }
      } else {
      	if (useDouble) {
         	to(i) = from(i) match {
        	case aa:FMat => DMat(aa)
        	case aa:SMat => SDMat(aa)
        	case aa:DMat => aa;
        	case aa:SDMat => aa;
        	}
      	} else {
         	to(i) = from(i) match {
        	case aa:IMat => aa
		case aa:FMat => aa
        	case aa:SMat => aa
        	case aa:DMat => FMat(aa);
        	case aa:SDMat => SMat(aa);
        	}
      	}
      }
    }
  }

  def updatePass(ipass:Int) = {}

  def convertMat(a:Mat):Mat = {
  	Model.convertMat(a, useGPU, opts.useDouble).asInstanceOf[Mat];
  }

  def convertMat(a:ND):ND = {
  	Model.convertMat(a, useGPU, opts.useDouble);
  }

  def combineModels(ipass:Int, model: Model):Model = this;
  def combineModels(model: Model):Model = combineModels(0, model);

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
  }

	class Options extends Opts {}

  def convertMat(a:ND, useGPU:Boolean, useDouble:Boolean):ND = {
	   a match {
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
      case g:FND => if (useGPU) {
      	GND(g);
      } else {
      	g
      }
      case g:GND => if (useGPU) {
      	g
      } else {
      	FND(g)
      }
      case tt:TMat => new TMat(tt.nrows, tt.ncols, tt.y, tt.x, tt.tiles.map(convertMat(_, useGPU, useDouble).asInstanceOf[Mat]));
    }
  }
}
