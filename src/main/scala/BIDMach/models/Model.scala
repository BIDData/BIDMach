package BIDMach.models
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,GMat,GDMat,GIMat,GSMat,GSDMat,HMat,IMat,JSON,LMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import scala.collection.mutable.ListBuffer

/**
 * Abstract class with shared code for all models
 */
abstract class Model(val opts:Model.Opts = new Model.Options) extends Serializable {
  
  var datasource:DataSource = null
  
  var _modelmats:Array[Mat] = null
  
  var parent_model:Model = null
  
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
  
  var updatemats:Array[Mat] = null
  
  var mats:Array[Mat] = null
  
  var gmats:Array[Mat] = null
  
  var useGPU = false
  
  var useDouble = false
  
  var putBack = -1
  
  var refresh = true
  
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
  }
  
  def copyFrom(mod:Model) = {
    setmodelmats(new Array[Mat](mod.modelmats.length));
    for (i <- 0 until modelmats.length) {
      modelmats(i) = mod.modelmats(i);
    }
  }
  
  def saveMetaData(fname:String) = {}
  
  def loadMetaData(fname:String) = {}
  
  def save(fname:String) = {
    import java.io._
    for (i <- 0 until modelmats.length) {
      val mat = modelmats(i);
      val f = new File(fname+"modelmat%02d.lz4" format i);
      f.getParentFile().mkdirs();
      saveMat(fname+"modelmat%02d.lz4" format i, cpu(mat));
    }
    val pw = new PrintWriter(new File(fname+"options.json"));
    pw.print(JSON.toJSON(opts, true));
    pw.close;
    val out  = new FileOutputStream(fname+"options.ser")
    val output = new ObjectOutputStream(out);
    output.writeObject(opts);
    output.close;
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
      while ((new java.io.File(fname+"modelmat%02d.lz4" format n)).exists) {
        mlist += loadMat(fname+"modelmat%02d.lz4" format n);
        n += 1;
      }
      setmodelmats(mlist.toArray);
    }
    val in = new FileInputStream(fname+"options.ser");
    val input = new ObjectInputStream(in);
    val newopts = input.readObject.asInstanceOf[Model.Opts];
    input.close;
    opts.copyFrom(newopts)
  }
  
  def bind(ds:DataSource):Unit = {
	  datasource = ds;
	  mats = datasource.next;
	  datasource.reset;
	  putBack = datasource.opts.putBack;
	  useGPU = opts.useGPU && Mat.hasCUDA > 0;
	  useDouble = opts.useDouble;
	  if (useGPU || useDouble) {
	  	gmats = new Array[Mat](mats.length);
	  } else {
	  	gmats = mats;
	  }
  }
  
  def init():Unit
  
  def dobatch(mats:Array[Mat], ipass:Int, here:Long)                                       // Calculate an update for the updater
  
  def evalbatch(mats:Array[Mat], ipass:Int, here:Long):FMat                                // Scores (log likelihoods)
  
  def dobatchg(amats:Array[Mat], ipass:Int, here:Long) = {
    if (useGPU) copyMats(amats, gmats)            		
    dobatch(gmats, ipass, here)
    if ((useGPU || useDouble) && putBack >= 0) {
    	for (i <- 1 to putBack) {
    		amats(i) <-- gmats(i)
    	}
    }
  }
  
  def evalbatchg(amats:Array[Mat], ipass:Int, here:Long):FMat = {
    if (useGPU) copyMats(amats, gmats)
    val v = evalbatch(gmats, ipass, here)
    if ((useGPU || useDouble) && putBack >= 0) {
      for (i <- 1 to putBack) {
        amats(i) <-- gmats(i)
      }
    }
	v
  }

  def copyMats(from:Array[Mat], to:Array[Mat]) = {
    for (i <- 0 until from.length) {
      if (useGPU) {
        if (useDouble) {
         	to(i) = from(i) match {
        	case aa:FMat => GDMat(aa)
        	case aa:IMat => GIMat(aa)
        	case aa:SMat => GSDMat(aa)
        	case aa:GDMat => aa
        	case aa:GMat => GDMat(aa)
        	}         
        } else {
        	to(i) = from(i) match {
        	case aa:FMat => GMat(aa)
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
        	}
      	}
      }
    }
  }
  
  def updatePass(ipass:Int) = {}
  
  def convertMat(a:Mat):Mat = {
  	Model.convertMat(a, useGPU, opts.useDouble);
  }
}


object Model {
	trait Opts extends BIDMat.Opts{
	  var nzPerColumn:Int = 0
	  var startBlock = 8000
	  var useGPU = true
	  var useDouble = false
	  var doubleScore = false
	  var dim = 256
	  var debug = 0;
  }
	
	class Options extends Opts {} 
  
  def convertMat(a:Mat, useGPU:Boolean, useDouble:Boolean):Mat = {	
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
    }
  }
}
