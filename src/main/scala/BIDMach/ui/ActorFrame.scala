package BIDMach.ui

import BIDMat.{Mat,FMat,Image,MatFunctions}
import java.util.concurrent.Future;
import BIDMach.Learner
import javax.swing._;
import com.mxgraph.swing.mxGraphComponent;
import com.mxgraph.view.mxGraph;
import scala.collection.immutable.HashMap
import akka.actor.{Actor,Props,ActorSystem,ActorRef};
import ptolemy.plot._
import BIDMat.{Plotting,MyPlot}

class ActorFrame(val name:String = "display") extends Actor{    
    //def _liveplot2(fn:()=>Mat, interval:Float)(xlog:Boolean=false, ylog:Boolean=false, isconnected:Boolean=true, bars:Boolean=false, marks:Int = 0) = {
    var p:MyPlot = new MyPlot;
    val isconnected = true
    //p.setXLog(xlog);
    //p.setYLog(ylog);
    //p.setBars(bars);
    //p.setConnected(isconnected);
    //p.setMarksStyle(marksmat(marks));
    val dataset = 0;
    p.frame = new PlotFrame("Figure "+ActorFrame.ifigure, p);
  	p.frame.setVisible(true);
  	p.done = false;
    def receive() =  {
        case mat:Mat => {
            p.clear(0);
      		Plotting._replot(Array(mat), p, dataset, isconnected);
        }
    }      
    ActorFrame.ifigure += 1
}

object ActorFrame{
    var ifigure = 0
}

class SimpleActorFrame(val name:String = "display") extends Actor{
    val frame = new JFrame(name)
    val panel = new JPanel()
    val title = new JLabel("here")
    panel.add(title)
    frame.setContentPane(panel)
    frame.setSize(400, 300);
    frame.setVisible(true);
    def receive() =  {
        case a:Any => title.setText(a.toString)
    }
}