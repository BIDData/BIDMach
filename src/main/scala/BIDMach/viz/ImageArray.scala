package BIDMach.viz


import BIDMat.{BMat,Mat,SBMat,CMat,DMat,FMat,FFilter,IMat,HMat,GDMat,GFilter,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat,TMat,Image}
import BIDMat.MatFunctions._
import BIDMat.MatFunctions
import BIDMat.SciFunctions._
import java.util.concurrent.Future;
import BIDMach.Learner
import BIDMach.networks.Net
import BIDMach.networks.layers._
import javax.swing._;
import javax.swing.event._
import scala.collection.mutable.HashMap
import java.awt.event._;
import java.awt._;    
import java.awt.image.BufferedImage
import ptolemy.plot._
import BIDMat.{Plotting,MyPlot,Image}
import jupyter.api.Publish



class ImageArray(name: String = "plot") {
    val id = {ImageArray.tot+=1; ImageArray.tot}
    //var server = {if (ImageArray.server == null) ImageArray.server = new WebServer(); ImageArray.server}
    var img: Image = null;
    val controlPanel: JPanel = new JPanel();
    val frame: JFrame = new JFrame;

    implicit val publish = new NonNotebook;
    
    def init(data: FMat) {
        img = Image(data);
        img.show;
        //frame = img.frame;
        frame.getContentPane().add(controlPanel)
        frame.getContentPane().setLayout(new BoxLayout(frame.getContentPane(),BoxLayout.PAGE_AXIS))        
        //controlPanel.setLayout(new FlowLayout)//new BoxLayout(controlPanel,BoxLayout.PAGE_AXIS));
        controlPanel.setLayout(new BoxLayout(controlPanel,BoxLayout.PAGE_AXIS));
        frame.pack;
        frame.setVisible(true);
    }
        
    def plot_image(data_ :Mat, tensorFormat: Int = 1) {
        val data = utils.packImages(data_, tensorFormat)
        if (img == null)
            init(data);
        img.redraw(data);    
    }
    
    def plot_web(data:IMat) {
        //server.send(scala.util.parsing.json.JSONObject())
    }    
    
    def add_slider(name:String,callback:Int=>Float,initV:Int = 50, precision:Int = 2) {
        val p = new JPanel();
        val sliderLabel = new JLabel(name, SwingConstants.CENTER);
        val rangeLabel = new JLabel(("%."+precision+"f") format callback(initV), SwingConstants.CENTER);
        val slider = new JSlider(SwingConstants.HORIZONTAL,0,100,initV);
        slider.addChangeListener(new ChangeListener{
                                    override def stateChanged(e:ChangeEvent){
                                        val source = e.getSource().asInstanceOf[JSlider];
                                        val v = callback(source.getValue());
                                        rangeLabel.setText(("%."+precision+"f") format v)
                                    }})
        p.add(sliderLabel);
        p.add(slider);
        p.add(rangeLabel);
        controlPanel.add(p);
        frame.pack
    }
    
    def add_combobox(items: Array[String],callback:(Int, String)=>Unit) = {
        //val p = new JPanel();
        //val label = new JLabel("test", SwingConstants.CENTER);
        val box = new JComboBox(items);
        box.addItemListener(new ItemListener{
            override def itemStateChanged(e:ItemEvent){ 
                if (e.getStateChange() == ItemEvent.SELECTED) {
                    callback(box.getSelectedIndex,e.getItem.toString)                        
                }
            }
        });
        ///p.add(label);
        //p.add(box);           
        controlPanel.add(box);
        //if (frame != null)frame.pack;
        frame.pack
        box            
    }
    
}

object ImageArray{
    var tot = 0
}
