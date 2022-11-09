package org.tensorflow.lite.examples.detection.deepsort;

 public class Detection {

     float[] tlwh;
     float confidence;
     float[] feature;

     Detection(float[] tlwh,float confidence, float[] feature){
        this.tlwh = tlwh;
        this.confidence = confidence;
        this.feature = feature;
    }

     float[] to_tlbr(){
        float[] ret = tlwh.clone();
        ret[2] += ret[0];
        ret[3] += ret[1];

        return ret;
    }
    
     float[] to_xyah(){
        float[] ret = tlwh.clone();
        ret[0] += (ret[2] / 2);
        ret[1] += (ret[3] / 2);
        ret[2] /= ret[3];

        return ret;
    }
}
