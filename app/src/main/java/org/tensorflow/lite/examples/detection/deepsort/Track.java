package org.tensorflow.lite.examples.detection.deepsort;

import android.util.Pair;
import java.util.ArrayList;
import java.util.Arrays;

class Track {
     static class Trackstate{
         static int Tentative = 1;
         static int Confirmed = 2;
         static int Deleted = 3;
    }
    float[] mean;
    float[][] covariance;
    int track_id;
    private int hits = 1;
    int time_since_update = 0;
    private int state = Trackstate.Tentative;

    private final int n_init;
    private final int max_age;

     ArrayList<float[]> features=new ArrayList<>();
     Track(float[] mean1,float[][] covariance1,int track_id1,int n_init1,int max_age1,float[] feature1){
        mean=mean1;
        covariance=covariance1;
        track_id=track_id1;
        n_init=n_init1;
        max_age=max_age1;
        if(feature1!=null)
            features.add(feature1);
    }
    float[] to_tlwh(){
        float[] ret = Arrays.copyOfRange(mean.clone(),0,4);
        ret[2] *= ret[3];
        ret[0] -= (ret[2] / 2);
        ret[1] -= (ret[3] / 2);
        return ret;
    }
     float[] to_tlbr(){
        float[] ret = to_tlwh();
        ret[2] = ret[0] + ret[2];
        ret[3] = ret[1] + ret[3];
        return ret;
    }
     void predict(Kalman_filter kf){
        Pair<float[],float[][]> sample = kf.predict(mean,covariance);
        mean = sample.first;
        covariance = sample.second;
         time_since_update+=1;

    }
     void update(Kalman_filter kf, Detection detection){
        Pair<float[],float[][]> sample = kf.update(mean,covariance,detection.to_xyah());
        mean = sample.first;
        covariance = sample.second;
        features.add(detection.feature);
        hits+=1;
        time_since_update = 0;
        if(state==Trackstate.Tentative && hits>= n_init){
            state=Trackstate.Confirmed;
        }
    }
     void mark_missed(){
        if (state == Trackstate.Tentative){
            state=Trackstate.Deleted;
        }
        else if(time_since_update > max_age){
            state=Trackstate.Deleted;
        }
    }
     boolean is_confirmed(){
        return (state == Trackstate.Confirmed);
    }
     boolean is_deleted(){
        return (state == Trackstate.Deleted);
    }
}
