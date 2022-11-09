package org.tensorflow.lite.examples.detection.deepsort;

import android.annotation.SuppressLint;
import android.util.Log;
import android.util.Pair;
import java.util.*;

public class Deepsort_RBC {
    private final Tracker tracker;

    public Deepsort_RBC() {
        NN_Matching.NearestNeighborDistanceMetrix metric = new NN_Matching().new NearestNeighborDistanceMetrix("cosine", 0.5f, 100);
        tracker = new Tracker(metric,0.8f,45,10);
    }
    public Pair<Tracker, List<Detection>> run_deep_sort(List<float[]> feature, List<Float> out_scores, List<float[]> bboxes){
        if(bboxes.size() == 0){
            tracker.predict();
            System.out.println("No Detections");
            return new Pair<>(tracker,null);
        }
        List<Detection> dets = new ArrayList<>();
        for(int i=0,j=out_scores.size()-1;i<out_scores.size();i++,j--)
            dets.add(new Detection(bboxes.get(i),out_scores.get(i),feature.get(i)));

        //todo nms for better bounding box mapping

        tracker.predict();
        tracker.update(dets);

        return new Pair<>(tracker,dets);
    }

    private float[][] np_sub(float[][] a, float[][] b){
        float[][] c = new float[a.length][a[0].length];
        if(b.length==1){
            for(int i=0;i<a.length;i++) {
                for (int j = 0; j < a[0].length; j++)
                    c[i][j] = a[i][j] - b[0][j];
            }
        }
        else{
            for(int i=0;i<a.length;i++) {
                for (int j = 0; j < a[0].length; j++)
                    c[i][j] = a[i][j] - b[i][j];
            }
        }

        return c;
    }
    private float[] np_sub(float[] a,float[] b){
        float[] c = new float[a.length];
        for(int i=0;i<a.length;i++)
            c[i]=a[i]-b[i];
        return c;
    }
    private float np_mean(float[] a){
        float sum = 0;
        for (float v : a) sum += v;
        return sum/a.length;
    }
    private float[][] list_to_array(ArrayList<float[]> sample){
        float[][] outarr = new float[sample.size()][sample.get(0).length];
        for(int i=0;i<sample.size();i++) outarr[i] = sample.get(i);
        return outarr;
    }
    private float norm(float[][] x){
        float sum = 0f;
        for (float[] floats : x) {
            for (int j = 0; j < x[0].length; j++)
                sum += Math.pow(floats[j], 2);
        }
        return (float) Math.sqrt(sum);
    }
    private static float miou(float[] bbox1,float[] bbox2){
        float x_left = Math.max(bbox1[0], bbox2[0]);
        float y_top = Math.max(bbox1[1], bbox2[1]);
        float x_right = Math.min(bbox1[2], bbox2[2]);
        float y_bottom = Math.min(bbox1[3], bbox2[3]);
        if(x_right<x_left || y_bottom<y_top)
            return 0.0f;
        float intersection_area = (x_right-x_left)*(y_bottom-y_top);
        float bb1_area = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1]);
        float bb2_area = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1]);
        float iou = intersection_area/(bb1_area+bb2_area-intersection_area);
        return iou;
    }

    @SuppressLint("NewApi")
    public Object[] remap(List<Track> tracks, Map<Integer, StateInfo> my_dict, Map<Integer, Integer> disjoint){
        List<Integer> track_ids = new ArrayList<>();
        for(Track track : tracks){
            if ((!track.is_confirmed()) || track.time_since_update > 1)  // to check if track is confirmed
                continue;
            track_ids.add(track.track_id);
            int key = -1;
            float max_iou =0f;
            // compare track id with my_dict and find disjoints
            for(Map.Entry<Integer,StateInfo> entry : my_dict.entrySet()){
                int k1 = entry.getKey();
                StateInfo v1 = entry.getValue();
                boolean flag = false;
                float cur_iou = miou(track.to_tlbr(), v1.isStationary_flag() ? v1.getSbbox() : v1.getMbbox());
                if(v1.isStationary_flag()){
                    if(cur_iou>0.7 && norm(np_sub(list_to_array(track.features),list_to_array(v1.getSfeature())))<20)  //if object is stationary, but id has changed
                        flag = true;
                }
                else if(cur_iou>0.4)    //if object is not stationary and id are swapped
                    flag = true;
                if(flag && (cur_iou>max_iou)){
                    key = k1;
                    max_iou = cur_iou;
                }
            }
            if(key!=-1){
                if(track.track_id!=key)
                    disjoint.put(track.track_id,key);
            }
        }

        List<Integer> keys = new ArrayList<>();
        List<Integer> values = new ArrayList<>();
        // to remap the dictionary entry to current id from tracker
        for(Map.Entry<Integer,Integer> entry : disjoint.entrySet()){   //to remove old ids from occlusion list
            int k = entry.getKey();
            int v = entry.getValue();
            for(Map.Entry<Integer,StateInfo> entry1 : my_dict.entrySet()) {
                if (entry1.getValue().getOcc_list() !=null && entry1.getValue().getOcc_list().contains(v)) //id is remapped to another new id
                    entry1.getValue().getOcc_list().remove(Integer.valueOf(v));
            }
            if((my_dict.containsKey(v))&&(!track_ids.contains(v))){
                if(my_dict.containsKey(k))
                    my_dict.replace(k, my_dict.get(v));
                else
                    my_dict.put(k,my_dict.get(v));
                Objects.requireNonNull(my_dict.get(k)).setId(k);
                my_dict.remove(v);
            }
            keys.add(k);
            values.add(v);
        }
        Iterator it = disjoint.entrySet().iterator();
        while(it.hasNext()) {            //to remove deleted ids from disjoint set
            Map.Entry item = (Map.Entry) it.next();
            int k = (int) item.getKey();
            int v = (int) item.getValue();
            if(values.contains(k)) {
                disjoint.replace(keys.get(values.indexOf(k)),v);
                it.remove();
            }
        }
        Object[] arrayObjects = new Object[2];
        arrayObjects[0] = track_ids;
        arrayObjects[1] = disjoint;
        return arrayObjects;
    }
    public Map<Integer,StateInfo> delete_removedbox(Map<Integer, StateInfo> my_dict, List<Integer> track_ids, Map<Integer, Integer> disjoint) {
        List<Integer> rm_list = new ArrayList<>();
        for(int key:my_dict.keySet()){
            if(disjoint.containsKey(key))   //assign new id in the state info
                my_dict.get(key).setId(disjoint.get(key));
            if((!track_ids.contains(key))&&(!my_dict.get(key).isStationary_flag())){
                my_dict.get(key).setMissing_ts(my_dict.get(key).getMissing_ts()+1);
                if(my_dict.get(key).getMissing_ts() == 20)
                    rm_list.add(key);
            }
            else if(track_ids.contains(key))      //object in system
                my_dict.get(key).setMissing_ts(0);
        }
        for(int key:rm_list){
            disjoint.remove(key);
            if(my_dict.get(key).isRandom()){
                //Log.i("Protocol Message","CV ROI-1 Neither");
            }
            my_dict.remove(key);
        }
        return my_dict;
    }
    private Map<Integer,StateInfo> check_occlusion(Map<Integer,StateInfo>  my_dict,Track track){
        for(int key:my_dict.keySet()){
            float[] bbox = my_dict.get(track.track_id).isStationary_flag() ? my_dict.get(track.track_id).getSbbox() : my_dict.get(track.track_id).getMbbox();
            if((my_dict.get(key).isStationary_flag())&&(key!=track.track_id)&&(miou(bbox, my_dict.get(key).getSbbox())>0f)){    //occlusion
                if(my_dict.get(key).getOcc_list() ==null)
                    my_dict.get(key).setOcc_list(new ArrayList<>());
                my_dict.get(key).getOcc_list().add(track.track_id);
                my_dict.get(key).setTimestamp(0);
            }
            else if(my_dict.get(key).getOcc_list() !=null && my_dict.get(key).getOcc_list().contains(track.track_id))
                my_dict.get(key).getOcc_list().remove(new Integer(track.track_id)); // .remove(track.track_id);
        }
        return my_dict;
    }
    @SuppressLint("NewApi")
    public Map<Integer,StateInfo> update_dict(List<Track> tracks, Map<Integer, StateInfo> my_dict, long ts, int ROI_height1, int ROI_height2) {
        for(Track track:tracks){
            if((!track.is_confirmed())||track.time_since_update>1)
                continue;
            if(my_dict.containsKey(track.track_id)){      //existing box
                StateInfo box = my_dict.get(track.track_id);
                float[] prev_box = box.isStationary_flag() ? box.getSbbox() : box.getMbbox();
                box.setUpdate_frame(ts);
                if(!(box.isStationary_flag() && box.getOcc_list() ==null)){
                    box.setVelocity(np_sub(track.to_tlbr(), box.getMbbox()));
                    box.setMbbox(track.to_tlbr());
                    box.setMfeature(track.features);
                    if(np_mean(box.getVelocity())<2.0f)
                        box.setTimestamp(box.getTimestamp()+1);
                    else
                        box.setTimestamp(0);
                    if(box.getTimestamp() ==50){
                        box.setStationary_flag(true);
                        box.setSbbox(box.getMbbox());
                        box.setSfeature(box.getMfeature());
                    }
                }
                else{
                    box.setVelocity(np_sub(track.to_tlbr(), box.getSbbox()));
                    box.setMbbox(track.to_tlbr());
                    box.setMfeature(track.features);
                }
                if((np_mean(box.getVelocity())>2)||(box.getVelocity()[0]<-5.0f && box.getVelocity()[2]<-5.0f)){
                    box.setStationary_flag(false);
                    box.setTimestamp(0);
                    box.setSbbox(null);
                    box.setSfeature(null);
                }
                float[] bbox = track.to_tlbr();
                if((((bbox[0]+bbox[2])/2)>ROI_height1) && (((prev_box[0]+prev_box[2])/2)<ROI_height1))
                    box.setEntry1(true);
                if((((bbox[0]+bbox[2])/2)>ROI_height2) && (((prev_box[0]+prev_box[2])/2)<ROI_height2))
                    box.setEntry2(true);
                if((((bbox[0]+bbox[2])/2)<ROI_height1) && (((prev_box[0]+prev_box[2])/2)>ROI_height1))
                    box.setExit1(true);
                if((((bbox[0]+bbox[2])/2)<ROI_height2) && (((prev_box[0]+prev_box[2])/2)>ROI_height2))
                    box.setExit2(true);

                my_dict.replace(track.track_id,box);
            }
            else{     //new box enters the system
                StateInfo box = new StateInfo(track.track_id,track.to_tlbr(),null,track.features,null,0,0,null,false,null,ts);
                int ROI = 1;
                float[] bbox = track.to_tlbr();
                if(((bbox[0]+bbox[2])/2)<ROI_height1){
                    ROI = 1 ;
                }else if(((bbox[0]+bbox[2])/2)>ROI_height1 &&((bbox[0]+bbox[2])/2)<ROI_height2){
                    ROI = 2 ;
                }else if(((bbox[0]+bbox[2])/2)>ROI_height2){
                    ROI = 3;
                }
                box.setROI(ROI);
                my_dict.put(track.track_id,box);
            }
            my_dict = check_occlusion(my_dict,track);   //check for occlusion
        }
        return my_dict;
    }




}
