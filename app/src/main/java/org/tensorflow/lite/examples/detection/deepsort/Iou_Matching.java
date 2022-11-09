package org.tensorflow.lite.examples.detection.deepsort;

import java.util.Arrays;

class Iou_Matching {

    private float[] iou(float[] bbox, float[][] candidates){
        float[] bbox_tl = new float[2];
        bbox_tl[0] = bbox[0];
        bbox_tl[1] = bbox[1];

        float[] bbox_br = new float[2];
        bbox_br[0] = bbox[0] + bbox[2];
        bbox_br[1] = bbox[1] + bbox[3];

        float[][] candidates_tl = new float[candidates.length][2];
        float[][] candidates_br = new float[candidates.length][2];
        for(int i=0;i<candidates_tl.length;i++){
            candidates_tl[i][0] = candidates[i][0];
            candidates_tl[i][1] = candidates[i][1];

            candidates_br[i][0] = candidates[i][0] + candidates[i][2];
            candidates_br[i][1] = candidates[i][1] + candidates[i][3];
        }

        float[][] tl = new float[candidates_tl.length][candidates_tl[0].length];
        for(int i=0;i<candidates_tl.length;i++){
            tl[i][0] = Math.max(bbox_tl[0],candidates_tl[i][0]);
            tl[i][1] = Math.max(bbox_tl[1],candidates_tl[i][1]);
        }

        float[][] br = new float[candidates_br.length][candidates_br[0].length];
        for(int i=0;i<candidates_br.length;i++){
            br[i][0] = Math.min(bbox_br[0],candidates_br[i][0]);
            br[i][1] = Math.min(bbox_br[1],candidates_br[i][1]);
        }

        float[][] wh = new float[candidates_tl.length][candidates_tl[0].length];
        for(int i=0;i<candidates_br.length;i++) {
            for (int j=0;j<candidates_br[0].length;j++) {
                wh[i][j] = Math.max(0f,br[i][j]-tl[i][j]);
            }
        }
        float[] area_intersection = new float[wh.length];
        for(int j=0;j<area_intersection.length;j++) {
            area_intersection[j] = wh[j][0]*wh[j][1];
        }

        float area_bbox = bbox[2] * bbox[3];

        float[] area_candidates = new float[candidates.length];
        for(int j=0;j<candidates.length;j++){
            area_candidates[j] = candidates[j][2] * candidates[j][3];
        }

        for(int j=0;j<area_intersection.length;j++) area_intersection[j] /= (area_bbox + area_candidates[j] - area_intersection[j]);
        return area_intersection;
    }

    float[][] iou_cost(Track[] tracks, Detection[] detections, int[] track_indices, int[] detection_indices){
        if(track_indices.length == 0){
            track_indices = new int[tracks.length];
            for(int i=0;i<tracks.length;i++)
                track_indices[i] = i;
        }
        if(detection_indices.length == 0){
            detection_indices = new int[detections.length];
            for(int i=0;i<detections.length;i++)
                detection_indices[i] = i;
        }

        float[][] cost_matrix = new float[track_indices.length][detection_indices.length];
        for(int i=0;i<track_indices.length;i++){
            int track_idx = track_indices[i];
            if(tracks[track_idx].time_since_update > 1){
                Arrays.fill(cost_matrix[i],(float) 1e+5);
                continue;
            }
            float[] bbox = tracks[track_idx].to_tlwh();

            float[][] candidates = new float[detection_indices.length][bbox.length];

            for(int j=0;j<detection_indices.length ;j++){
                    candidates[j] = detections[detection_indices[j]].tlwh;
            }
            for(int k=0;k<cost_matrix[0].length;k++){
                cost_matrix[i][k] = 1.0f - iou(bbox,candidates)[k];
            }
        }
        return cost_matrix;
    }
}
