package org.tensorflow.lite.examples.detection.deepsort;

import android.util.Pair;
import java.util.*;

class LinearAssignment {

    private final Iou_Matching iou_matching;

    LinearAssignment(){
        iou_matching = new Iou_Matching();
    }

    Object[] min_cost_matching(Tracker.Gated_metric metric, float max_distance, List<Track> tracks, List<Detection> detections, List<Integer> track_indices, List<Integer> detection_indices) {
        List<Pair<Integer,Integer>> matches = new ArrayList<>();
        List<Integer>  unmatched_tracks = new ArrayList<>();
        List<Integer> unmatched_detections = new ArrayList<>();
        Object[] arrayObjects = new Object[3];

        if(track_indices == null){
            track_indices = new ArrayList<>();
            for(int i=0;i<tracks.size();i++)
                track_indices.add(i);
        }
        if(detection_indices == null){
            detection_indices = new ArrayList<>();
            for(int i=0;i<detections.size();i++)
                detection_indices.add(i);
        }
        if(detection_indices.size() == 0 || track_indices.size() == 0){
            arrayObjects[0] = new ArrayList<>();
            arrayObjects[1] = track_indices;
            arrayObjects[2] = detection_indices;
            return arrayObjects;
        }

        Track[] mtrack = new Track[tracks.size()];
        for(int i=0;i<tracks.size();i++){
            mtrack[i] = tracks.get(i);
        }
        Detection[] mdetection = new Detection[detections.size()];
        for(int i=0;i<detections.size();i++){
            mdetection[i] = detections.get(i);
        }
        int[] mtrack_indices = new int[track_indices.size()];
        for(int i=0;i<track_indices.size();i++){
            mtrack_indices[i] = track_indices.get(i);
        }
        int[] mdetection_indices = new int[detection_indices.size()];
        for(int i=0;i<detection_indices.size();i++){
            mdetection_indices[i] = detection_indices.get(i);
        }
        float[][] cost_matrix;
        if(metric == null)
            cost_matrix = iou_matching.iou_cost(mtrack, mdetection, mtrack_indices, mdetection_indices);
        else
            cost_matrix = metric.gated_metric(tracks,detections,track_indices,detection_indices);
        for (int row = 0; row < cost_matrix.length; row++)
            for (int col = 0; col < cost_matrix[row].length; col++)
                if(cost_matrix[row][col] > max_distance)
                    cost_matrix[row][col] = max_distance + 1e-5f;

        Linear_assignment linear_assignment = new Linear_assignment();
        int[][] indices = linear_assignment.linear_assignment(cost_matrix);
        int flag;
        for (int col = 0; col < detection_indices.size(); col++) {
            flag = -1;
            for (int[] index : indices){
                if (col == index[1]) {
                    flag = 0;
                    break;
                }
            }
            if(flag == -1)
                unmatched_detections.add(detection_indices.get(col));
        }
        int flag0;
        for (int row = 0; row < track_indices.size(); row++) {
            flag0 = -1;
            for (int[] index : indices) {
                if (row == index[0]) {
                    flag0 = 0;
                    break;
                }
            }
            if(flag0 == -1)
                unmatched_tracks.add(track_indices.get(row));
        }
        for (int[] rc : indices) {
            int tidx = track_indices.get(rc[0]);
            int didx = detection_indices.get(rc[1]);
            if (cost_matrix[rc[0]][rc[1]] > max_distance) {
                unmatched_tracks.add(tidx);
                unmatched_detections.add(didx);
            } else {
                matches.add(new Pair<>(tidx, didx));
            }
        }
        arrayObjects [0] = matches;
        arrayObjects [1] = unmatched_tracks;
        arrayObjects [2] = unmatched_detections;
        return arrayObjects;
    }

    Object[] matching_cascade(Tracker.Gated_metric metric, float max_distance, int cascade_depth, List<Track> tracks, List<Detection> detections, List<Integer> track_indices, List<Integer> detection_indices) {
        List<Pair<Integer,Integer>> matches= new ArrayList<>();
        List<Integer> unmatched_tracks;
        Object[] arrayObjects = new Object[3];

        if(track_indices == null){
            track_indices = new ArrayList<>();
            for(int i=0;i<tracks.size();i++)
                track_indices.add(i);
        }
        if(detection_indices == null){
            detection_indices = new ArrayList<>();
            for(int i=0;i<detections.size();i++)
                detection_indices.add(i);
        }
        List<Integer> unmatched_detections = new ArrayList<>(detection_indices);
        for (int level = 0; level<cascade_depth; level++) {
            if (unmatched_detections.isEmpty())
                break;
            List<Integer> track_indices_l = new ArrayList<>();
            for(int t: track_indices) {
                if (tracks.get(t).time_since_update == 1 + level)
                    track_indices_l.add(t);
            }
            if (track_indices_l.isEmpty())
                continue;
            Object[] result = min_cost_matching(metric,max_distance, tracks, detections, track_indices_l, unmatched_detections);

            List<Pair<Integer,Integer>>  matches_l = (List<Pair<Integer, Integer>>) result[0];
            unmatched_detections = (List<Integer>)result[2];
            matches.addAll(matches_l);
        }
        Set setA = new HashSet(track_indices);

        Set setB = new HashSet();
        for(int i=0;i<matches.size();i++)
            setB.add(matches.get(i).first);

        setA.removeAll(setB);

        unmatched_tracks = new ArrayList<>(setA);

        arrayObjects [0] = matches;
        arrayObjects [1] = unmatched_tracks;
        arrayObjects [2] = unmatched_detections;
        return arrayObjects;
    }

    float[][] gate_cost_matrix(Kalman_filter kf, float[][] cost_matrix, List<Track> tracks, List<Detection> detections,
                               List<Integer> track_indices, List<Integer> detection_indices,float gated_cost, boolean only_position) {
        int gating_dim;
        if(only_position) gating_dim = 2; else gating_dim = 4;
        float gating_threshold = kf.chi2inv95[gating_dim];
        List<float[]> measurements = new ArrayList<>();
        for(int i:detection_indices)
            measurements.add(detections.get(i).to_xyah());

        float[][] tmeasurements = new float[measurements.size()][measurements.get(0).length];
        int c = 0;
        for(float[] m:measurements)
            tmeasurements[c++] = m;
        for(int i=0;i<track_indices.size();i++) {
            Track track = tracks.get(track_indices.get(i));
            float[] gating_distance = kf.gating_distance(track.mean, track.covariance, tmeasurements, only_position);
            for (int j = 0; j < cost_matrix[0].length; j++)
                if (gating_distance[j] > gating_threshold)
                    cost_matrix[i][j] = gated_cost;
        }
        return cost_matrix;
    }
}
