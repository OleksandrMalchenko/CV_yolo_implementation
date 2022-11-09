package org.tensorflow.lite.examples.detection.deepsort;

import android.util.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Tracker{

    private final NN_Matching.NearestNeighborDistanceMetrix mmetric_;
    private final float mmax_iou_distance_;
    private final Integer mmax_age_;
    private final Integer mn_init_;
    private final Kalman_filter mkf_;
    public List<Track> mtracks_;
    private Integer mnext_id_;
    private final LinearAssignment linearAssignment;

    interface Gated_metric {
        float[][] gated_metric(List<Track> tracks, List<Detection> dets, List<Integer> track_indices, List<Integer> detection_indices);
    }

    class Metric implements Gated_metric {
        @Override
        public float[][] gated_metric(List<Track> tracks, List<Detection> dets, List<Integer> track_indices, List<Integer> detection_indices) {
            float[][] features = new float[detection_indices.size()][1024];
            int[] targets = new int[track_indices.size()];

            for(int i=0;i<detection_indices.size();i++)
                features[i] = dets.get(detection_indices.get(i)).feature;
            for(int i=0;i<track_indices.size();i++)
                targets[i] = tracks.get(track_indices.get(i)).track_id;
            float[][] cost_matrix = mmetric_.distance(features, targets);
            cost_matrix = linearAssignment.gate_cost_matrix(
                    mkf_, cost_matrix, tracks, dets, track_indices, detection_indices,(float)1e+5,false);
            return cost_matrix;
        }
    }

    Tracker(NN_Matching.NearestNeighborDistanceMetrix metric, float max_iou_distance, Integer max_age, Integer n_init) {
        mmetric_ = metric;
        mmax_iou_distance_ = max_iou_distance;
        mmax_age_ = max_age;
        mn_init_ = n_init;
        mkf_ = new Kalman_filter();
        mtracks_ = new ArrayList<>();
        mnext_id_ = 1;
        linearAssignment = new LinearAssignment();
    }

    void predict() {
        for (Track t : mtracks_) {
            t.predict(mkf_);
        }
    }

    void update(List<Detection> detections) {
        Object[] result = _match(detections);
        List<Pair<Integer, Integer>> matches = (List<Pair<Integer, Integer>>) result[0];
        List<Integer> unmatched_tracks = (List<Integer>) result[1];
        List<Integer> unmatched_detections = (List<Integer>) result[2];

        for(Pair<Integer,Integer> track_det_ids:matches){
            if(mtracks_.size() > 0)
                mtracks_.get(track_det_ids.first).update(mkf_,detections.get(track_det_ids.second));
        }
        for (int track_idx : unmatched_tracks) {
            if(mtracks_.size() > 0)
                mtracks_.get(track_idx).mark_missed();
        }

        for (int detection_idx : unmatched_detections) {
            _initiate_track(detections.get(detection_idx));
        }

        List<Track> tmptracks = new ArrayList<>();

        for (Track t : mtracks_) {
            if (!t.is_deleted())
                tmptracks.add(t);
        }
        mtracks_.clear();
        mtracks_.addAll(tmptracks);

        List<Integer> active_targets = new ArrayList<>();

        for (Track t : mtracks_) {
            if (t.is_confirmed())
                active_targets.add(t.track_id);
        }

        List<float[]> features = new ArrayList<>();
        List<Integer> targets = new ArrayList<>();

        for (Track track : mtracks_) {
            if (!track.is_confirmed())
                continue;
            features.addAll(track.features);

            for(float[] ignored :track.features)
                targets.add(track.track_id);


            float[] last_ft = track.features.get(track.features.size()-1);
            track.features.clear();
            track.features.add(last_ft);
        }

        int[] mtargets = new int[targets.size()];
        int[] mactive_targets = new int[active_targets.size()];
        float[][] mfeatures = new float[features.size()][1024];

        for(int i=0;i<targets.size();i++)
            mtargets[i] = targets.get(i);

        for(int i=0;i<active_targets.size();i++)
            mactive_targets[i] = active_targets.get(i);

        for(int i=0;i<features.size();i++)
            System.arraycopy(features.get(i), 0, mfeatures[i], 0, features.get(0).length);

        mmetric_.partial_fit(mfeatures, mtargets, mactive_targets);
    }

    private Object[] _match(List<Detection> detections) {
        List<Integer> confirmed_tracks = new ArrayList<>();
        List<Integer> unconfirmed_tracks = new ArrayList<>();

        for(int i=0;i<mtracks_.size();i++)
            if (mtracks_.get(i).is_confirmed())
                confirmed_tracks.add(i);
        for(int i=0;i<mtracks_.size();i++)
            if(!mtracks_.get(i).is_confirmed())
                unconfirmed_tracks.add(i);
        Gated_metric metric = new Metric();
        List<Integer> dummy = null;
        Object[] result = linearAssignment.matching_cascade(metric,mmetric_.matching_threshold, mmax_age_,
        mtracks_, detections, confirmed_tracks, dummy);

        List<Map.Entry<Integer, Integer>> matches_a;
        List<Integer> unmatched_tracks_a;
        List<Integer> unmatched_detections;
        List<Integer> iou_track_candidates = new ArrayList<>();
        List<Integer> remaining_tracks = new ArrayList<>();
        List<Map.Entry<Integer, Integer>> matches_b;
        List<Map.Entry<Integer, Integer>> matches;

        if(result[0] == null) matches_a = new ArrayList<>();
        else matches_a = (List<Map.Entry<Integer, Integer>>) result[0];

        if(result[1] == null) unmatched_tracks_a = new ArrayList<>();
        else unmatched_tracks_a = (List<Integer>) result[1];

        if(result[2] == null) unmatched_detections = new ArrayList<>();
        else unmatched_detections = (List<Integer>) result[2];

        for(int k:unmatched_tracks_a){
            if(mtracks_.size() >0 && mtracks_.get(k).time_since_update == 1)
                remaining_tracks.add(k);
        }

        iou_track_candidates.addAll(unconfirmed_tracks);
        iou_track_candidates.addAll(remaining_tracks);
        List<Integer> temp = new ArrayList<>();

        for(int k:unmatched_tracks_a){
            if(mtracks_.size() > 0 && mtracks_.get(k).time_since_update != 1)
                temp.add(k);
        }
        unmatched_tracks_a.clear();
        unmatched_tracks_a.addAll(temp);


        Object[] LAresult = linearAssignment.min_cost_matching(null,mmax_iou_distance_, mtracks_,
                detections, iou_track_candidates, unmatched_detections);


        List<Integer> unmatched_tracks_b;

        if(LAresult[0] == null) matches_b = new ArrayList<>();
        else matches_b = (List<Map.Entry<Integer, Integer>>) LAresult[0];

        if(LAresult[1] == null) unmatched_tracks_b = new ArrayList<>();
        else unmatched_tracks_b = (List<Integer>) LAresult[1];

        if(LAresult[2] == null) unmatched_detections = new ArrayList<>();
        else unmatched_detections = (List<Integer>) LAresult[2];

        matches = new ArrayList<>(matches_a);

        matches.addAll(matches_b);

        List<Integer> unmatched_tracks = new ArrayList<>(unmatched_tracks_a);

        unmatched_tracks.addAll(unmatched_tracks_b);

        Object[] arrayObjects = new Object[3];
        arrayObjects[0] = matches;
        arrayObjects[1] = unmatched_tracks;
        arrayObjects[2] = unmatched_detections;


        return arrayObjects;
    }

    private void _initiate_track(Detection detection) {
        Pair<float[],float[][]> result = mkf_.initiate(detection.to_xyah());
        float[] mean = result.first;
        float[][] covariance = result.second;
        Track track = new Track(mean, covariance, mnext_id_, mn_init_, mmax_age_, detection.feature);
        mtracks_.add(track);
        mnext_id_ += 1;
    }
}