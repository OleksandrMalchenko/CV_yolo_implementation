package org.tensorflow.lite.examples.detection.deepsort;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

class NN_Matching {
    private float[][] list_to_array(ArrayList<float[]> sample){
        float[][] outarr = new float[sample.size()][sample.get(0).length];
        for(int i=0;i<sample.size();i++) outarr[i] = sample.get(i);
        return outarr;
    }
    private float[][] np_mat_mul(float[][] a, float[][] b){
        float[][] res = new float[a.length][b[0].length];
        for (int i=0;i<a.length;i++){
            for (int j=0;j<b[0].length;j++){
                res[i][j] = 0.0f;
                for (int k=0;k<a[0].length;k++){
                    res[i][j]+=a[i][k]*b[k][j];
                }
            }
        }
        return res;
    }
    private void clip(float[][] a){
        for(int i=0;i<a.length;i++){
            for(int j=0;j<a[0].length;j++){
                if(a[i][j]< (float) 0)
                    a[i][j]= (float) 0;
                else if(a[i][j]> Float.POSITIVE_INFINITY)
                    a[i][j]= Float.POSITIVE_INFINITY;
            }
        }
    }
    private float[] norm(float[][] x){

        float[] sum = new float[x.length];
        for(int i=0;i<x.length;i++){
            for(int j=0;j<x[0].length;j++)
                sum[i] += Math.pow(x[i][j],2);
        }
        for(int i=0;i<sum.length;i++){
            sum[i] = (float) Math.sqrt(sum[i]);
        }
        return sum;
    }
    private float[][] np_divide(float[][] a, float[] div){
        float[][] _a = new float[a.length][a[0].length];
        for(int i=0;i<a.length;i++){
            for(int j=0;j<a[0].length;j++)
                _a[i][j] = a[i][j] / div[i];
        }
        return _a;
    }
    private float[][] np_sub(float[][] a){
        float[][] c=new float[a.length][a[0].length];
        for(int i=0;i<a.length;i++){
            for(int j=0;j<a[0].length;j++)
                c[i][j]= 1.0f - a[i][j];
        }
        return c;
    }
    private float[] np_max(float[] a){
        float[] max = new float[a.length];
        for(int i=0;i<a.length;i++){
            if(a[i]< (float) 0)
                max[i] = 0.0f;
            else
                max[i] = a[i];
        }
        return max;
    }
    private float[][] transpose(float[][] matrix){
        float[][] trans = new float[matrix[0].length][matrix.length];
        for(int i=0;i<matrix[0].length;i++){
            for(int j=0;j<matrix.length;j++)
                trans[i][j] = matrix[j][i];
        }
        return trans;
    }
    private float[][] np_dot(float[][] a, float[][] b){
        float[][] res = new float[a.length][b[0].length];
        for (int i=0;i<a.length;i++){
            for (int j=0;j<b[0].length;j++){
                res[i][j] = 0.0f;
                for (int k=0;k<a[0].length;k++)
                    res[i][j]+=a[i][k]*b[k][j];
            }
        }
        return res;
    }
    private float[] min_distance(float[][] a){
        float[] res = new float[a[0].length];
        Arrays.fill(res,(float)1e+10);
        for (float[] floats : a) {
            for (int j = 0; j < a[0].length; j++) {
                if (res[j] > floats[j])
                    res[j] = floats[j];
            }
        }
        return res;
    }

    private float[][] _pdist(float[][] a, float[][] b) {
        if (a.length == 0 || b.length == 0)
            return new float[a.length][b.length];
        float[] a2 = new float[a.length];
        float[] b2 = new float[b.length];
        for (int i = 0; i < a.length; i++) {
            float asum = 0.0f;
            for (int j = 0; j < a[0].length; j++) asum += a[i][j] * a[i][j];
            a2[i] = asum;
        }
        for (int i = 0; i < b.length; i++) {
            float bsum = 0.0f;
            for (int j = 0; j < b[0].length; j++) bsum += b[i][j] * b[i][j];
            b2[i] = bsum;
        }
        float[][] b_T = transpose(b);
        float[][] a_dot_b_T = np_mat_mul(a,b_T);
        float[][] r2 = new float[a_dot_b_T.length][a_dot_b_T[0].length];
        for(int i=0;i<a.length;i++){
            for(int j=0;j<a[0].length;j++)
                r2[i][j] = -2 * a_dot_b_T[i][j] + a2[i] + b2[j];
        }
        clip(r2);
        return r2;
    }
    private float[][] _cosine_distance(float[][] a, float[][] b) {
        float[][] _a = np_divide(a,norm(a));
        float[][] _b = np_divide(b,norm(b));
        return np_sub(np_dot(_a,transpose(_b)));
    }
    private float[] _nn_euclidean_distance(float[][] x, float[][] y){
        float[][] distance = _pdist(x,y);
        return np_max(min_distance(distance));
    }
    private float[] _nn_cosine_distance(float[][] x, float[][] y){

        float[][] distance = _cosine_distance(x,y);
        return min_distance(distance);
    }

    class NearestNeighborDistanceMetrix{
        int _metric;
        float matching_threshold;
        int budget;
        Map<Integer, ArrayList<float[]>> samples;
        private void setdefault(int target,float[] feature){
            if(samples.containsKey(target))
                Objects.requireNonNull(samples.get(target)).add(feature);
            else {
                ArrayList<float[]> first = new ArrayList<>();
                first.add(feature);
                samples.put(target,first);
            }
        }

        NearestNeighborDistanceMetrix(String metric, float matching_threshold, int budget){
            if(metric.equals("euclidean"))
                _metric = 0;
            else if(metric.equals("cosine"))
                _metric = 1;
            else
                throw new Error("Invalid metric; must be either 'euclidean' or 'cosine'");
            this.matching_threshold = matching_threshold;
            this.budget = budget;
            samples = new HashMap<>();
        }

        void partial_fit(float[][] features, int[] targets, int[] active_targets){
            for(int i=0;i<features.length;i++){
                setdefault(targets[i],features[i]);
                if(budget > 0) {
                    for (int j : samples.keySet()){
                        if (Objects.requireNonNull(samples.get(j)).size() > budget) {
                            Objects.requireNonNull(samples.get(j)).remove(0);
                        }
                    }
                }
            }
            Map<Integer, ArrayList<float[]>> samples_temp = new HashMap<>();
            for(int k:active_targets)
                if(samples.containsKey(k)) samples_temp.put(k,samples.get(k));
            samples.clear();
            samples = samples_temp;
        }

        float[][] distance(float[][] features, int[] targets){
            float[][] cost_matrix = new float[targets.length][features.length];
            if(samples.size() == 0) return cost_matrix;
            for(int i=0;i<targets.length;i++){
                if(_metric == 0)
                    cost_matrix[i] = _nn_euclidean_distance(list_to_array(Objects.requireNonNull(samples.get(targets[i]))),features);
                else
                    cost_matrix[i] = _nn_cosine_distance(list_to_array(Objects.requireNonNull(samples.get(targets[i]))),features);
            }
            return cost_matrix;
        }
    }
}
