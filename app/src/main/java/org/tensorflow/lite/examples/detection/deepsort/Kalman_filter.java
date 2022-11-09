package org.tensorflow.lite.examples.detection.deepsort;

import android.util.Pair;
import java.util.Arrays;

class Kalman_filter {

    private float[][] np_eye(int x, int y){
        float[][] sample = new float[x][y];
        //Arrays.fill(sample,0);
        for (int i=0;i<x;i++){
            sample[i][i] = 1;
        }
        return sample;
    }
    private float[] np_zeros_like(float[] inp){
        float[] sample = new float[inp.length];
        Arrays.fill(sample,0.0f);
        return sample;
    }
    private float[] np_r(float[] first, float[] second) {
        float[] both = Arrays.copyOf(first, first.length+second.length);
        System.arraycopy(second, 0, both, first.length, second.length);
        return both;
    }
    private float[][] np_diag(float[] inp){
        float[][] sample = new float[inp.length][inp.length];
        //Arrays.fill(sample,0.0f);
        for (int i=0;i<inp.length;i++){
            sample[i][i] = inp[i];
        }
        return sample;
    }
    private float[] np_square(float[] sample){
        float[] samp = new float[sample.length];
        for (int i = 0; i<sample.length; i++){
            samp[i] = sample[i]*sample[i];
        }
        return samp;
    }
    private float[][] np_square(float[][] sample){
        float[][] samp = new float[sample.length][sample[0].length];
        for (int i = 0; i<sample.length; i++) {
            for (int j = 0; j < sample[0].length; j++) {
                samp[i][j] = sample[i][j] * sample[i][j];
            }
        }
        return samp;
    }
    private float[] np_dot_2_1(float[][] arr, float[] mean){
        float[] mean1 = new float[arr.length];
        for (int i=0;i<arr.length;i++){
            float x = 0.0f ;
            for(int j=0;j<mean.length;j++){
                x+=arr[i][j]*mean[j];
            }
            mean1[i]=x;
        }
        return mean1;
    }
    private float[] np_dot_1_2(float[] a,float[][] b){

        float[] mean1 = new float[b[0].length];
        for (int i=0;i<b[0].length;i++){
            float x = 0.0f ;
            for(int j=0;j<a.length;j++){
                x+=b[j][i]*a[j];
            }
            mean1[i]=x;
        }
        return mean1;
    }
    private float[][] transpose(float[][] matrix){
        float[][] trans = new float[matrix[0].length][matrix.length];
        for(int i=0;i<matrix[0].length;i++){
            for(int j=0;j<matrix.length;j++){
                trans[i][j]= matrix[j][i];
            }
        }
        return trans;
    }
    private float[][] np_add(float[][] a,float[][] b){
        float[][] c = new float[a.length][a[0].length];
        for(int i=0;i<a.length;i++){
            for(int j=0;j<a.length;j++){
                c[i][j]=a[i][j]+b[i][j];
            }
        }
        return c;
    }
    private float[] np_add(float[] a,float[] b){
        float[] c = new float[a.length];
        for(int i=0;i<a.length;i++){
            c[i]=a[i]+b[i];
        }
        return c;
    }
    private float[] np_sub(float[] a,float[] b){
        float[] c = new float[a.length];
        for(int i=0;i<a.length;i++){
            c[i]=a[i]-b[i];
        }
        return c;
    }
    private float[][] np_sub(float[][] a,float[][] b){
        float[][] c = new float[a.length][a[0].length];
        for(int i=0;i<a.length;i++){
            for(int j=0;j<a.length;j++){
                c[i][j]=a[i][j]-b[i][j];
            }
        }
        return c;
    }
    private float[][] np_sub_21(float[][] a,float[] b){
        float[][] c = new float[a.length][a[0].length];
        for(int i=0;i<a.length;i++){
            for(int j=0;j<a[0].length;j++)
                c[i][j] = a[i][j]-b[j];
        }
        return c;
    }
    private float[][] np_mat_mul(float[][] a,float[][] b){
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
    private float[][] np_multi_dot(float[][] a,float[][] b,float[][] c){
        float[][] res = np_mat_mul(a,b);
        res = np_mat_mul(res,c);
        return res;
    }
    private float[] np_sum(float[][] a){
        float[] sum = new float[a[0].length];
        for (float[] floats : a) {
            for (int j = 0; j < a[0].length; j++) {
                sum[j] += floats[j];
            }
        }
        return sum;
    }
    private float[][] chol_factor_fn(float[][] a,float flag){
        int m = a.length;
        float[][] l = new float[m][m];
        for(int i = 0; i< m;i++){
            for(int j = 0; j< m;j++)
                l[i][j]=flag;
        }
        for(int i = 0; i< m;i++){
            for(int k = 0; k < (i+1); k++){
                float sum = 0.0f;
                for(int j = 0; j < k; j++){
                    sum += l[i][j] * l[k][j];
                }
                if (i == k) l[i][i] = (float) Math.sqrt(a[i][i] - sum);
                else        l[i][k] = (float) (1.0 / l[k][k] * (a[i][k] - sum));
            }
        }
        return l;
    }
    private float[][] chol_solve_fn(float[][] L,float[][] X){
        int n = L.length;
        int nx = X[0].length;
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < nx; j++) {
                for (int i = 0; i < k ; i++)
                    X[k][j] -= X[i][j]*L[k][i];
                X[k][j] /= L[k][k];
            }
        }
        for (int k = n-1; k >= 0; k--) {
            for (int j = 0; j < nx; j++) {
                for (int i = k+1; i < n ; i++)
                    X[k][j] -= X[i][j]*L[i][k];
                X[k][j] /= L[k][k];
            }
        }
        return X;
    }
    private float[][] chol_solve_triangle(float[][] L,float[][] X){
        int n = L.length;
        int nx = X[0].length;
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < nx; j++) {
                for (int i = 0; i < k ; i++)
                    X[k][j] -= X[i][j]*L[k][i];
                X[k][j] /= L[k][k];
            }
        }
        return X;
    }
    float[] chi2inv95=new float[]{0.0f,3.8415f, 5.9915f, 7.8147f, 9.4f, 11.070f, 12.592f, 14.067f, 15.507f, 16.919f};
    private final float _std_weight_position;
    private final float _std_weight_velocity;
    private final float[][] _motion_mat;
    private final float[][] _update_mat;
    Kalman_filter() {
        int ndim = 4;
        int dt = 1;
        _motion_mat = np_eye(2*ndim,2*ndim);
        for (int i = 0; i < ndim; i++) {
            _motion_mat[i][ndim+i] = dt;
        }
        _update_mat = np_eye(ndim,2*ndim);
        _std_weight_position =  (1.f/50);
        _std_weight_velocity =  (1.f/200);
    }
    Pair<float[], float[][]> initiate(float[] measurement){
        float[] mean_pos = measurement.clone();
        float[] mean_vel = np_zeros_like(mean_pos);
        float[] mean =np_r(mean_pos,mean_vel);
        float[] std = new float[]{
                2 * _std_weight_position * measurement[3],
                2 * _std_weight_position * measurement[3],
                (float)1e-2,
                2 * _std_weight_position * measurement[3],
                10 * _std_weight_velocity * measurement[3],
                10 * _std_weight_velocity * measurement[3],
                (float)1e-5,
                10 * _std_weight_velocity * measurement[3]
        };
        float[][] covariance = np_diag(np_square(std));
        return new Pair<>(mean,covariance);
    }
    Pair<float[],float[][]> predict(float[] mean, float[][] covariance){
        float[] std_pos = {
                _std_weight_position * mean[3],
                _std_weight_position * mean[3],
                (float) 1e-2,
                _std_weight_position * mean[3]
        };
        float[] std_vel = {
                _std_weight_velocity * mean[3],
                _std_weight_velocity * mean[3],
                (float) 1e-5,
                _std_weight_velocity * mean[3]
        };
        float[][] motion_cov = np_diag(np_square(np_r(std_pos,std_vel)));
        mean = np_dot_2_1(_motion_mat,mean);
        covariance = np_add(np_multi_dot(_motion_mat,covariance,transpose(_motion_mat)),motion_cov);
        return new Pair<>(mean,covariance);
    }
    private Pair<float[],float[][]> project(float[] mean, float[][] covariance){
        float[] std = {
                _std_weight_position * mean[3],
                _std_weight_position * mean[3],
                (float)1e-1,
                _std_weight_position * mean[3]
        };
        mean = np_dot_2_1(_update_mat,mean);
        float[][] innovation_cov = np_diag(np_square(std));
        covariance = np_add(np_multi_dot(_update_mat,covariance,transpose(_update_mat)),innovation_cov);
        return new Pair<>(mean,covariance);
    }
    Pair<float[],float[][]> update(float[] mean, float[][] covariance, float[] measurement){
        float[] projected_mean;
        float[][] projected_cov;
        Pair<float[],float[][]> sample= project(mean,covariance);

        projected_mean = sample.first;
        projected_cov = sample.second;
        float[][] chol_factor = chol_factor_fn(projected_cov,1.0f);
        float[][] kalman_gain = transpose(chol_solve_fn(chol_factor,transpose(np_mat_mul(covariance,transpose(_update_mat)))));
        float[] innovation = np_sub(measurement,projected_mean);

        float[] new_mean = np_add(mean,np_dot_1_2(innovation,transpose(kalman_gain)));
        float[][] new_covariance =np_sub(covariance,np_multi_dot(kalman_gain,projected_cov,transpose(kalman_gain)));
        return new Pair<>(new_mean,new_covariance);

    }
    float[] gating_distance(float[] mean1, float[][] covariance1, float[][] measurements, boolean only_position){
        Pair<float[],float[][]> sample= project(mean1,covariance1);
        float[] mean = sample.first;
        float[][] covariance = sample.second;
        float[][] chol_factor = chol_factor_fn(covariance,0.0f);
        float[][] d = np_sub_21(measurements,mean);
        d = transpose(d);
        float[][] z = chol_solve_triangle(chol_factor,d);
        float[] squared_maha = np_sum(np_square(z));
        return squared_maha;
    }
}
