// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.tensorflow.lite.examples.detection.snpe;

import android.graphics.Rect;
import android.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

class Result {
    int classIndex;
    Float score;
    Rect rect;

    public Result(int cls, Float output, Rect rect) {
        this.classIndex = cls;
        this.score = output;
        this.rect = rect;
    }
};

public class PrePostProcessor {
    // for yolov5 model, no need to apply MEAN and STD
    static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
    static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

    // model input image size
    static int mInputWidth = 320;
    static int mInputHeight = 320;

    // model output is of size 25200*(num_of_class+5)
    private static int mOutputRow = 6300; // as decided by the YOLOv5 model for input image of size 640*640
    private static int mOutputColumn = 6; // left, top, right, bottom, score and 80 class probability
    private static float mThreshold = 0.30f; // score above which a detection is generated
    private static int mNmsLimit = 15;

    static String[] mClasses;

    // The two methods nonMaxSuppression and IOU below are ported from https://github.com/hollance/YOLO-CoreML-MPSNNGraph/blob/master/Common/Helpers.swift
    /**
     Removes bounding boxes that overlap too much with other boxes that have
     a higher score.
     - Parameters:
     - boxes: an array of bounding boxes and their scores
     - limit: the maximum number of boxes that will be selected
     - threshold: used to decide whether boxes overlap too much
     */
    static ArrayList<Result> nonMaxSuppression(ArrayList<Result> boxes, int limit, float threshold) {

        // Do an argsort on the confidence scores, from high to low.
        Collections.sort(boxes,
                new Comparator<Result>() {
                    @Override
                    public int compare(Result o1, Result o2) {
                        return o1.score.compareTo(o2.score);
                    }
                });

        ArrayList<Result> selected = new ArrayList<>();
        boolean[] active = new boolean[boxes.size()];
        Arrays.fill(active, true);
        int numActive = active.length;

        // The algorithm is simple: Start with the box that has the highest score.
        // Remove any remaining boxes that overlap it more than the given threshold
        // amount. If there are any boxes left (i.e. these did not overlap with any
        // previous boxes), then repeat this procedure, until no more boxes remain
        // or the limit has been reached.
        boolean done = false;
        for (int i=0; i<boxes.size() && !done; i++) {
            if (active[i]) {
                Result boxA = boxes.get(i);
                selected.add(boxA);
                if (selected.size() >= limit) break;

                for (int j=i+1; j<boxes.size(); j++) {
                    if (active[j]) {
                        Result boxB = boxes.get(j);
                        if (IOU(boxA.rect, boxB.rect) > threshold) {
                            active[j] = false;
                            numActive -= 1;
                            if (numActive <= 0) {
                                done = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        return selected;
    }

    /**
     Computes intersection-over-union overlap between two bounding boxes.
     */
    static float IOU(Rect a, Rect b) {
        float areaA = (a.right - a.left) * (a.bottom - a.top);
        if (areaA <= 0.0) return 0.0f;

        float areaB = (b.right - b.left) * (b.bottom - b.top);
        if (areaB <= 0.0) return 0.0f;

        float intersectionMinX = Math.max(a.left, b.left);
        float intersectionMinY = Math.max(a.top, b.top);
        float intersectionMaxX = Math.min(a.right, b.right);
        float intersectionMaxY = Math.min(a.bottom, b.bottom);
        float intersectionArea = Math.max(intersectionMaxY - intersectionMinY, 0) *
                Math.max(intersectionMaxX - intersectionMinX, 0);
        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    static ArrayList<Result> outputsToNMSPredictions(float[] outputs, float imgScaleX, float imgScaleY, float ivScaleX, float ivScaleY, float startX, float startY) {
        ArrayList<Result> results = new ArrayList<>();
//        Log.d("snpe_engine", "input size: " + outputs.length + ", " + imgScaleX + ", " + imgScaleY);
        for (int c=0;c<outputs.length;c+=6) {

            if (outputs[c+4]>=mThreshold) {
                float cx = outputs[c];
                float cy = outputs[c+1];
                float w = outputs[c+2];
                float h = outputs[c+3];
                int gridX, gridY;
                int anchor_gridX, anchor_gridY;
                int[] anchorX = {10,16,33,30,62,59,116,156,373};
                int[] anchorY = {13,30,23,61,45,119,90,198,326};
                int[] num_filters = {4800,1200,300};
                int[] filter_size = {40,20,10};
                int stride;
                int ci = (int)(c/6);
                if (ci<num_filters[0]) {
                    gridX = (ci%(filter_size[0]*filter_size[0]))%filter_size[0];
                    gridY = (int)((ci%(filter_size[0]*filter_size[0]))/filter_size[0]);
                    anchor_gridX = anchorX[((int)(ci/(filter_size[0]*filter_size[0])))];
                    anchor_gridY = anchorY[((int)(ci/(filter_size[0]*filter_size[0])))];
                    stride = 8;
                } else if (ci>=num_filters[0]&&ci<num_filters[0]+num_filters[1]) {
                    gridX = ((ci-num_filters[0])%(filter_size[1]*filter_size[1]))%filter_size[1];
                    gridY = (int)(((ci-num_filters[0])%(filter_size[1]*filter_size[1]))/filter_size[1]);
                    anchor_gridX = anchorX[(int)((ci-num_filters[0])/(filter_size[1]*filter_size[1]))+3];
                    anchor_gridY = anchorY[(int)((ci-num_filters[0])/(filter_size[1]*filter_size[1]))+3];
                    stride = 16;
                } else {
                    gridX = ((ci-num_filters[1]-num_filters[0])%(filter_size[2]*filter_size[2]))%filter_size[2];
                    gridY = (int)(((ci-num_filters[1]-num_filters[0])%(filter_size[2]*filter_size[2]))/filter_size[2]);
                    anchor_gridX = anchorX[(int)((ci-num_filters[1]-num_filters[0])/(filter_size[2]*filter_size[2]))+6];
                    anchor_gridY = anchorY[(int)((ci-num_filters[1]-num_filters[0])/(filter_size[2]*filter_size[2]))+6];
                    stride = 32;
                }
                cx = (float)(cx*2-0.5+gridX)*stride;
                cy = (float)(cy*2-0.5+gridY)*stride;
                w = w*2*w*2*anchor_gridX;
                h = h*2*h*2*anchor_gridY;
                float left = Math.max(Math.min(imgScaleX * (cx-w/2), 319), 0);
                float top = Math.max(Math.min(imgScaleY * (cy-h/2), 319), 0);
                float right = Math.max(Math.min(imgScaleX * (cx+w/2), 319), 0);
                float bottom = Math.max(Math.min(imgScaleY * (cy+h/2), 319), 0);
                float obj_conf = outputs[c+4];
                Log.d("snpe_engine", "3333333: " + left + ", " + top +
                        ", " + right + ", " + bottom);
                Rect rect = new Rect((int)(startX+left), (int)(startY+top), (int)(startX+right), (int)(startY+bottom));
                Result result = new Result(0, obj_conf, rect);
                results.add(result);
            }
        }

        return nonMaxSuppression(results, mNmsLimit, mThreshold);
    }
}
